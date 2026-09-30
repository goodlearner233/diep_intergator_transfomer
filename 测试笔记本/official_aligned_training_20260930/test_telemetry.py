"""Actual Lightning update tests for the independent official-aligned observer."""
from __future__ import annotations

import copy
import gzip
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("aligned_telemetry", ROOT / "training_official_aligned" / "telemetry.py")
telemetry_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry_module)
Telemetry = telemetry_module.Telemetry


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line, parse_constant=lambda text: (_ for _ in ()).throw(ValueError(text))) for line in stream]


class Toy(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 1)
        self.automatic_optimization = True
        self.energy_weight, self.force_weight, self.stress_weight = 1.0, 1.0, 0.1
        self.loss_params = {"delta": 1.0}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True, weight_decay=0.00001)

    def calculate(self, batch):
        graph, x = batch
        counts = torch.tensor([1, 2], device=self.device)
        energy = self.linear(x).reshape(-1)
        force = energy.repeat_interleave(counts).unsqueeze(-1).expand(-1, 3)
        stress = energy[:, None, None].expand(-1, 3, 3)
        labels = tuple(torch.zeros_like(v) for v in (energy, force, stress))
        loss = F.huber_loss(energy / counts, labels[0]) + F.huber_loss(force, labels[1]) + 0.1 * F.huber_loss(stress, labels[2])
        return {"loss": loss, "preds": (energy, force, stress), "labels": labels,
                "num_atoms": counts, "indices": graph.sample_idx, "metric_weight": graph.metric_weight}

    def training_step(self, batch, batch_idx):
        return self.calculate(batch)

    def validation_step(self, batch, batch_idx):
        return self.calculate(batch)

    def test_step(self, batch, batch_idx):
        return self.calculate(batch)


def collate(items):
    index = int(items[0])
    graph = SimpleNamespace(sample_idx=torch.tensor([index * 2, index * 2 + 1]),
                            edge_index=torch.tensor([[0, 1, 2], [0, 2, 1]]),
                            metric_weight=torch.tensor([1.0, 0.0 if index == 99 else 1.0]))
    x = torch.tensor([[20.0 + index, -40.0], [60.0, 20.0 - index]])
    return graph, x


class TelemetryTests(unittest.TestCase):
    def run_training(self, directory, enabled, accelerator="cpu"):
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(123)
        model = Toy()
        callback = Telemetry(directory, enabled=enabled, coordinate_count=128, flush_every=1)
        trainer = L.Trainer(accelerator=accelerator, devices=1, max_epochs=2,
                            logger=False, enable_checkpointing=False, enable_progress_bar=False,
                            enable_model_summary=False, callbacks=[callback],
                            gradient_clip_val=2.0, accumulate_grad_batches=2,
                            num_sanity_val_steps=1, deterministic=True)
        train = DataLoader(list(range(6)), batch_size=1, collate_fn=collate)
        val = DataLoader([99], batch_size=1, collate_fn=collate)
        trainer.fit(model, train_dataloaders=train, val_dataloaders=val)
        fit_session = callback.session_dir
        params = copy.deepcopy(model.state_dict())
        opt = copy.deepcopy(trainer.optimizers[0].state_dict())
        trainer.test(model, dataloaders=val)
        return params, opt, fit_session, callback.session_dir

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_observer_equality_and_allocator_records(self):
        with tempfile.TemporaryDirectory() as root:
            off, offopt, _, _ = self.run_training(Path(root) / "off", False, "gpu")
            on, onopt, session, _ = self.run_training(Path(root) / "on", True, "gpu")
            for key in off:
                self.assertTrue(torch.equal(off[key], on[key]), key)
            for pi in offopt["state"]:
                for key in offopt["state"][pi]:
                    self.assertTrue(torch.equal(offopt["state"][pi][key], onopt["state"][pi][key]), key)
            batches = rows(session / "batches.jsonl.gz")
            self.assertTrue(all(b["cuda_allocator"]["allocated_bytes"] > 0 for b in batches))
            self.assertTrue(all(b["cuda_allocator"]["max_allocated_bytes_since_batch_start"] >= b["cuda_allocator"]["allocated_bytes"] for b in batches))

    def test_real_updates_equal_and_observer_records_correct_amsgrad(self):
        with tempfile.TemporaryDirectory() as root:
            off, offopt, _, _ = self.run_training(Path(root) / "off", False)
            on, onopt, session, test_session = self.run_training(Path(root) / "on", True)
            for key in off:
                self.assertTrue(torch.equal(off[key], on[key]), key)
            for pi in offopt["state"]:
                for key in offopt["state"][pi]:
                    self.assertTrue(torch.equal(offopt["state"][pi][key], onopt["state"][pi][key]), key)
            self.assertEqual(offopt["param_groups"], onopt["param_groups"])
            updates = rows(session / "optimizer_updates.jsonl.gz")
            coords = rows(session / "coordinates.jsonl.gz")
            batches = rows(session / "batches.jsonl.gz")
            self.assertEqual(len(updates), 6)
            self.assertEqual(len(coords), 6)
            self.assertEqual(sum(b["phase"] == "train" for b in batches), 12)
            self.assertEqual(sum(b["phase"] == "sanity_val" for b in batches), 1)
            self.assertTrue(all(b["cuda_allocator"]["allocated_bytes"] is None for b in batches))
            self.assertTrue(all(b["finite"] for b in batches))
            self.assertTrue(any(b["padding_sample_count"] == 1 for b in batches))
            self.assertNotEqual(session, test_session)
            self.assertEqual(rows(test_session / "batches.jsonl.gz")[0]["phase"], "test")
            for update, coordinate in zip(updates, coords):
                self.assertGreater(update["gradient_l2_before_clip"], 2)
                self.assertLessEqual(update["gradient_l2_after_clip"], 2.000001)
                self.assertEqual(len(update["rank0_microbatches"]), 2)
                self.assertTrue(update["finite"])
                before, after = coordinate["before"], coordinate["after"]
                for i, step in enumerate(after["steps"]):
                    expected = math.sqrt(after["max_v"][i] / (1 - 0.999 ** step)) + 0.00000001
                    self.assertAlmostEqual(after["denominator"][i], expected, delta=0.000001)
                    self.assertGreaterEqual(after["max_v"][i], after["v"][i])
                    self.assertAlmostEqual(after["parameter"][i] - before["parameter"][i], coordinate["parameter_delta"][i], places=12)
                    self.assertAlmostEqual(coordinate["adamw_decay_delta_formula"][i], -0.001 * 0.00001 * before["parameter"][i], places=15)
                    self.assertAlmostEqual(coordinate["remaining_delta_including_rounding"][i] + coordinate["adamw_decay_delta_formula"][i], coordinate["parameter_delta"][i], places=15)

    def test_rank_sessions_do_not_collide_and_no_rng_consumption(self):
        with tempfile.TemporaryDirectory() as root:
            model = Toy()
            optimizer = model.configure_optimizers()
            locations = []
            for rank in (0, 1, 0):
                trainer = SimpleNamespace(global_rank=rank, world_size=2, precision="32-true", optimizers=[optimizer])
                observer = Telemetry(root)
                rng_before = torch.get_rng_state().clone()
                observer.setup(trainer, model, "fit")
                observer.on_fit_start(trainer, model)
                self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
                locations.append(observer.session_dir)
                if rank == 1:
                    self.assertFalse((observer.session_dir / "optimizer_updates.jsonl.gz").exists())
                observer.close()
            self.assertEqual(len(set(locations)), 3)

    def test_triplets_match_line_graph_and_strict_nonfinite_json(self):
        graph = SimpleNamespace(
            pos=torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 2, 0], [10.0, 0, 0]]),
            edge_index=torch.tensor([[0, 0, 0, 1, 1, 2], [1, 2, 3, 0, 2, 0]]),
            batch=torch.tensor([0, 0, 0, 1]), num_graphs=2)
        callback = Telemetry("unused")
        callback._capture_workload(SimpleNamespace(threebody_cutoff=4.0), (graph,), {})
        self.assertEqual(callback._workload["n_triplets"], 4)
        self.assertEqual(callback._workload["triplets_per_structure"], [4, 0])
        # Match the actual repository routine, not a second copy of the formula.
        sys.path.insert(0, str(ROOT / "src"))
        from matgl.graph._compute import create_line_graph
        src, dst = graph.edge_index
        vector = graph.pos[dst] - graph.pos[src]
        actual = create_line_graph(graph.edge_index, vector.norm(dim=1), vector, None, 4, 4.0)
        self.assertEqual(callback._workload["n_triplets"], actual["line_edge_index"].shape[1])
        bad = {"loss": float("inf"), "values": [float("nan"), 1.0]}
        self.assertFalse(telemetry_module._finite(bad))
        self.assertEqual(json.loads(json.dumps(telemetry_module._safe(bad), allow_nan=False)), {"loss": None, "values": [None, 1.0]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
