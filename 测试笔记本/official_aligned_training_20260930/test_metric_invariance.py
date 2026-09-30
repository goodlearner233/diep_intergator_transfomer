"""Test new entry's pooled E/F/S metrics, evaluation padding, resume guards and cosine.

Synthetic fixed predictions intentionally isolate aggregation from model accuracy.
No production files are changed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import types

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training_official_aligned"))

import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.data import collate_fn_pes
from train import AlignedModule, EvaluationBatchSampler, MaskedEvaluationDataset, FORMAT


COUNTS = [2, 3, 5, 1, 4]


class SyntheticDataset(Dataset):
    def __len__(self):
        return len(COUNTS)

    def __getitem__(self, i):
        count = COUNTS[i]
        true_energy = -2.0 * count
        pred_energy = true_energy + count * (i - 1.2) * 0.7
        true_force = torch.zeros(count, 3)
        pred_force = torch.arange(count * 3, dtype=torch.float32).reshape(count, 3) * (i + 1) * 0.11 - 0.6
        true_stress = torch.zeros(3, 3)
        pred_stress = torch.arange(9, dtype=torch.float32).reshape(3, 3) * 0.1 * (i + 1) - 0.3
        graph = Data(num_nodes=count, sample_idx=torch.tensor([i]),
                     pred_energy=torch.tensor([pred_energy]), pred_force=pred_force,
                     pred_stress=pred_stress)
        labels = {"energies": true_energy, "forces": true_force, "stresses": true_stress}
        return graph, torch.eye(3), torch.tensor([0.0, 0.0]), labels


def make_module():
    module = AlignedModule(1.0, [0.0] * len(DEFAULT_ELEMENTS), "metric-test-signature")
    module._trainer = types.SimpleNamespace(sanity_checking=False)
    def fixed_forward(self, graph, lattice, state):
        return graph.pred_energy, graph.pred_force, graph.pred_stress, torch.zeros(1)
    module.forward = types.MethodType(fixed_forward, module)
    module.logged = {}
    def capture(self, values, **kwargs):
        self.logged.update({key: float(value) for key, value in values.items()})
    module.log_dict = types.MethodType(capture, module)
    module.start_phase("val")
    return module


def evaluate_packs(packs):
    dataset = MaskedEvaluationDataset(SyntheticDataset())
    module = make_module()
    visits = []
    for pack in packs:
        batch = collate_fn_pes([dataset[i] for i in pack])
        output = module.step(batch, "val")
        visits += [{"source_id": int(i), "metric_weight": bool(valid)}
                   for i, valid in zip(output["indices"], output["metric_weight"])]
    return module, visits


def main():
    torch.set_num_threads(1)
    folder = HERE / f"metric_invariance_{time.time_ns()}"
    folder.mkdir()
    configurations = {}
    reference = None
    for budget, world in [(5, 1), (6, 1), (15, 1), (5, 2), (6, 2), (15, 2)]:
        modules, all_visits, rank_pack_counts = [], [], []
        for rank in range(world):
            packs = list(EvaluationBatchSampler(COUNTS, budget, rank, world))
            rank_pack_counts.append(len(packs))
            module, visits = evaluate_packs(packs)
            modules.append(module)
            all_visits += visits
        assert len(set(rank_pack_counts)) == 1
        # Exact same summation as the distributed all_reduce used by end_phase.
        combined = sum((m.phase_stats["val"] for m in modules), torch.zeros(23, dtype=torch.float64))
        modules[0].phase_stats["val"] = combined
        modules[0].end_phase("val")
        metrics = modules[0].logged
        valid_ids = [v["source_id"] for v in all_visits if v["metric_weight"]]
        assert sorted(valid_ids) == list(range(len(COUNTS))), valid_ids
        assert metrics["val_structure_visits"] == len(COUNTS)
        pooled = {key: value for key, value in metrics.items() if "_pooled_" in key}
        assert len(pooled) == 10
        if reference is None:
            reference = pooled
        else:
            for key, expected in reference.items():
                torch.testing.assert_close(torch.tensor(pooled[key], dtype=torch.float64),
                                           torch.tensor(expected, dtype=torch.float64),
                                           rtol=0.000000000001, atol=0.000000000001)
        configurations[f"budget{budget}_world{world}"] = {
            "rank_forward_counts": rank_pack_counts,
            "raw_structure_visits": len(all_visits),
            "real_structure_visits": len(valid_ids),
            "excluded_padding_visits": len(all_visits) - len(valid_ids),
            "metrics": metrics}

    # Independently calculate pooled metrics from every true component once.
    dataset = SyntheticDataset()
    all_records = [dataset[i] for i in range(len(dataset))]
    energy_errors = torch.cat([(record[0].pred_energy - record[3]["energies"]) / COUNTS[i]
                               for i, record in enumerate(all_records)]).double()
    force_errors = torch.cat([r[0].pred_force - r[3]["forces"] for r in all_records]).double()
    stress_errors = torch.cat([r[0].pred_stress - r[3]["stresses"] for r in all_records]).double()
    direct = {}
    for name, errors in zip(("Energy", "Force", "Stress"), (energy_errors, force_errors, stress_errors)):
        absolute = errors.abs()
        vals = {"MAE": absolute.mean(), "RMSE": errors.square().mean().sqrt(),
                "Huber": torch.where(absolute <= 1.0, 0.5 * errors.square(), absolute - 0.5).mean()}
        for metric, value in vals.items():
            key = f"val_pooled_{name}_{metric}"
            # FP32 order of division/subtraction can change energy last digits.
            assert abs(reference[key] - float(value)) < 0.000001, (key, reference[key], value)
            direct[key] = float(value)

    module = make_module()
    rejected = {}
    for label, checkpoint in {
        "old_original_checkpoint": {"state_dict": {}},
        "mismatched_data_or_configuration": {"aligned_format": FORMAT, "aligned_signature": "wrong"},
    }.items():
        try:
            module.on_load_checkpoint(checkpoint)
        except ValueError as error:
            rejected[label] = str(error)
        else:
            raise AssertionError(f"Did not reject {label}")
    module.on_load_checkpoint({"aligned_format": FORMAT, "aligned_signature": "metric-test-signature"})
    configured = module.configure_optimizers()
    opt = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    rates = {"0": opt.param_groups[0]["lr"]}
    for epoch in range(1, 201):
        opt.step()
        scheduler.step()
        if epoch in (1, 100, 199, 200):
            rates[str(epoch)] = opt.param_groups[0]["lr"]
    assert abs(rates["0"] - 0.001) < 0.000000000001
    assert abs(rates["100"] - 0.000505) < 0.000000000001
    assert abs(rates["200"] - 0.00001) < 0.000000000001
    assert scheduler.last_epoch == 200
    result = {"passed": True, "synthetic_fixed_predictions": True,
              "configurations": configurations, "independent_component_metrics": direct,
              "checkpoint_rejections": rejected, "cosine_learning_rates": rates,
              "scope": "Aggregation/padding protocol test, not model accuracy; rank statistics summed locally exactly as all_reduce"}
    (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(folder), "passed": True,
                      "configurations": list(configurations), "checkpoint_rejections": rejected,
                      "cosine_learning_rates": rates}, indent=2))


if __name__ == "__main__":
    main()
