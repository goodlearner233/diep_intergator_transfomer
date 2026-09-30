"""Actual Lightning 200-epoch scheduler/resume audit using the production hooks.

Only build_potential is replaced by a tiny differentiable E/F/S predictor to make
400 total epochs inexpensive. Production step/metrics/optimizer/checkpoint hooks,
gradient accumulation and clipping execute unchanged. This is control-flow proof,
not an accuracy or full-model speed benchmark.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training_official_aligned"))

import lightning as L
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from matgl.graph.data import collate_fn_pes
import train as entry


class TinyPotential(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.9, -0.6, 0.3]))

    def forward(self, g, lat, state_attr):
        count = torch.bincount(g.batch)
        energy = self.weight[0] * count.float()
        force = self.weight[1] * torch.ones(g.num_nodes, 3)
        stress = self.weight[2] * torch.ones(3 * len(count), 3)
        return energy, force, stress


class TinyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        count = index % 3 + 1
        graph = Data(num_nodes=count, sample_idx=torch.tensor([index]))
        labels = {"energies": -0.15 * count,
                  "forces": torch.full((count, 3), 0.15),
                  "stresses": torch.full((3, 3), -0.15)}
        return graph, torch.eye(3), torch.zeros(2), labels


class TinyData(L.LightningDataModule):
    train_sampler = None

    def train_dataloader(self):
        return DataLoader(TinyDataset(), batch_size=2, shuffle=False, collate_fn=collate_fn_pes)

    def val_dataloader(self):
        return DataLoader(TinyDataset(), batch_size=3, shuffle=False, collate_fn=collate_fn_pes)


class Trace(Callback):
    def __init__(self):
        self.rows = []

    def on_train_epoch_start(self, trainer, module):
        opt = trainer.optimizers[0]
        scheduler = trainer.lr_scheduler_configs[0].scheduler
        self.rows.append({"epoch": trainer.current_epoch, "global_step": trainer.global_step,
                          "lr": opt.param_groups[0]["lr"], "scheduler_last_epoch": scheduler.last_epoch})


class TracedModule(entry.AlignedModule):
    def __init__(self):
        super().__init__(1.0, [0.0], "scheduler-control-flow-audit", epochs=200)
        self.scheduler_calls = []

    def lr_scheduler_step(self, scheduler, metric):
        before = scheduler.last_epoch
        super().lr_scheduler_step(scheduler, metric)
        self.scheduler_calls.append({"epoch": self.current_epoch, "before": before,
                                     "after": scheduler.last_epoch})


def run(folder, stop, resume=None, emergency_every=7):
    folder.mkdir(exist_ok=True)
    L.seed_everything(42)
    model = TracedModule()
    trace = Trace()
    best = ModelCheckpoint(dirpath=folder / "checkpoints", filename="best-{epoch:03d}-{step}",
                           monitor="val_Total_Loss", mode="min", save_top_k=1,
                           save_on_train_epoch_end=True)
    history = ModelCheckpoint(dirpath=folder / "checkpoints", filename="epoch-{epoch:03d}-step-{step}",
                              auto_insert_metric_name=False, save_top_k=-1, save_last=True,
                              every_n_epochs=1, save_on_train_epoch_end=True)
    emergency = ModelCheckpoint(dirpath=folder / "checkpoints" / "emergency", filename="step-latest",
                                every_n_train_steps=emergency_every, save_top_k=1, enable_version_counter=False,
                                save_on_train_epoch_end=False)
    trainer = L.Trainer(default_root_dir=str(folder), logger=CSVLogger(str(folder), name="logs"),
                        callbacks=[trace, entry.FiniteGradients(), LearningRateMonitor(logging_interval="epoch"),
                                   best, history, emergency], accelerator="cpu", devices=1,
                        max_epochs=stop, accumulate_grad_batches=2, precision="32-true",
                        gradient_clip_val=2.0, gradient_clip_algorithm="norm", inference_mode=False,
                        use_distributed_sampler=False, num_sanity_val_steps=2, log_every_n_steps=1,
                        enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(model, datamodule=TinyData(), ckpt_path=str(resume) if resume else None)
    checkpoint = torch.load(history.last_model_path, map_location="cpu", weights_only=False)
    return model, trace.rows, checkpoint, best.best_model_path


def assert_same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b), (a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b, (a, b)


def main():
    torch.set_num_threads(1)
    entry.build_potential = lambda force_rms, element_refs: TinyPotential()
    output = HERE / f"lightning_200_epoch_schedule_{time.time_ns()}"
    output.mkdir()
    full_model, full_rows, full, _ = run(output / "uninterrupted", 200)
    first_model, first_rows, first, _ = run(output / "resumed", 73)
    resume_path = output / "resumed" / "checkpoints" / "last.ckpt"
    assert first["epoch"] == 72 and first["global_step"] == 146
    assert first["lr_schedulers"][0]["last_epoch"] == 73
    resumed_model, resumed_rows, resumed, selected = run(output / "resumed", 200, resume_path)
    assert_same(full["state_dict"], resumed["state_dict"])
    assert_same(full["optimizer_states"], resumed["optimizer_states"])
    assert_same(full["lr_schedulers"], resumed["lr_schedulers"])
    assert full_rows == first_rows + resumed_rows
    calls = first_model.scheduler_calls + resumed_model.scheduler_calls
    assert full_model.scheduler_calls == calls
    assert len(calls) == 200
    assert all(row == {"epoch": i, "before": i, "after": i + 1} for i, row in enumerate(calls))
    for row in full_rows:
        epoch = row["epoch"]
        expected = 0.00001 + 0.5 * (0.001 - 0.00001) * (1 + math.cos(math.pi * epoch / 200))
        assert abs(row["lr"] - expected) < 0.00000000000001, row
        assert row["scheduler_last_epoch"] == epoch and row["global_step"] == 2 * epoch
    assert full["global_step"] == 400 and full["epoch"] == 199
    assert full["lr_schedulers"][0]["last_epoch"] == 200
    final_lr = full["optimizer_states"][0]["param_groups"][0]["lr"]
    assert abs(final_lr - 0.00001) < 0.00000000000001
    # Every epoch checkpoint, including best, must contain the scheduler's NEXT
    # epoch position rather than the just-finished epoch's old learning rate.
    for cp_path in (output / "uninterrupted" / "checkpoints").glob("*.ckpt"):
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        assert cp["lr_schedulers"][0]["last_epoch"] == cp["epoch"] + 1, cp_path
    result = {"passed": True, "actual_lightning_epochs": 400, "campaign_epochs": 200,
              "optimizer_updates_per_campaign": 400, "scheduler_calls_per_campaign": len(calls),
              "interrupted_after_completed_epochs": 73, "resumed_first_epoch": resumed_rows[0],
              "learning_rates_at_epoch_start": {str(i): full_rows[i]["lr"] for i in (0, 1, 73, 100, 199)},
              "learning_rate_after_completed_epoch_200": final_lr,
              "bitwise_identical_after_resume": ["weights", "AdamW first/second/max second moments and step",
                                                   "optimizer groups", "scheduler state", "epoch LR trace"],
              "all_epoch_and_best_checkpoints_have_updated_scheduler": True,
              "best_checkpoint_after_resume": selected,
              "scope": "Actual Lightning loop and production AlignedModule hooks, synthetic 3-parameter EFS predictor; not full-model training."}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "epoch_trace.json").write_text(json.dumps(full_rows, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **result}, indent=2))


if __name__ == "__main__":
    main()
