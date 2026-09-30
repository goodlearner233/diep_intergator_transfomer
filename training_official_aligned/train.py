"""Fresh DIEPformer training with the official DIEP training protocol.

No original training file is imported as an entry point or modified. Run --help.
This entry uses the local 121-channel DIEPformer, NOT the official sum descriptor.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning as L
import numpy as np
import torch
from filelock import FileLock, Timeout as FileLockTimeout
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset, Subset

from matgl.apps.pes import Potential
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.data import collate_fn_pes
from matgl.models import M3GNet
from matgl.utils.training import xavier_init
from data_pipeline import MaxAtomsBatchSampler, load_prepared, prepare_data, sha256_file, distribution_versions
from telemetry import Telemetry

OFFICIAL_COMMIT = "3bf9cd4cf4fe43081fa37b89edff28eeb2ef9cca"
FORMAT = "diepformer-official-aligned-v1"


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def _launcher_identity():
    """Shared launch identity for external torchrun workers on one node."""
    error_file = os.environ.get("TORCHELASTIC_ERROR_FILE")
    if error_file:
        # torchrun allocates a unique attempt directory, even with run ID "none".
        # The last two components are <rank>/error.json and differ per worker.
        return {"torchrun_attempt": str(Path(error_file).parent.parent),
                "world_size": os.environ.get("WORLD_SIZE")}
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if run_id and run_id.lower() != "none":
        return {"torchrun_id": run_id, "master_addr": os.environ.get("MASTER_ADDR"),
                "master_port": os.environ.get("MASTER_PORT"),
                "world_size": os.environ.get("WORLD_SIZE"),
                "restart": os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")}
    return None


@contextmanager
def run_admission(output_dir, worker_timeout=120.0):
    """One writer group per run directory, including during fit/test/export.

    Rank zero owns an OS file lock. Lightning subprocesses inherit a fresh UUID;
    independently launched torchrun workers use their shared launch identity to
    read that UUID. A static torchrun ID is never used as the UUID itself.
    """
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(output / ".run.lock"))
    active_path = output / ".active_run.json"
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    launcher = _launcher_identity()
    if rank == 0:
        try:
            lock.acquire(timeout=0)
        except FileLockTimeout as exc:
            raise RuntimeError("Another training invocation is using this output directory; "
                               "do not start a concurrent fresh run or resume") from exc
        token = str(uuid.uuid4())
        os.environ["DIEP_ALIGNED_RUN_TOKEN"] = token
        try:
            write_json(active_path, {"token": token, "launcher": launcher, "rank_zero_pid": os.getpid()})
            yield token
        finally:
            try:
                active_path.unlink(missing_ok=True)
            finally:
                lock.release()
        return

    inherited = os.environ.get("DIEP_ALIGNED_RUN_TOKEN")
    if not inherited and launcher is None:
        raise RuntimeError("Worker has no verifiable parent launch. Use Python with --devices "
                           "or a standard single-node torchrun launch")
    deadline = time.monotonic() + worker_timeout
    while time.monotonic() < deadline:
        try:
            active = json.loads(active_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            active = None
        matches = active and (active.get("launcher") == launcher if launcher is not None
                              else active.get("token") == inherited)
        if matches:
            try:
                lock.acquire(timeout=0)
            except FileLockTimeout:
                os.environ["DIEP_ALIGNED_RUN_TOKEN"] = active["token"]
                yield active["token"]
                return
            else:
                # A stale descriptor from an interrupted process is not admission.
                lock.release()
        time.sleep(0.05)
    raise RuntimeError("Worker could not join an active matching rank-zero launch")


@contextmanager
def math_attention():
    # Context is entered inside every rank's forward, including force backward.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        context = sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        context = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    with context:
        yield


def build_potential(force_rms, element_refs):
    model = M3GNet(
        element_types=DEFAULT_ELEMENTS, cutoff=5.0, threebody_cutoff=4.0,
        diep_grid_half_length=5.0, diep_grid_spacing=1.0,
        dim_node_embedding=64, dim_edge_embedding=64, nblocks=3,
        is_intensive=False, readout_type="transformer", transformer_nhead=4,
        transformer_num_layers=1, transformer_dim_ff=128, transformer_dropout=0.0,
    )
    xavier_init(model)
    return Potential(model=model, data_mean=0.0, data_std=float(force_rms),
                     element_refs=np.asarray(element_refs), calc_stresses=True)


class MaskedEvaluationDataset(Dataset):
    """Negative indices mark repeated DDP padding, never real evaluation samples."""
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        valid = index >= 0
        index = index if valid else -index - 1
        graph, *rest = self.subset[index]
        graph = graph.clone()
        graph.metric_weight = torch.tensor([float(valid)])
        return (graph, *rest)


class EvaluationBatchSampler:
    """Same number of forwards per rank; each real sample counted exactly once."""
    def __init__(self, atom_counts, max_atoms, rank, world_size):
        packs = list(MaxAtomsBatchSampler(atom_counts, max_atoms, shuffle=False,
                                         rank=0, num_replicas=1, pad=False))
        if not packs:
            raise ValueError("Validation/test split must not be empty")
        real_count = len(packs)
        total = math.ceil(real_count / world_size) * world_size
        packs += [[-i - 1 for i in packs[j % real_count]]
                  for j in range(total - real_count)]
        self.packs = packs[rank::world_size]

    def __len__(self):
        return len(self.packs)

    def __iter__(self):
        yield from self.packs


class TrainingData(L.LightningDataModule):
    def __init__(self, output_dir, budget=1000, workers=0, seed=42):
        super().__init__()
        self.output_dir, self.budget = Path(output_dir), budget
        self.workers, self.seed = workers, seed
        self.prepared = None
        self.train_sampler = None

    def setup(self, stage=None):
        if self.prepared is None:
            self.prepared = load_prepared(self.output_dir)

    def loader(self, phase):
        p = self.prepared
        indices = p["splits"][phase]
        subset = Subset(p["dataset"], indices)
        counts = [p["atom_counts"][i] for i in indices]
        rank, world = self.trainer.global_rank, self.trainer.world_size
        if phase == "train":
            sampler = MaxAtomsBatchSampler(counts, self.budget, shuffle=True,
                                           rank=rank, num_replicas=world,
                                           seed=self.seed, pad=True)
            self.train_sampler = sampler
        else:
            sampler = EvaluationBatchSampler(counts, self.budget, rank, world)
            subset = MaskedEvaluationDataset(subset)
        return DataLoader(subset, batch_sampler=sampler, collate_fn=collate_fn_pes,
                          num_workers=self.workers, pin_memory=self.trainer.accelerator.__class__.__name__ == "CUDAAccelerator",
                          persistent_workers=self.workers > 0,
                          generator=torch.Generator().manual_seed(self.seed + rank))

    def train_dataloader(self):
        return self.loader("train")

    def val_dataloader(self):
        return self.loader("val")

    def test_dataloader(self):
        return self.loader("test")


class AlignedModule(L.LightningModule):
    def __init__(self, force_rms, element_refs, signature, epochs=200,
                 lr=0.001, min_lr=0.00001):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_potential(force_rms, element_refs)
        self.signature = signature
        self.epochs, self.lr, self.min_lr = epochs, lr, min_lr
        self.phase_stats = {}
        self.energy_weight, self.force_weight, self.stress_weight = 1.0, 1.0, 0.1
        self.loss_params = {"delta": 1.0}
        self.latest_metrics = {}

    def forward(self, g, lat, state_attr):
        # Lightning disables grads for validation; coordinate differentiation still needs them.
        with torch.enable_grad(), math_attention():
            return self.model(g=g, lat=lat, state_attr=state_attr)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.999),
                                eps=0.00000001, weight_decay=0.00001, amsgrad=True)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs,
                                                       eta_min=self.min_lr)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["aligned_format"] = FORMAT
        checkpoint["aligned_signature"] = self.signature
        progress = checkpoint.get("loops", {}).get("fit_loop", {}).get("epoch_progress", {}).get("current", {})
        checkpoint["aligned_resume_scope"] = (
            "epoch-boundary" if int(progress.get("processed", 0)) > int(checkpoint.get("epoch", -1))
            else "diagnostic-only"
        )

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("aligned_format") != FORMAT:
            raise ValueError("Only checkpoints created by this independent entry may be resumed")
        if checkpoint.get("aligned_signature") != self.signature:
            raise ValueError("Resume rejected: data, model, code or training configuration differs")
        trainer = self._trainer
        if trainer is not None and getattr(getattr(trainer, "state", None), "fn", None) == "fit":
            progress = checkpoint.get("loops", {}).get("fit_loop", {}).get("epoch_progress", {}).get("current", {})
            if not progress or int(progress.get("processed", 0)) <= int(checkpoint.get("epoch", -1)):
                raise ValueError(
                    "Incomplete-epoch checkpoints are diagnostic-only snapshots: partial batch/metric "
                    "state cannot be restored reliably. Resume checkpoints/last.ckpt or another "
                    "complete-epoch checkpoint instead."
                )
            # Epoch checkpoints are saved after processed increments, but before
            # completed increments. Mid-epoch emergency saves retain the prior
            # processed count; checkpoint['epoch'] + 1 would overcount those.
            completed = max(int(progress.get("processed", 0)), int(progress.get("completed", 0)))
            if trainer.max_epochs is not None and 0 <= trainer.max_epochs < completed:
                raise ValueError(
                    f"Resume stop limit {trainer.max_epochs} is below the checkpoint's "
                    f"{completed} completed epochs. Increase/remove --stop-after-epochs; "
                    "keep --epochs equal to the original full campaign."
                )

    def start_phase(self, phase):
        # [3 targets x (absolute sum, squared sum, Huber sum, component count),
        #  3 targets x (batch MAE, batch RMSE, batch Huber) weighted by structures,
        #  weighted total loss, structure count].
        self.phase_stats[phase] = torch.zeros(23, dtype=torch.float64, device=self.device)

    def on_train_epoch_start(self):
        self.start_phase("train")
        sampler = self.trainer.datamodule.train_sampler
        if sampler is not None:
            sampler.set_epoch(self.current_epoch)

    def on_validation_epoch_start(self):
        self.start_phase("val")

    def on_test_epoch_start(self):
        self.start_phase("test")

    def step(self, batch, phase):
        graph, lat, state, energy, force, stress = batch
        total_e, pred_f, pred_s, *_ = self(graph, lat, state)
        counts = torch.bincount(graph.batch)
        pred_e = total_e.reshape(-1) / counts
        true_e = energy.reshape(-1) / counts
        preds, labels = (pred_e, pred_f, pred_s), (true_e, force, stress)
        losses = [torch.nn.functional.huber_loss(p, y, delta=1.0) for p, y in zip(preds, labels)]
        loss = losses[0] + losses[1] + 0.1 * losses[2]
        valid = getattr(graph, "metric_weight", torch.ones_like(counts)).to(torch.bool)
        outputs = {"loss": loss, "preds": (total_e.detach(), pred_f.detach(), pred_s.detach()),
                   "labels": (energy.detach(), force.detach(), stress.detach()),
                   "num_atoms": counts.detach(), "indices": graph.sample_idx.detach(),
                   "metric_weight": valid.detach()}
        finite = torch.isfinite(loss.detach()).to(torch.int32)
        # Evaluation uses equally many forwards on every rank too. Synchronize
        # rejection before one rank exits while another waits at epoch reduction.
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not bool(finite):
            reason = f"Nonfinite {phase} loss on at least one rank"
            for callback in self.trainer.callbacks:
                if isinstance(callback, Telemetry):
                    callback.record_failed_batch(self.trainer, self, phase, batch, outputs, reason)
            raise FloatingPointError(f"Nonfinite {phase} loss; refusing to update parameters")
        nvalid = int(valid.sum())
        if nvalid:
            masks = (valid, torch.repeat_interleave(valid, counts), valid.repeat_interleave(3))
            stats = self.phase_stats[phase]
            with torch.no_grad():
                for i, (p, y, mask) in enumerate(zip(preds, labels, masks)):
                    diff = (p.detach()[mask] - y.detach()[mask]).double()
                    absdiff = diff.abs()
                    huber = torch.where(absdiff <= 1, 0.5 * diff.square(), absdiff - 0.5)
                    part = torch.stack((absdiff.sum(), diff.square().sum(), huber.sum(),
                                        diff.new_tensor(diff.numel())))
                    stats[i * 4:i * 4 + 4] += part
                    stats[12 + i * 3:15 + i * 3] += nvalid * torch.stack(
                        (part[0] / part[3], (part[1] / part[3]).sqrt(), part[2] / part[3]))
                stats[21] += loss.detach().double() * nvalid
                stats[22] += nvalid
        return outputs

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, "test")

    def end_phase(self, phase):
        stats = self.phase_stats[phase]
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
        if stats[22] <= 0:
            raise RuntimeError(f"No real samples in {phase}")
        metrics = {f"{phase}_Total_Loss": stats[21] / stats[22],
                   f"{phase}_structure_visits": stats[22]}
        pooled_loss = stats.new_zeros(())
        for i, task in enumerate(("Energy", "Force", "Stress")):
            part = stats[i * 4:i * 4 + 4]
            pooled = (part[0] / part[3], (part[1] / part[3]).sqrt(), part[2] / part[3])
            for j, metric in enumerate(("MAE", "RMSE", "Huber")):
                metrics[f"{phase}_{task}_{metric}"] = stats[12 + i * 3 + j] / stats[22]
                metrics[f"{phase}_pooled_{task}_{metric}"] = pooled[j]
            pooled_loss += (0.1 if i == 2 else 1.0) * pooled[2]
        metrics[f"{phase}_pooled_Total_Loss"] = pooled_loss
        if not self.trainer.sanity_checking:
            self.latest_metrics.update({k: float(v) for k, v in metrics.items()})
        self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=False)
        if phase == "train":
            self.log("completed_epochs", float(self.current_epoch + 1), sync_dist=False)

    def on_train_epoch_end(self):
        self.end_phase("train")
        if self.trainer.is_global_zero:
            values = dict(self.latest_metrics)
            values.update(epoch=self.current_epoch, completed_epochs=self.current_epoch + 1,
                          global_step=self.global_step,
                          lr=self.trainer.optimizers[0].param_groups[0]["lr"])
            path = Path(self.trainer.default_root_dir) / "epoch_history.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(values, allow_nan=False) + "\n")

    def on_validation_epoch_end(self):
        self.end_phase("val")

    def on_test_epoch_end(self):
        self.end_phase("test")


class FiniteGradients(Callback):
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        finite = torch.ones((), device=pl_module.device, dtype=torch.int32)
        for p in pl_module.parameters():
            if p.grad is not None:
                finite *= torch.isfinite(p.grad).all().to(torch.int32)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
        if not bool(finite):
            raise FloatingPointError("Nonfinite gradient: stopped before AdamW update; inspect telemetry")


class RunHistory(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            lr = trainer.optimizers[0].param_groups[0]["lr"]
            print(f"EPOCH_START epoch={trainer.current_epoch} lr={lr:.10f}", flush=True)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True, help="Raw MatPES PBE JSON/JSONL (stresses in kbar)")
    p.add_argument("--element-refs", type=Path, required=True, help="MatPES-PBE-atoms.json")
    p.add_argument("--output-dir", type=Path, required=True, help="New output directory, separate from old runs")
    p.add_argument("--epochs", type=int, default=200, help="Full campaign length and cosine period")
    p.add_argument("--stop-after-epochs", type=int, help="Stop after this many total epochs, without changing cosine period")
    p.add_argument("--resume", type=Path, help="Full checkpoint from this SAME new run; prefer epoch checkpoints")
    p.add_argument("--accelerator", choices=("gpu", "cpu"), default="gpu")
    p.add_argument("--devices", type=int, default=4)
    p.add_argument("--accumulate-grad-batches", type=int, default=1)
    p.add_argument("--max-atoms", type=int, default=150, help="Maximum atoms in ONE structure")
    p.add_argument("--max-atoms-per-batch", type=int, default=1000, help="Atom budget per GPU microbatch")
    p.add_argument("--split-policy", choices=("official", "legacy-membership"), default="official")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--min-lr", type=float, default=0.00001)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--step-checkpoint-every", type=int, default=500,
                   help="Rolling diagnostic-only snapshot every N updates; resume from epoch checkpoints; 0 disables")
    p.add_argument("--telemetry-coordinates", type=int, default=128)
    p.add_argument("--no-telemetry", action="store_true")
    p.add_argument("--skip-final-test", action="store_true")
    p.add_argument("--no-progress-bar", action="store_true")
    args = p.parse_args(argv)
    if min(args.epochs, args.devices, args.accumulate_grad_batches, args.max_atoms,
           args.max_atoms_per_batch) < 1:
        p.error("Epochs, devices, accumulation and atom budgets must be positive")
    if not 0 < args.min_lr <= args.lr or not math.isfinite(args.lr):
        p.error("Require 0 < min_lr <= lr and finite learning rates")
    if args.num_workers < 0 or args.step_checkpoint_every < 0:
        p.error("Worker and checkpoint counts must be nonnegative")
    if not 1 <= args.telemetry_coordinates <= 4096:
        p.error("telemetry-coordinates must be within 1..4096; --no-telemetry disables logging")
    if args.stop_after_epochs is not None and not 0 < args.stop_after_epochs <= args.epochs:
        p.error("stop-after-epochs must be within the full campaign")
    return args


def initialize_run(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Admission supplies a fresh UUID shared only by this currently active group.
    token = os.environ["DIEP_ALIGNED_RUN_TOKEN"]
    files = sorted((ROOT / "src" / "matgl").rglob("*.py")) + sorted(Path(__file__).parent.glob("*.py"))
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in files}
    config = {k: v for k, v in vars(args).items() if k not in {
        "data", "element_refs", "output_dir", "resume", "stop_after_epochs", "no_telemetry",
        "telemetry_coordinates", "skip_final_test", "no_progress_bar", "num_workers", "accelerator"}}
    config.update(format=FORMAT, official_commit=OFFICIAL_COMMIT, source_hashes=hashes,
                  optimizer="AdamW", betas=[0.9, 0.999], epsilon=0.00000001,
                  weight_decay=0.00001, amsgrad=True, gradient_clip_norm=2.0,
                  loss="Huber", delta=1.0, weights=[1.0, 1.0, 0.1], precision="32-true",
                  model="121-grid DIEP / 3 M3GNet blocks / 1 Transformer layer, 4 heads, 64 dims / Linear sum")
    with FileLock(str(output / ".prepare.lock")):
        manifest_path = output / "run_config.json"
        old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        if old and not args.resume and old["invocation_token"] != token:
            raise ValueError("Output already belongs to a run. Use a NEW directory or explicit --resume")
        if args.resume and (old is None or not args.resume.is_file()):
            raise ValueError("Resume requires the original run directory and an existing checkpoint")
        meta = prepare_data(args.data.resolve(), output, max_atoms=args.max_atoms,
                            cutoff=5.0, seed=args.seed, split_policy=args.split_policy,
                            element_refs_path=args.element_refs.resolve(), threebody_cutoff=4.0)
        config["data_manifest_sha256"] = meta["manifest_sha256"]
        signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        if old and old["signature"] != signature:
            raise ValueError("Run configuration/source/data changed; use a new run directory")
        if not old:
            record = {"configuration": config, "signature": signature, "invocation_token": token,
                      "data_path": str(args.data.resolve()), "refs_path": str(args.element_refs.resolve()),
                      "metadata": meta, "versions": distribution_versions((
                          "torch", "lightning", "torch-geometric", "numpy", "pymatgen", "pymatgen-core",
                          "ase", "filelock")), "python_version": sys.version}
            write_json(manifest_path, record)
            for p in files:
                dest = output / "source_snapshot" / p.relative_to(ROOT)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
    return meta, signature


def _run_training(args):
    if args.accelerator == "gpu" and (not torch.cuda.is_available() or torch.cuda.device_count() < args.devices):
        raise RuntimeError(f"Requested {args.devices} GPUs, visible {torch.cuda.device_count()}. Set --devices explicitly")
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    meta, signature = initialize_run(args)
    module = AlignedModule(meta["force_rms"], meta["element_refs"], signature,
                           epochs=args.epochs, lr=args.lr, min_lr=args.min_lr)
    data = TrainingData(args.output_dir, args.max_atoms_per_batch, args.num_workers, args.seed)
    checkpoints = args.output_dir / "checkpoints"
    best = ModelCheckpoint(dirpath=checkpoints, filename="best-{epoch:03d}-{step}",
                           monitor="val_Total_Loss", mode="min", save_top_k=1,
                           save_on_train_epoch_end=True)
    history = ModelCheckpoint(dirpath=checkpoints, filename="epoch-{epoch:03d}-step-{step}",
                              auto_insert_metric_name=False, save_top_k=-1, save_last=True,
                              every_n_epochs=1, save_on_train_epoch_end=True)
    callbacks = [Telemetry(args.output_dir, enabled=not args.no_telemetry,
                            coordinate_count=args.telemetry_coordinates),
                 FiniteGradients(), RunHistory(), LearningRateMonitor(logging_interval="epoch"), best, history]
    if args.step_checkpoint_every:
        callbacks.append(ModelCheckpoint(dirpath=checkpoints / "emergency", filename="step-latest",
                                          every_n_train_steps=args.step_checkpoint_every,
                                          save_top_k=1, enable_version_counter=False,
                                          save_on_train_epoch_end=False))
    logger = CSVLogger(str(args.output_dir), name="logs")
    strategy = DDPStrategy(find_unused_parameters=True, broadcast_buffers=False) if args.devices > 1 else "auto"
    trainer = L.Trainer(default_root_dir=str(args.output_dir), logger=logger, callbacks=callbacks,
                        accelerator=args.accelerator, devices=args.devices, strategy=strategy,
                        max_epochs=args.stop_after_epochs or args.epochs,
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        precision="32-true", gradient_clip_val=2.0, gradient_clip_algorithm="norm",
                        inference_mode=False, use_distributed_sampler=False,
                        num_sanity_val_steps=2, log_every_n_steps=1,
                        enable_progress_bar=not args.no_progress_bar)
    print(f"NEW_PROTOCOL seed={args.seed} AdamW AMSGrad beta2=0.999 lr={args.lr:.8f} "
          f"min_lr={args.min_lr:.8f} epochs={args.epochs} force_rms={meta['force_rms']:.8f}", flush=True)
    trainer.fit(module, datamodule=data, ckpt_path=str(args.resume.resolve()) if args.resume else None)
    if trainer.is_global_zero:
        write_json(args.output_dir / "training_status.json", {
            "completed_epochs": trainer.current_epoch, "target_epochs": args.epochs,
            "global_step": trainer.global_step, "best_checkpoint": best.best_model_path,
            "best_val_Total_Loss": float(best.best_model_score) if best.best_model_score is not None else None,
            "last_checkpoint": history.last_model_path,
            "lr_after_completed_epochs": trainer.optimizers[0].param_groups[0]["lr"]})
    if trainer.current_epoch >= args.epochs and not args.skip_final_test:
        if not best.best_model_path:
            raise RuntimeError("No best validation checkpoint exists; refusing to test the last model implicitly")
        results = trainer.test(module, datamodule=data, ckpt_path=best.best_model_path)
        if trainer.is_global_zero:
            write_json(args.output_dir / "final_test.json", {"checkpoint": best.best_model_path,
                                                           "selection": "minimum val_Total_Loss", "metrics": results})
            module.model.save(args.output_dir / "best_potential", metadata={
                "signature": signature, "selected_checkpoint": best.best_model_path,
                "includes_element_refs_and_force_rms": True})
    return 0


def main(argv=None):
    args = parse_args(argv)
    with run_admission(args.output_dir):
        return _run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
