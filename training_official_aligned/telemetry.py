"""Read-only, streaming E/F/S and AdamW diagnostics for FP32 Lightning/DDP.

Each rank writes its own uniquely named session. Batch errors are rank-local;
rank zero observes synchronized optimizer updates (it does not claim that its
sample IDs represent other ranks). No full gradient/checkpoint history is kept.
One temporary parameter snapshot measures an actual update. These observations
provide diagnostic clues, not a causal test or Hessian/curvature measurement.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import time
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    return value


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(v) for v in value)
    return True


def _stats(tensor: torch.Tensor) -> torch.Tensor:
    """Six scalar reductions on the tensor's device; transfer only afterwards."""
    value = tensor.detach().to(torch.float64).reshape(-1)
    if not value.numel():
        return torch.zeros(6, dtype=torch.float64, device=value.device)
    return torch.stack((value.square().sum().sqrt(), value.abs().max(), value.mean(),
                        value.min(), value.max(), (~torch.isfinite(value)).sum().double()))


def _stat_dict(row: list[float], elements: int) -> dict[str, Any]:
    return dict(zip(("l2", "max_abs", "mean", "min", "max", "nonfinite_elements"), row)) | {"elements": elements}


def _parameter_group(name: str) -> str:
    """Use the recovery observer's name-prefix grouping, including block index."""
    parts = name.split(".")
    length = min(3, len(parts) - 1)
    if len(parts) > length + 1 and parts[length].isdigit():
        length += 1
    return ".".join(parts[:max(1, length)])


def _squares_and_nonfinite(tensor: torch.Tensor) -> torch.Tensor:
    """Two device scalars; temporary storage is limited to one parameter."""
    value = tensor.detach().to(torch.float64).reshape(-1)
    return torch.stack((value.square().sum(), (~torch.isfinite(value)).sum().double()))


class Telemetry(L.Callback):
    """Observe complete E/F/S batches and AdamW updates without changing tensors.

    Supports automatic optimization, one non-capturable/non-differentiable AdamW,
    FP32, and DDP's synchronized gradients. Compressed JSONL size grows with the
    number of batches/updates, with bounded records, buffering and RAM usage.
    ``preds`` and ``labels`` must contain TOTAL energy, force and stress tensors;
    ``num_atoms`` and ``indices`` retain the original input sample identities.
    """

    def __init__(self, output_dir: str | Path, enabled: bool = True,
                 coordinate_count: int = 128, flush_every: int = 20,
                 module_statistics: bool = True) -> None:
        super().__init__()
        if not 1 <= coordinate_count <= 4096 or flush_every < 1:
            raise ValueError("coordinate_count must be 1..4096 and flush_every >= 1")
        self.output_dir = Path(output_dir)
        self.enabled = bool(enabled)
        self.coordinate_count = int(coordinate_count)
        self.flush_every = int(flush_every)
        self.module_statistics = bool(module_statistics)
        self.session_dir: Path | None = None
        self._files: dict[str, Any] = {}
        self._counts: dict[str, int] = {}
        self._hook = None
        self._workload_hooks: list[Any] = []
        self._workload: dict[str, Any] | None = None
        self._trainer = None
        self._module = None
        self._parameters: list[tuple[str, torch.Tensor]] = []
        self._groups: list[int] = []
        self._offsets: list[tuple[int, int]] = []
        self._coordinate_indices: torch.Tensor | None = None
        self._coordinate_parameters: list[int] = []
        self._pending: dict[str, Any] | None = None
        self._contexts: dict[str, dict[str, Any]] = {}
        self._window: list[dict[str, Any]] = []
        self._logging_error: str | None = None

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str) -> None:
        if not self.enabled or self._files:
            return
        self._trainer, self._module = trainer, pl_module
        self._logging_error = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.output_dir / (f"{stage}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
                                            f"_rank{trainer.global_rank}_{uuid.uuid4().hex[:12]}")
        self.session_dir.mkdir(exist_ok=False)
        names = ["batches", "events"]
        if trainer.global_rank == 0 and stage == "fit":
            names += ["optimizer_updates", "coordinates"]
        for name in names:
            self._files[name] = gzip.open(self.session_dir / f"{name}.jsonl.gz", "xt", encoding="utf-8", compresslevel=3)
            self._counts[name] = 0
        self._event("session_started", stage=stage, rank=int(trainer.global_rank), world_size=int(trainer.world_size),
                    torch_version=torch.__version__, lightning_version=L.__version__,
                    metadata={
                        "schema_version": 2,
                        "batch_scope": "rank-local; join ranks using epoch/batch/update counters",
                        "optimizer_scope": "rank zero; DDP gradients synchronized before observation",
                        "energy_error_units": "eV/atom", "force_error_units": "eV/Angstrom", "stress_error_units": "GPa",
                        "elapsed_seconds": "unsynchronized host wall time from batch-start hook; not GPU kernel time or dataloader time",
                        "memory_bytes": "PyTorch CUDA allocator on this rank/device, includes observer overhead; not all device/process memory",
                        "batch_errors": "Local unweighted errors include any eval padding; metric_weight marks samples contributing to reported validation/test metrics",
                        "denominator": "sqrt(max_exp_avg_sq / (1-beta2**step)) + eps for AMSGrad; exp_avg_sq otherwise",
                        "decay_delta": "-lr*weight_decay*old_parameter for active AdamW parameters; remainder includes floating-point rounding",
                        "module_statistics_enabled": self.module_statistics,
                        "module_statistics": "Parameter-name prefixes (including block index), not activation statistics. Before/after clip gradients and actual parameter delta include only available tensors; absent gradients are counted explicitly. Derived Adam summaries omit uninitialized slots and use each parameter's own optimizer group and state step. Ordinary v supplies v_hat/sqrt_v_hat; AMSGrad max_v supplies the actual denominator. Float64 mathematical summaries may differ slightly from fused/FP32 optimizer arithmetic.",
                        "nonfinite": "nonfinite floating values encoded null; finite flags retain failure information",
                        "failure_evidence": "Failed loss batches are explicitly recorded before the training entry raises. Exception events retain pending scalar/module/fixed-coordinate diagnostics, never the full parameter snapshot. A failure on another DDP rank may leave this rank's batch finite.",
                        "limits": "Fixed scalar coordinates can miss anomalies; no Hessian or all-gradient history; no causal attribution",
                    })
        for child in pl_module.modules():
            if hasattr(child, "threebody_cutoff") and hasattr(child, "diep_grid_half_length"):
                self._workload_hooks.append(child.register_forward_pre_hook(self._capture_workload, with_kwargs=True))
        self._flush()

    @torch.no_grad()
    def _capture_workload(self, module, args, kwargs):
        """Count actual directed source-sharing triples before M3GNet builds them.

        Matches graph._compute: keep distances <= threebody_cutoff; each source
        with degree d contributes d*(d-1) ordered pairs of distinct edges.
        Receives Potential's private geometry clone, never mutates that graph.
        """
        graph = kwargs.get("g", args[0] if args else None)
        if graph is None or getattr(graph, "pos", None) is None:
            return
        src, dst = graph.edge_index
        shift = getattr(graph, "pbc_offshift", None)
        vector = graph.pos[dst] - graph.pos[src]
        if shift is not None:
            vector = vector + shift
        keep = vector.norm(dim=1) <= float(module.threebody_cutoff)
        degree = torch.bincount(src[keep], minlength=graph.pos.shape[0])
        count = degree * (degree - 1)
        membership = getattr(graph, "batch", None)
        if membership is None:
            membership = torch.zeros(graph.pos.shape[0], dtype=torch.long, device=src.device)
        n_graphs = int(getattr(graph, "num_graphs", 1))
        per_graph = torch.zeros(n_graphs, dtype=torch.long, device=src.device)
        per_graph.scatter_add_(0, membership, count)
        edges = torch.bincount(membership[src], minlength=n_graphs)
        values = torch.stack((edges, per_graph)).cpu().tolist()
        self._workload = {"n_edges": sum(values[0]), "n_triplets": sum(values[1]),
                          "edges_per_structure": values[0], "triplets_per_structure": values[1],
                          "triplet_count_source": "forward geometry; directed source degree*(degree-1)",
                          "triplet_cutoff_angstrom": float(module.threebody_cutoff)}

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not self.enabled:
            return
        if not pl_module.automatic_optimization or str(trainer.precision) not in {"32", "32-true"}:
            raise ValueError("Telemetry requires automatic optimization and FP32")
        if len(trainer.optimizers) != 1 or not isinstance(trainer.optimizers[0], torch.optim.AdamW):
            raise ValueError("Telemetry requires exactly one torch.optim.AdamW")
        optimizer = trainer.optimizers[0]
        if any(g.get("capturable", False) or g.get("differentiable", False) or g.get("maximize", False)
               for g in optimizer.param_groups):
            raise ValueError("Telemetry supports minimizing, noncapturable, nondifferentiable AdamW")
        if getattr(pl_module, "loss_params", {}).get("reduction", "mean") != "mean":
            raise ValueError("Telemetry requires mean Huber loss")
        if trainer.global_rank != 0:
            return
        known = {id(p): name for name, p in pl_module.named_parameters()}
        entries = [(known[id(p)], p, gi) for gi, group in enumerate(optimizer.param_groups)
                   for p in group["params"] if p.requires_grad]
        entries.sort(key=lambda item: item[0])
        self._parameters = [(name, p) for name, p, _ in entries]
        self._groups = [gi for _, _, gi in entries]
        if len({p.device for _, p in self._parameters}) != 1:
            raise ValueError("Each telemetry rank expects parameters on one device")
        self._offsets = []
        total = 0
        for _, p in self._parameters:
            self._offsets.append((total, total + p.numel()))
            total += p.numel()
        count = min(self.coordinate_count, total)
        # Exact integer spread, no RNG and no dependency on global torch seeds.
        positions = [i * (total - 1) // max(count - 1, 1) for i in range(count)]
        self._coordinate_indices = torch.tensor(positions, device=self._parameters[0][1].device, dtype=torch.long)
        metadata = []
        self._coordinate_parameters = []
        for position in positions:
            pi = next(i for i, (start, stop) in enumerate(self._offsets) if start <= position < stop)
            self._coordinate_parameters.append(pi)
            metadata.append({"name": self._parameters[pi][0], "parameter_flat_index": position - self._offsets[pi][0],
                             "optimizer_group": self._groups[pi]})
        self._event("optimizer_observer_started", optimizer=type(optimizer).__name__, coordinates=metadata,
                    coordinate_count=count, optimizer_groups=self._group_settings(optimizer),
                    parameter_name_groups=[{"name": name, "module": _parameter_group(name),
                                            "shape": list(p.shape), "elements": p.numel(),
                                            "optimizer_group": gi}
                                           for (name, p), gi in zip(self._parameters, self._groups)]
                    if self.module_statistics else [])
        self._hook = optimizer.register_step_post_hook(self._after_optimizer_step)

    @staticmethod
    def _group_settings(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
        return [{key: (float(g[key]) if key in {"lr", "eps", "weight_decay"} else g.get(key))
                 for key in ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize")}
                for g in optimizer.param_groups]

    def _start_batch(self, phase: str, trainer: L.Trainer, pl_module: L.LightningModule,
                     batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        if not self.enabled:
            return
        if dataloader_idx != 0:
            raise ValueError("Telemetry expects one dataloader per phase")
        graph = batch[0]
        indices = getattr(graph, "sample_idx", None)
        if indices is None:
            raise ValueError("Telemetry requires graph.sample_idx containing original sample IDs")
        context = {"epoch": int(trainer.current_epoch), "batch_idx": int(batch_idx),
                   "prediction_global_step": int(trainer.global_step),
                   "sample_ids": indices.detach().cpu().reshape(-1).tolist(),
                   "accumulation_index": len(self._window) + 1 if phase == "train" else 0,
                   "started": time.perf_counter()}
        self._contexts[phase] = context
        self._workload = None
        if phase == "train":
            self._window.append({k: context[k] for k in ("batch_idx", "sample_ids", "prediction_global_step")})
        device = pl_module.device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start_batch("train", trainer, pl_module, batch, batch_idx)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._start_batch("val", trainer, pl_module, batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._start_batch("test", trainer, pl_module, batch, batch_idx, dataloader_idx)

    @torch.no_grad()
    def _record_batch(self, phase, trainer, pl_module, outputs, batch, batch_idx, failure_reason=None):
        if not self.enabled:
            return
        # Keep the original context available if a reduction or write fails.
        context = dict(self._contexts[phase])
        # Capture memory before the observer performs any reductions/copies.
        device = pl_module.device
        memory = {key: None for key in ("allocated_bytes", "reserved_bytes", "max_allocated_bytes_since_batch_start",
                                       "max_reserved_bytes_since_batch_start")}
        if device.type == "cuda":
            memory = dict(zip(memory, (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device),
                                       torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device))))
        elapsed = time.perf_counter() - context.pop("started")
        if not isinstance(outputs, Mapping) or any(k not in outputs for k in ("preds", "labels", "num_atoms", "indices")):
            raise ValueError("Telemetry requires preds/labels/num_atoms/indices outputs")
        counts = outputs["num_atoms"].detach().cpu().long().reshape(-1)
        ids = outputs["indices"].detach().cpu().reshape(-1).tolist()
        if ids != context["sample_ids"] or counts.numel() != len(ids) or not len(ids) or bool((counts <= 0).any()):
            raise ValueError("Telemetry sample IDs and atom counts must align with the batch")
        preds = [v.detach().cpu() for v in outputs["preds"][:3]]
        labels = [v.detach().cpu() for v in outputs["labels"][:3]]
        size = len(ids)
        ep, et = preds[0].reshape(-1) / counts, labels[0].reshape(-1) / counts
        fp, ft = preds[1].reshape(-1, 3), labels[1].reshape(-1, 3)
        sp, st = preds[2].reshape(size, -1), labels[2].reshape(size, -1)
        if ep.numel() != size or fp.shape != ft.shape or fp.shape[0] != int(counts.sum()) or sp.shape != st.shape:
            raise ValueError("Telemetry prediction and reference shapes differ")
        pairs = ((ep, et), (fp, ft), (sp, st))
        errors = [(p - t).double() for p, t in pairs]
        loss_params = dict(getattr(pl_module, "loss_params", {}))
        huber = [float(F.huber_loss(p, t, **loss_params)) for p, t in pairs]
        total = sum(float(getattr(pl_module, f"{name}_weight")) * loss
                    for name, loss in zip(("energy", "force", "stress"), huber))
        graph = batch[0]
        edges = getattr(graph, "edge_index", None)
        triples = getattr(graph, "line_edge_index", None)
        weights = outputs.get("metric_weight", getattr(graph, "metric_weight", None))
        weights = weights.detach().cpu().reshape(-1).tolist() if weights is not None else [1.0] * size
        if len(weights) != size:
            raise ValueError("Telemetry metric weights must match sample IDs")
        row = {**context, "phase": "sanity_val" if phase == "val" and trainer.sanity_checking else phase,
               "rank": int(trainer.global_rank), "world_size": int(trainer.world_size),
               "global_step_after_batch": int(trainer.global_step), "n_structures": size,
               "n_atoms": int(counts.sum()), "atom_counts": counts.tolist(),
               "n_edges": int(edges.shape[1]) if edges is not None else None,
               "n_triplets": int(triples.shape[1]) if triples is not None else None,
               "metric_weight": weights, "padding_sample_count": sum(float(w) == 0 for w in weights),
               "host_elapsed_seconds_unsynchronized": elapsed, "cuda_allocator": memory,
               "energy_abs_error_per_atom": errors[0].abs().tolist(),
               "force_mae_per_structure": [float(part.abs().mean()) for part in errors[1].split(counts.tolist())],
               "stress_mae_per_structure": errors[2].abs().mean(dim=1).tolist(),
               "weighted_total_huber": total}
        if failure_reason is not None:
            row["status"] = "failed"
            row["failure_reason"] = str(failure_reason)
            reported = outputs.get("loss")
            row["reported_total_loss"] = float(reported.detach().cpu()) if isinstance(reported, torch.Tensor) else reported
        if self._workload is not None:
            row.update(self._workload)
        for name, error, loss in zip(("energy", "force", "stress"), errors, huber):
            row.update({f"{name}_mae": float(error.abs().mean()), f"{name}_rmse": float(error.square().mean().sqrt()),
                        f"{name}_huber": loss})
        row["finite"] = _finite(row) and all(bool(torch.isfinite(v).all()) for pair in pairs for v in pair)
        self._write("batches", row)
        self._contexts.pop(phase, None)
        if phase == "train" and int(trainer.global_step) > context["prediction_global_step"]:
            self._window.clear()

    def record_failed_batch(self, trainer, pl_module, phase, batch, outputs, reason):
        """Best-effort evidence before the training entry raises a loss error.

        ``outputs`` uses the normal E/F/S telemetry contract and may contain a
        nonfinite ``loss`` tensor. This never raises over the training failure;
        a logging failure is retained for the subsequent exception event.
        """
        if not self.enabled:
            return
        try:
            context = self._contexts[phase]
            self._record_batch(phase, trainer, pl_module, outputs, batch, context["batch_idx"],
                               failure_reason=reason)
            self._flush(durable=True)
        except Exception as error:
            self._logging_error = f"Failed-batch evidence: {type(error).__name__}: {error}"

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._record_batch("train", trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._record_batch("val", trainer, pl_module, outputs, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._record_batch("test", trainer, pl_module, outputs, batch, batch_idx)

    def _flat(self, gradients=False):
        return torch.cat([(p.grad.detach() if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          if gradients else p.detach().reshape(-1) for _, p in self._parameters])

    @torch.no_grad()
    def _optimizer_snapshot(self, optimizer):
        params, grads = self._flat(), self._flat(gradients=True)
        fields: dict[str, list[torch.Tensor]] = {key: [] for key in ("m", "v", "max_v", "denominator")}
        steps, initialized, active = [], [], []
        for (_, p), gi in zip(self._parameters, self._groups):
            state = optimizer.state.get(p, {})
            step = float(state.get("step", 0))
            group = optimizer.param_groups[gi]
            zero = torch.zeros_like(p)
            m, v = state.get("exp_avg", zero), state.get("exp_avg_sq", zero)
            max_v = state.get("max_exp_avg_sq", zero)
            source = max_v if group.get("amsgrad", False) else v
            denominator = source.sqrt() / math.sqrt(1.0 - group["betas"][1] ** step) + group["eps"] if step > 0 else zero
            for key, value in zip(fields, (m, v, max_v, denominator)):
                fields[key].append(value.detach().reshape(-1))
            steps.append(step)
            initialized.append(step > 0)
            active.append(p.grad is not None)
        flat_fields = {key: torch.cat(parts) for key, parts in fields.items()}
        all_fields = {"parameter": params, "gradient": grads, **flat_fields}
        names = list(all_fields)
        stats = torch.stack([_stats(v) for v in all_fields.values()]).cpu().tolist()
        coordinates = torch.stack([v[self._coordinate_indices].double() for v in all_fields.values()]).cpu().tolist()
        selected = {key: row for key, row in zip(names, coordinates)}
        selected["steps"] = [steps[i] for i in self._coordinate_parameters]
        selected["active_gradients"] = [active[i] for i in self._coordinate_parameters]
        selected["state_initialized"] = [initialized[i] for i in self._coordinate_parameters]
        # Zero-filled missing-state slots are declared, not interpreted as a real denominator.
        selected["denominator"] = [value if present else None for value, present in
                                   zip(selected["denominator"], selected["state_initialized"])]
        selected["gradient"] = [value if present else None for value, present in
                                zip(selected["gradient"], selected["active_gradients"])]
        return params, {key: _stat_dict(row, params.numel()) for key, row in zip(names, stats)}, selected

    @torch.no_grad()
    def _module_snapshot(self, optimizer, delta: torch.Tensor | None = None):
        """Stream per-parameter reductions; retain no model-sized state copies.

        The existing global update snapshot supplies ``delta`` views. Only a
        small row of scalars per parameter survives this loop. Initialized Adam
        state is observed even when its current gradient is absent, because it
        remains optimizer history; counts distinguish the two populations.
        """
        if not self.module_statistics:
            return {}
        rows, descriptors = [], []
        for pi, ((name, parameter), gi) in enumerate(zip(self._parameters, self._groups)):
            zero = torch.zeros((), dtype=torch.float64, device=parameter.device)
            parameter_stats = _squares_and_nonfinite(parameter)
            gradient_stats = _squares_and_nonfinite(parameter.grad) if parameter.grad is not None else torch.stack((zero, zero))
            start, stop = self._offsets[pi]
            delta_stats = _squares_and_nonfinite(delta[start:stop]) if delta is not None else torch.stack((zero, zero))
            group = optimizer.param_groups[gi]
            state = optimizer.state.get(parameter, {})
            step = float(state.get("step", 0))
            initialized = step > 0 and "exp_avg" in state and "exp_avg_sq" in state
            adam_values = torch.stack((zero,) * 8)
            if initialized:
                beta1, beta2 = group["betas"]
                m_hat = state["exp_avg"].detach().double() / (1.0 - beta1 ** step)
                invalid = ~torch.isfinite(m_hat)
                m_squared = m_hat.square().sum()
                del m_hat
                v_hat = state["exp_avg_sq"].detach().double() / (1.0 - beta2 ** step)
                invalid.logical_or_(~torch.isfinite(v_hat))
                v_squared = v_hat.square().sum()
                sqrt_v_hat = v_hat.sqrt_()
                invalid.logical_or_(~torch.isfinite(sqrt_v_hat))
                sqrt_v_squared = sqrt_v_hat.square().sum()
                if group.get("amsgrad", False):
                    denominator = state["max_exp_avg_sq"].detach().double() / (1.0 - beta2 ** step)
                    denominator.sqrt_().add_(group["eps"])
                else:
                    denominator = sqrt_v_hat.add_(group["eps"])
                del sqrt_v_hat, v_hat
                invalid.logical_or_(~torch.isfinite(denominator))
                denominator_min, denominator_max = denominator.min(), denominator.max()
                inverse = denominator.reciprocal_()
                invalid.logical_or_(~torch.isfinite(inverse))
                adam_values = torch.stack((m_squared, v_squared, sqrt_v_squared,
                                           denominator_min, denominator_max,
                                           inverse.square().sum(), inverse.max(),
                                           invalid.sum().double()))
                del denominator, inverse, invalid
            rows.append(torch.cat((parameter_stats, gradient_stats, delta_stats, adam_values)))
            descriptors.append((_parameter_group(name), parameter.numel(), parameter.grad is not None,
                                initialized, step, gi, bool(group.get("amsgrad", False))))
        summaries: dict[str, Any] = {}
        values = torch.stack(rows).cpu().tolist() if rows else []
        for (name, count, active, initialized, step, gi, amsgrad), row in zip(descriptors, values):
            item = summaries.setdefault(name, {
                "elements": 0, "parameter_tensors": 0, "active_gradient_elements": 0,
                "active_gradient_tensors": 0, "parameter_l2_squared": 0.0,
                "gradient_l2_squared": 0.0, "delta_l2_squared": 0.0,
                "parameter_nonfinite_elements": 0, "gradient_nonfinite_elements": 0,
                "delta_nonfinite_elements": 0, "optimizer_groups": [],
                "adam": {"initialized_elements": 0, "initialized_tensors": 0,
                         "uninitialized_elements": 0, "uninitialized_tensors": 0,
                         "amsgrad_elements": 0, "m_hat_l2_squared": 0.0,
                         "v_hat_l2_squared": 0.0, "sqrt_v_hat_l2_squared": 0.0,
                         "inverse_denominator_l2_squared": 0.0,
                         "denominator_min": None, "denominator_max": None,
                         "inverse_denominator_max": None, "nonfinite_elements": 0,
                         "state_step_min": None, "state_step_max": None}})
            item["elements"] += count
            item["parameter_tensors"] += 1
            item["active_gradient_elements"] += count if active else 0
            item["active_gradient_tensors"] += int(active)
            if gi not in item["optimizer_groups"]:
                item["optimizer_groups"].append(gi)
            for key, value in zip(("parameter_l2_squared", "parameter_nonfinite_elements",
                                   "gradient_l2_squared", "gradient_nonfinite_elements",
                                   "delta_l2_squared", "delta_nonfinite_elements"), row[:6]):
                item[key] += int(value) if key.endswith("elements") else value
            adam = item["adam"]
            if not initialized:
                adam["uninitialized_elements"] += count
                adam["uninitialized_tensors"] += 1
                continue
            adam["initialized_elements"] += count
            adam["initialized_tensors"] += 1
            adam["amsgrad_elements"] += count if amsgrad else 0
            for key, value in zip(("m_hat_l2_squared", "v_hat_l2_squared", "sqrt_v_hat_l2_squared",
                                   "inverse_denominator_l2_squared"), (row[6], row[7], row[8], row[11])):
                adam[key] += value
            for key, value, operation in (("denominator_min", row[9], min),
                                          ("denominator_max", row[10], max),
                                          ("inverse_denominator_max", row[12], max),
                                          ("state_step_min", step, min), ("state_step_max", step, max)):
                previous = adam[key]
                adam[key] = value if previous is None else (operation(previous, value)
                                  if math.isfinite(previous) and math.isfinite(value) else float("nan"))
            adam["nonfinite_elements"] += int(row[13])
        for item in summaries.values():
            for dictionary in (item, item["adam"]):
                for key in list(dictionary):
                    if key.endswith("_l2_squared"):
                        dictionary[key.removesuffix("_squared")] = math.sqrt(dictionary.pop(key))
            if not item["adam"]["initialized_elements"]:
                for key in ("m_hat_l2", "v_hat_l2", "sqrt_v_hat_l2", "inverse_denominator_l2"):
                    item["adam"][key] = None
            if delta is None:
                item.pop("delta_l2")
                item.pop("delta_nonfinite_elements")
        return summaries

    @torch.no_grad()
    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.enabled or trainer.global_rank != 0:
            return
        if self._pending is not None:
            raise RuntimeError("Previous optimizer update has no post-step event")
        old, stats, coordinates = self._optimizer_snapshot(optimizer)
        self._pending = {"old": old, "statistics": stats, "coordinates": coordinates,
                         "modules": self._module_snapshot(optimizer),
                         "epoch": int(trainer.current_epoch), "global_step_before": int(trainer.global_step),
                         "groups": self._group_settings(optimizer), "rank0_microbatches": list(self._window)}

    @torch.no_grad()
    def _after_optimizer_step(self, optimizer, args, kwargs):
        if self._pending is None:
            raise RuntimeError("AdamW update was not preceded by Lightning's pre-clip callback")
        pending = self._pending
        pending["optimizer_step_returned"] = True
        new, after, coordinates = self._optimizer_snapshot(optimizer)
        delta = new - pending["old"]
        update_stats = _stat_dict(_stats(delta).cpu().tolist(), delta.numel())
        parameter_norm = pending["statistics"]["parameter"]["l2"]
        identity = {"epoch": pending["epoch"], "global_step_before": pending["global_step_before"],
                    "expected_global_step_after": pending["global_step_before"] + 1, "rank": 0}
        row = {**identity, "optimizer_groups": pending["groups"], "rank0_microbatches": pending["rank0_microbatches"],
               "gradient_l2_before_clip": pending["statistics"]["gradient"]["l2"],
               "gradient_l2_after_clip": after["gradient"]["l2"],
               "parameter_update": update_stats, "relative_parameter_update": update_stats["l2"] / parameter_norm if parameter_norm else None,
               "statistics_before": pending["statistics"], "statistics_after": after,
               "statistics_note": "Inactive gradient and uninitialized state slots contribute zero; denominator zero slots are not actual denominators"}
        if self.module_statistics:
            after_modules = self._module_snapshot(optimizer, delta=delta)
            row["module_statistics"] = {}
            for name, before_module in pending["modules"].items():
                after_module = after_modules[name]
                norm = before_module["parameter_l2"]
                row["module_statistics"][name] = {
                    "parameter_l2_before": norm,
                    "parameter_l2_after": after_module["parameter_l2"],
                    "gradient_l2_before_clip": before_module["gradient_l2"],
                    "gradient_l2_after_clip": after_module["gradient_l2"],
                    "parameter_update_l2": after_module["delta_l2"],
                    "relative_parameter_update": after_module["delta_l2"] / norm if norm else None,
                    "update_nonfinite_elements": after_module["delta_nonfinite_elements"],
                    "before": before_module, "after": after_module}
        row["finite"] = _finite(row)
        self._write("optimizer_updates", row)
        before = pending["coordinates"]
        decay = [-pending["groups"][self._groups[pi]]["lr"] * pending["groups"][self._groups[pi]]["weight_decay"] * old
                 if active else 0.0 for pi, old, active in zip(self._coordinate_parameters, before["parameter"], before["active_gradients"])]
        change = delta[self._coordinate_indices].double().cpu().tolist()
        coordinate_row = {**identity, "before": before, "after": coordinates, "parameter_delta": change,
                          "adamw_decay_delta_formula": decay,
                          "remaining_delta_including_rounding": [actual - shrink for actual, shrink in zip(change, decay)]}
        coordinate_row["finite"] = _finite(coordinate_row)
        self._write("coordinates", coordinate_row)
        self._window.clear()
        self._pending = None

    def _write(self, name, row):
        self._files[name].write(json.dumps(_safe(row), ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
        self._counts[name] += 1
        if self._counts[name] % self.flush_every == 0:
            self._files[name].flush()

    def _event(self, event, **values):
        if "events" in self._files:
            self._write("events", {"event": event, "utc": datetime.now(timezone.utc).isoformat(), **values})

    def _flush(self, durable=False):
        for stream in self._files.values():
            stream.flush()
            if durable:
                try:
                    os.fsync(stream.fileno())
                except (OSError, AttributeError):
                    pass

    def on_train_epoch_end(self, trainer, pl_module):
        if self.enabled:
            self._event("train_epoch_ended", epoch=int(trainer.current_epoch), global_step=int(trainer.global_step))
            self._flush(durable=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.enabled:
            self._flush(durable=True)

    def on_exception(self, trainer, pl_module, exception):
        if self.enabled:
            # The pending old-parameter tensor exists only to measure a normal
            # update. Preserve its scalar diagnostics, never serialize it.
            evidence = ({key: value for key, value in self._pending.items() if key != "old"}
                        if self._pending is not None else None)
            try:
                self._event("exception", error_type=type(exception).__name__, message=str(exception),
                            epoch=int(trainer.current_epoch), global_step=int(trainer.global_step),
                            active_contexts=self._contexts, optimizer_update_pending=self._pending is not None,
                            pending_optimizer_update=evidence, logging_error=self._logging_error)
            except Exception as error:
                self._logging_error = f"Exception evidence: {type(error).__name__}: {error}"
            finally:
                self.close(suppress_errors=True)

    def teardown(self, trainer, pl_module, stage):
        self.close()

    def close(self, suppress_errors=False):
        errors = []

        def attempt(function):
            try:
                function()
            except Exception as error:
                errors.append(error)

        if self._hook is not None:
            attempt(self._hook.remove)
            self._hook = None
        for hook in self._workload_hooks:
            attempt(hook.remove)
        self._workload_hooks.clear()
        self._workload = None
        if self._files:
            attempt(lambda: self._event("session_closed", rows=self._counts.copy()))
            attempt(lambda: self._flush(durable=True))
            for stream in self._files.values():
                attempt(stream.close)
        self._files.clear()
        self._counts.clear()
        self._pending = None
        self._contexts.clear()
        self._window.clear()
        self._parameters.clear()
        self._groups.clear()
        self._offsets.clear()
        self._coordinate_parameters.clear()
        self._coordinate_indices = None
        self._trainer = self._module = None
        if errors and not suppress_errors:
            raise errors[0]
