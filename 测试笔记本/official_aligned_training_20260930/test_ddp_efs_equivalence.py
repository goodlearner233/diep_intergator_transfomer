"""Real MatPES E/F/S: two-rank Gloo update versus averaged local-batch losses.

Diagnostic only; never edits production files or performs a campaign training run.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import socket
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "training_official_aligned"))

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from ase.stress import voigt_6_to_full_3x3_stress
from matgl.config import DEFAULT_ELEMENTS
from matgl.graph.data import collate_fn_pes
from matgl.utils.training import MGLDatasetLoader
from data_pipeline import load_element_references
from train import build_potential, math_attention


def loss(model, batch):
    graph, lat, state, true_e, true_f, true_s = batch
    pred_e, pred_f, pred_s, *_ = model(g=graph, lat=lat, state_attr=state)
    counts = torch.bincount(graph.batch)
    huber = torch.nn.functional.huber_loss
    values = (huber(pred_e.reshape(-1) / counts, true_e.reshape(-1) / counts),
              huber(pred_f, true_f), huber(pred_s, true_s))
    return values[0] + values[1] + 0.1 * values[2]


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),
                             eps=0.00000001, weight_decay=0.00001, amsgrad=True,
                             foreach=False)


def state(model, opt, gradients, total_loss, before_clip):
    return {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(opt.state_dict()),
            "gradients": gradients, "loss": float(total_loss), "before_clip": float(before_clip)}


def worker(rank, folder):
    torch.set_num_threads(1)
    folder = Path(folder)
    package = torch.load(folder / "inputs.pt", weights_only=False)
    dist.init_process_group("gloo", init_method=package["rendezvous"],
                            rank=rank, world_size=2)
    try:
        model = build_potential(package["rms"], package["refs"])
        model.load_state_dict(package["initial"])
        model.train()
        wrapped = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        opt = optimizer(model)
        with math_attention():
            value = loss(wrapped, package["batches"][rank])
            value.backward()
        grads = {name: None if p.grad is None else p.grad.detach().clone()
                 for name, p in model.named_parameters()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        torch.save(state(model, opt, grads, value, norm), folder / f"rank_{rank}.pt")
    finally:
        dist.destroy_process_group()


def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    folder = HERE / f"ddp_equivalence_{time.time_ns()}"
    folder.mkdir()
    data_path = ROOT / "测试笔记本/官方训练入口对照_20260930/sample_24.json"
    records = json.loads(data_path.read_text(encoding="utf-8"))[:5]
    sample_path = folder / "sample_5.json"
    sample_path.write_text(json.dumps(records), encoding="utf-8")
    dataset = MGLDatasetLoader.from_json(sample_path, root=str(folder / "cache"),
                                         element_types=DEFAULT_ELEMENTS, cutoff=5.0,
                                         stress_unit="kbar")
    for i, record in enumerate(records):
        raw = np.asarray(record["stress"], dtype=float)
        dataset.labels["stresses"][i] = torch.tensor(
            (voigt_6_to_full_3x3_stress(raw) if raw.shape == (6,) else raw) * -0.1,
            dtype=torch.float32)
    batches = [collate_fn_pes([dataset[i] for i in group]) for group in ([0, 1], [2, 3, 4])]
    force = torch.cat([batch[4] for batch in batches])
    rms = float((force.square().sum() / force.shape[0]).sqrt())
    refs_map = load_element_references(HERE / "MatPES-PBE-atoms.json")
    refs = [refs_map.get(element, 0.0) for element in DEFAULT_ELEMENTS]
    sequential = build_potential(rms, refs)
    sequential.train()
    initial = copy.deepcopy(sequential.state_dict())
    with socket.socket() as temporary_socket:
        temporary_socket.bind(("127.0.0.1", 0))
        port = temporary_socket.getsockname()[1]
    torch.save({"rms": rms, "refs": refs, "initial": initial, "batches": batches,
                "rendezvous": f"tcp://127.0.0.1:{port}"}, folder / "inputs.pt")
    opt = optimizer(sequential)
    values = []
    with math_attention():
        for batch in batches:
            value = loss(sequential, batch)
            values.append(float(value))
            (value / 2).backward()
    grads = {name: None if p.grad is None else p.grad.detach().clone()
             for name, p in sequential.named_parameters()}
    norm = torch.nn.utils.clip_grad_norm_(sequential.parameters(), 2.0)
    opt.step()
    reference = state(sequential, opt, grads, sum(values) / 2, norm)
    torch.save(reference, folder / "sequential.pt")
    mp.spawn(worker, args=(str(folder),), nprocs=2, join=True)
    distributed = [torch.load(folder / f"rank_{rank}.pt", weights_only=False) for rank in range(2)]
    comparisons = {}
    for rank, output in enumerate(distributed):
        maximum_gradient, maximum_parameter, maximum_state = 0.0, 0.0, 0.0
        for name, expected in reference["gradients"].items():
            actual = output["gradients"][name]
            assert (expected is None) == (actual is None), name
            if expected is not None:
                maximum_gradient = max(maximum_gradient, float((actual - expected).abs().max()))
                torch.testing.assert_close(actual, expected, rtol=0.00005, atol=0.000002,
                                           msg=lambda msg: f"gradient {name}: {msg}")
        for name, expected in reference["model"].items():
            actual = output["model"][name]
            if torch.is_floating_point(expected):
                maximum_parameter = max(maximum_parameter, float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, rtol=0.00005, atol=0.000005,
                                       msg=lambda msg: f"parameter {name}: {msg}")
        for index, expected_state in reference["optimizer"]["state"].items():
            for key, expected in expected_state.items():
                actual = output["optimizer"]["state"][index][key]
                if isinstance(expected, torch.Tensor):
                    maximum_state = max(maximum_state, float((actual - expected).abs().max()))
                    torch.testing.assert_close(actual, expected, rtol=0.00005, atol=0.000002,
                                               msg=lambda msg: f"optimizer {index}/{key}: {msg}")
        comparisons[str(rank)] = {"max_abs_gradient_difference": maximum_gradient,
                                  "max_abs_parameter_difference": maximum_parameter,
                                  "max_abs_optimizer_state_difference": maximum_state,
                                  "preclip_gradient_norm": output["before_clip"]}
    for key in distributed[0]["model"]:
        assert torch.equal(distributed[0]["model"][key], distributed[1]["model"][key]), key
    result = {"passed": True, "backend": "Gloo CPU", "world_size": 2,
              "structure_counts_per_rank": [int(b[3].numel()) for b in batches],
              "atom_counts_per_rank": [int(b[4].shape[0]) for b in batches],
              "force_rms": rms, "batch_losses": values,
              "sequential_mean_loss": reference["loss"], "sequential_preclip_norm": reference["before_clip"],
              "comparisons": comparisons,
              "scope": "One real E/F/S backward and AdamW AMSGrad update, math attention, unequal batches; not CUDA NCCL throughput or long-run validation"}
    (folder / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"folder": str(folder), **result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
