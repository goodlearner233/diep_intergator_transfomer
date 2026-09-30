"""Bounded two-process Gloo regression for actual loss and gradient guards.

Only the model forward is replaced by small deterministic synthetic outputs.
AlignedModule.step, FiniteGradients, and Telemetry are the production classes.
"""
from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "training_official_aligned"))


def read_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def worker(rank, output, port):
    import lightning as L
    import torch
    import torch.distributed as dist
    from train import AlignedModule, FiniteGradients, Telemetry

    class SyntheticAligned(AlignedModule):
        def __init__(self, nonfinite_energy=False):
            # Avoid expensive graph construction while keeping the exact
            # production step, loss, collective guard and logging path.
            L.LightningModule.__init__(self)
            self.scale = torch.nn.Parameter(torch.tensor(0.2))
            self.nonfinite_energy = nonfinite_energy
            self.energy_weight, self.force_weight, self.stress_weight = 1.0, 1.0, 0.1
            self.loss_params = {"delta": 1.0}
            self.phase_stats = {}

        def forward(self, graph, lattice, state):
            energy = self.scale.expand(2)
            if self.nonfinite_energy:
                energy = energy * torch.tensor(float("nan"))
            return energy, self.scale.expand(3, 3), self.scale.expand(6, 3)

    torch.set_num_threads(1)
    # Windows FileStore cannot reliably encode this repository's Chinese path.
    store = dist.TCPStore("127.0.0.1", port, 2, rank == 0,
                          timeout=datetime.timedelta(seconds=25), use_libuv=False)
    dist.init_process_group("gloo", rank=rank, world_size=2, store=store,
                            timeout=datetime.timedelta(seconds=25))
    results = []
    try:
        for phase in ("val", "test", "train_gradient"):
            model = SyntheticAligned(nonfinite_energy=rank == 1 and phase != "train_gradient")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True)
            observer = Telemetry(output / phase)
            trainer = SimpleNamespace(global_rank=rank, world_size=2, precision="32-true",
                                      optimizers=[optimizer], current_epoch=2, global_step=3,
                                      sanity_checking=False, callbacks=[observer])
            model.trainer = trainer
            actual_phase = "train" if phase == "train_gradient" else phase
            model.start_phase(actual_phase)
            graph = SimpleNamespace(batch=torch.tensor([0, 1, 1]), sample_idx=torch.tensor([10 * rank, 10 * rank + 1]),
                                    metric_weight=torch.ones(2), edge_index=torch.tensor([[0, 1, 2], [0, 2, 1]]))
            batch = (graph, torch.eye(3).expand(2, 3, 3), torch.zeros(2),
                     torch.zeros(2), torch.zeros(3, 3), torch.zeros(6, 3))
            observer.setup(trainer, model, "fit" if phase != "test" else "test")
            if phase == "train_gradient":
                observer.on_fit_start(trainer, model)
                observer.on_train_batch_start(trainer, model, batch, 5)
            elif phase == "val":
                observer.on_validation_batch_start(trainer, model, batch, 5)
            else:
                observer.on_test_batch_start(trainer, model, batch, 5)
            before = model.scale.detach().clone()
            caught = False
            message = None
            try:
                outputs = model.step(batch, actual_phase)
                if phase == "train_gradient":
                    outputs["loss"].backward()
                    # Only rank 1 is faulty; the real collective gradient guard
                    # must still prevent BOTH ranks from reaching AdamW.step.
                    if rank == 1:
                        model.scale.grad.fill_(float("nan"))
                    observer.on_before_optimizer_step(trainer, model, optimizer)
                    FiniteGradients().on_before_optimizer_step(trainer, model, optimizer)
                    optimizer.step()
            except FloatingPointError as error:
                caught, message = True, str(error)
                observer.on_exception(trainer, model, error)
            finally:
                observer.close()
            assert caught, f"rank {rank} did not reject {phase}"
            assert torch.equal(before, model.scale.detach()), f"rank {rank} updated despite {phase} failure"
            assert not optimizer.state, "AdamW must not initialize/update state after rejection"
            events = read_rows(observer.session_dir / "events.jsonl.gz")
            failure = next(row for row in events if row["event"] == "exception")
            assert failure["error_type"] == "FloatingPointError"
            if phase != "train_gradient":
                batches = read_rows(observer.session_dir / "batches.jsonl.gz")
                assert len(batches) == 1
                record = batches[0]
                assert record["status"] == "failed" and record["phase"] == phase
                assert record["sample_ids"] == [10 * rank, 10 * rank + 1]
                assert record["finite"] == (rank == 0)
                assert record["batch_idx"] == 5
            else:
                assert failure["active_contexts"]["train"]["sample_ids"] == [10 * rank, 10 * rank + 1]
                if rank == 0:
                    pending = failure["pending_optimizer_update"]
                    assert pending is not None and "old" not in pending
                    assert not pending.get("optimizer_step_returned", False)
                    assert not read_rows(observer.session_dir / "optimizer_updates.jsonl.gz")
            results.append({"phase": phase, "rank": rank, "collective_rejection": True,
                            "parameters_unchanged": True, "optimizer_untouched": True,
                            "message": message, "session": str(observer.session_dir)})
            dist.barrier()
        (output / f"rank{rank}.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    if args.worker is not None:
        try:
            worker(args.worker, args.output, args.port)
        except Exception:
            traceback.print_exc()
            return 1
        return 0
    output = HERE / f"distributed_failure_validation_{time.time_ns()}"
    output.mkdir()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", MKL_THREADING_LAYER="SEQUENTIAL",
               PYTHONIOENCODING="utf-8", USE_LIBUV="0")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    children = [subprocess.Popen([sys.executable, "-X", "utf8", str(Path(__file__).resolve()),
                                  "--worker", str(rank), "--output", str(output), "--port", str(port)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace", env=env) for rank in (0, 1)]
    deadline = time.monotonic() + 90
    try:
        for rank, child in enumerate(children):
            text, _ = child.communicate(timeout=max(1, deadline - time.monotonic()))
            (output / f"rank{rank}.log").write_text(text, encoding="utf-8")
            assert child.returncode == 0, f"rank {rank} failed: {text[-5000:]}"
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
    records = [json.loads((output / f"rank{rank}.json").read_text(encoding="utf-8")) for rank in (0, 1)]
    result = {"passed": True, "world_size": 2, "backend": "gloo", "cases_per_rank": 3,
              "checks": ["rank1 NaN validation loss rejects both ranks and logs both failed batches",
                         "rank1 NaN test loss rejects both ranks and logs both failed batches",
                         "actual FiniteGradients rejects both ranks before optimizer update",
                         "parameters and optimizer state unchanged on every rejection"],
              "results": records, "output": str(output)}
    (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
