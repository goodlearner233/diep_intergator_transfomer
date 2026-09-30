"""Real MatPES fresh/resume/best-test/export and observer-invariance integration.

Uses only 24 already-existing local real structures, never the full dataset.
Run with the project's training environment. Results stay beside this file.
"""
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENTRY = ROOT / "training_official_aligned" / "train.py"
SAMPLE = HERE.parent / "官方训练入口对照_20260930" / "sample_24.json"
REFS = HERE / "MatPES-PBE-atoms.json"


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for x, y in zip(left, right):
            same(x, y)
    else:
        assert left == right


def run(output, extra, label, expected=0):
    env = dict(os.environ, MKL_THREADING_LAYER="SEQUENTIAL", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    env.pop("DIEP_ALIGNED_RUN_TOKEN", None)
    command = [sys.executable, "-X", "utf8", str(ENTRY), "--data", str(SAMPLE),
               "--element-refs", str(REFS), "--output-dir", str(output),
               "--accelerator", "cpu", "--devices", "1", "--epochs", "2",
               "--max-atoms-per-batch", "100", "--step-checkpoint-every", "1",
               "--no-progress-bar", *extra]
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)
    (output.parent / f"{label}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if expected == 0:
        assert result.returncode == 0, result.stderr[-6000:]
    else:
        assert result.returncode != 0
    return result


def main():
    suite = HERE / f"entry_validation_{time.time_ns()}"
    suite.mkdir()
    observed, quiet = suite / "observed", suite / "quiet"
    run(observed, ["--stop-after-epochs", "1"], "fresh")
    first = torch.load(observed / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    assert first["epoch"] == 0 and first["global_step"] == 2
    assert abs(first["optimizer_states"][0]["param_groups"][0]["lr"] - 0.000505) < 0.000000000001
    assert not (observed / "final_test.json").exists()
    run(quiet, ["--stop-after-epochs", "1", "--no-telemetry"], "observer_off")
    other = torch.load(quiet / "checkpoints" / "last.ckpt", map_location="cpu", weights_only=False)
    same(first["state_dict"], other["state_dict"])
    same(first["optimizer_states"], other["optimizer_states"])
    checkpoint = observed / "checkpoints" / "last.ckpt"
    run(observed, ["--resume", str(checkpoint)], "resumed")
    status = json.loads((observed / "training_status.json").read_text())
    assert status["completed_epochs"] == 2 and status["global_step"] == 4
    assert abs(status["lr_after_completed_epochs"] - 0.00001) < 0.000000000001
    history = [json.loads(line) for line in (observed / "epoch_history.jsonl").read_text().splitlines()]
    assert [row["epoch"] for row in history] == [0, 1]
    assert all("train_Total_Loss" in row and "val_Total_Loss" in row for row in history)
    final = json.loads((observed / "final_test.json").read_text())
    best = torch.load(final["checkpoint"], map_location="cpu", weights_only=False)
    assert abs(status["best_val_Total_Loss"] - min(row["val_Total_Loss"] for row in history)) < 0.0000001
    exported = torch.load(observed / "best_potential" / "state.pt", map_location="cpu", weights_only=False)
    expected = {k.removeprefix("model."): v for k, v in best["state_dict"].items()}
    same(expected, exported)
    assert "data_std" in exported and any("element_refs" in k for k in exported)
    for path in observed.glob("*/*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        assert rows
    overwritten = run(observed, [], "fresh_overwrite_rejected", expected=1)
    assert "Output already belongs to a run" in overwritten.stderr
    mismatch = run(observed, ["--resume", str(checkpoint), "--lr", "0.0005"],
                   "wrong_resume_rejected", expected=1)
    assert "configuration/source/data changed" in mismatch.stderr
    result = {"passed": True, "real_samples": 24,
              "checks": ["real EFS fresh fit", "real EFS observer on/off bitwise weights and Adam state",
                         "same-run resume preserves optimizer and cosine position", "no early final test",
                         "explicit best-validation checkpoint final test", "full Potential export equals checkpoint",
                         "scale and refs retained", "epoch history current metrics", "gzip logs readable",
                         "fresh overwrite rejected", "changed learning rate resume rejected"],
              "output": str(observed)}
    (suite / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
