"""Isolate optimizer-observer overhead on the real DIEPformer parameter layout.

This is NOT an E/F/S training or graph-memory benchmark. It uses deterministic
synthetic gradients and actual AdamW updates, no graph forward or backward.
Each mode/repetition runs in a fresh child process. CUDA peaks include observer
setup and every measured update; batch hooks are deliberately not called because
they reset the CUDA peak counter. CPU RSS sampling can miss short transients;
Windows peak_wset is also recorded, but covers the entire process lifetime.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
MODES = ("none", "base", "modules")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def worker(args):
    # Heavy imports happen only in children, outside the measured window.
    sys.path.insert(0, str(ROOT / "training_official_aligned"))
    import psutil
    import torch
    import numpy as np
    from train import AlignedModule, DEFAULT_ELEMENTS, Telemetry

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing to silently use CPU")
    device = torch.device(args.device)
    model = AlignedModule(1.0, np.zeros(len(DEFAULT_ELEMENTS)), signature="synthetic-memory-benchmark").to(device)
    optimizer = model.configure_optimizers()["optimizer"]
    parameters = [p for p in model.parameters() if p.requires_grad]
    for i, parameter in enumerate(parameters):
        parameter.grad = torch.full_like(parameter, ((i % 31) - 15) * 0.0001 + 0.00005)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def set_gradients(step):
        # Reuse allocations. Values vary by parameter tensor and step and are
        # identical in all modes. No RNG calls in the measured section.
        for i, parameter in enumerate(parameters):
            parameter.grad.fill_(((i % 31) - 15) * 0.0001 + (step + 1) * 0.00005)

    for step in range(args.warmup):
        set_gradients(step)
        optimizer.step()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if args.observer_warmup_updates:
        warm_trainer = SimpleNamespace(global_rank=0, world_size=1, precision="32-true",
                                       optimizers=[optimizer], current_epoch=0,
                                       global_step=args.warmup, sanity_checking=False)
        warm_observer = None
        if args.mode != "none":
            warm_observer = Telemetry(output / "observer_warmup_streams", module_statistics=args.mode == "modules",
                                      coordinate_count=128, flush_every=1)
            warm_observer.setup(warm_trainer, model, "fit")
            warm_observer.on_fit_start(warm_trainer, model)
        for step in range(args.warmup, args.warmup + args.observer_warmup_updates):
            set_gradients(step)
            if warm_observer is not None:
                warm_observer.on_before_optimizer_step(warm_trainer, model, optimizer)
            optimizer.step()
            warm_trainer.global_step += 1
        if warm_observer is not None:
            warm_observer.close()
        del warm_observer, warm_trainer
    synchronize()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    process = psutil.Process()
    baseline_info = process.memory_info()
    baseline_rss = int(baseline_info.rss)
    baseline_peak_wset = getattr(baseline_info, "peak_wset", None)
    rss_samples = [baseline_rss]
    stopped = threading.Event()

    def sample_rss():
        while not stopped.wait(0.002):
            rss_samples.append(int(process.memory_info().rss))

    monitor = threading.Thread(target=sample_rss, daemon=True)
    monitor.start()
    baseline_gpu = ({"allocated_bytes": int(torch.cuda.memory_allocated(device)),
                     "reserved_bytes": int(torch.cuda.memory_reserved(device))}
                    if device.type == "cuda" else None)
    trainer = SimpleNamespace(global_rank=0, world_size=1, precision="32-true",
                              optimizers=[optimizer], current_epoch=0,
                              global_step=args.warmup + args.observer_warmup_updates, sanity_checking=False)
    observer = None
    setup_started = time.perf_counter()
    if args.mode != "none":
        observer = Telemetry(output / "streams", module_statistics=args.mode == "modules",
                             coordinate_count=128, flush_every=1)
        observer.setup(trainer, model, "fit")
        observer.on_fit_start(trainer, model)
    synchronize()
    setup_seconds = time.perf_counter() - setup_started
    step_seconds = []
    measured_start = args.warmup + args.observer_warmup_updates
    for step in range(measured_start, measured_start + args.updates):
        synchronize()
        started = time.perf_counter()
        set_gradients(step)
        if observer is not None:
            observer.on_before_optimizer_step(trainer, model, optimizer)
        optimizer.step()  # Observer's post-step hook runs here when enabled.
        trainer.global_step += 1
        synchronize()
        step_seconds.append(time.perf_counter() - started)
        rss_samples.append(int(process.memory_info().rss))
    gpu = ({"baseline": baseline_gpu,
            "allocated_bytes_after_updates": int(torch.cuda.memory_allocated(device)),
            "reserved_bytes_after_updates": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device))}
           if device.type == "cuda" else None)
    if gpu:
        gpu["peak_allocated_above_baseline_bytes"] = gpu["peak_allocated_bytes"] - baseline_gpu["allocated_bytes"]
        gpu["peak_reserved_above_baseline_bytes"] = gpu["peak_reserved_bytes"] - baseline_gpu["reserved_bytes"]
    if observer is not None:
        observer.close()
    rss_samples.append(int(process.memory_info().rss))
    stopped.set()
    monitor.join()
    final_memory = process.memory_info()
    peak_wset = getattr(final_memory, "peak_wset", None)
    cpu = {"baseline_rss_bytes": baseline_rss, "sampled_peak_rss_bytes": max(rss_samples),
           "sampled_rss_peak_above_baseline_bytes": max(rss_samples) - baseline_rss,
           "final_rss_bytes": int(final_memory.rss), "rss_sample_count": len(rss_samples),
           "rss_sampling_interval_seconds": 0.002,
           "lifetime_peak_wset_before_observer_bytes": baseline_peak_wset,
           "lifetime_peak_wset_after_observer_bytes": peak_wset,
           "lifetime_peak_wset_increase_bytes": (peak_wset - baseline_peak_wset)
           if peak_wset is not None and baseline_peak_wset is not None else None}

    # Hash and decompress only after all memory measurements have ended.
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    optimizer_digest = hashlib.sha256()
    for parameter in parameters:
        for name, tensor in sorted(optimizer.state[parameter].items()):
            optimizer_digest.update(name.encode())
            optimizer_digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    streams = {}
    for path in sorted((output / "streams").rglob("*.jsonl.gz")):
        with gzip.open(path, "rb") as stream:
            raw = stream.read()
        streams[path.name] = {"compressed_bytes": path.stat().st_size,
                              "uncompressed_bytes": len(raw), "records": len(raw.splitlines())}
    disk = {"streams": streams, "total_compressed_bytes": sum(x["compressed_bytes"] for x in streams.values()),
            "total_uncompressed_bytes": sum(x["uncompressed_bytes"] for x in streams.values())}
    if observer is not None:
        assert streams["optimizer_updates.jsonl.gz"]["records"] == args.updates
        assert streams["coordinates.jsonl.gz"]["records"] == args.updates
    result = {"mode": args.mode, "device": args.device, "repetition": args.repetition,
              "seed": args.seed, "torch_version": str(torch.__version__),
              "cuda_version": torch.version.cuda,
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "platform": platform.platform(), "python": sys.version, "torch_cpu_threads": args.threads,
              "parameter_elements": sum(p.numel() for p in parameters),
              "parameter_tensors": len(parameters), "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
              "largest_parameter_elements": max(p.numel() for p in parameters),
              "largest_parameter_bytes": max(p.numel() * p.element_size() for p in parameters),
              "largest_parameter_name": max(((name, p.numel()) for name, p in model.named_parameters()
                                              if p.requires_grad), key=lambda item: item[1])[0],
              "optimizer": "AdamW FP32, amsgrad=True; same configuration as AlignedModule",
              "warmup_updates_without_observer": args.warmup, "measured_updates": args.updates,
              "warmup_updates_with_current_observer_mode": args.observer_warmup_updates,
              "observer_setup_seconds": setup_seconds, "update_seconds": step_seconds,
              "mean_update_seconds": statistics.mean(step_seconds), "cpu": cpu, "cuda_allocator": gpu,
              "disk": disk, "model_state_sha256": digest.hexdigest(),
              "optimizer_state_sha256": optimizer_digest.hexdigest(),
              "telemetry_source_sha256": hashlib.sha256((ROOT / "training_official_aligned" / "telemetry.py").read_bytes()).hexdigest(),
              "scope": "Actual DIEPformer parameter layout, synthetic fixed gradients, no graph/EFS forward/backward or batch records.",
              "limits": ["CUDA allocator peaks exclude CUDA context, libraries, driver, and other processes.",
                         "CPU sampled RSS can miss short transients; peak_wset is a process-lifetime high-water mark.",
                         "Memory includes observer setup and measured updates; timing separates setup from updates.",
                         "Disk totals include gzip headers/trailers and session metadata; bytes/update extrapolation is approximate."]}
    write_json(output / "result.json", result)
    print(json.dumps({"mode": args.mode, "repetition": args.repetition,
                      "result": str(output / "result.json")}, ensure_ascii=False))


def orchestrate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for repetition in range(args.repetitions):
        # Rotate order to reduce systematic temperature/cache/background bias.
        modes = MODES[repetition % len(MODES):] + MODES[:repetition % len(MODES)]
        for mode in modes:
            child_dir = output / f"{repetition + 1:02d}_{mode}"
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--mode", mode,
                       "--device", args.device, "--updates", str(args.updates), "--warmup", str(args.warmup),
                       "--observer-warmup-updates", str(args.observer_warmup_updates),
                       "--threads", str(args.threads), "--seed", str(args.seed),
                       "--repetition", str(repetition + 1), "--output", str(child_dir)]
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            run = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8")
            (output / f"{repetition + 1:02d}_{mode}.log").write_text(run.stdout + run.stderr, encoding="utf-8")
            if run.returncode:
                raise RuntimeError(f"Child {mode}/{repetition + 1} failed; see {output}")
            result = json.loads((child_dir / "result.json").read_text(encoding="utf-8"))
            results.append(result)
            print(run.stdout.strip(), flush=True)
    if len({r["model_state_sha256"] for r in results}) != 1 or len({r["optimizer_state_sha256"] for r in results}) != 1:
        raise AssertionError("Observer modes produced different model/optimizer states")
    if len({r["telemetry_source_sha256"] for r in results}) != 1:
        raise AssertionError("Telemetry source changed during benchmark; repeat against one source version")

    def metrics(result):
        values = {"cpu_sampled_rss_peak_above_baseline_bytes": result["cpu"]["sampled_rss_peak_above_baseline_bytes"],
                  "cpu_sampled_peak_rss_bytes": result["cpu"]["sampled_peak_rss_bytes"],
                  "disk_total_compressed_bytes": result["disk"]["total_compressed_bytes"],
                  "mean_update_seconds": result["mean_update_seconds"]}
        if result["cuda_allocator"]:
            values.update({"cuda_" + key: result["cuda_allocator"][key] for key in
                           ("peak_allocated_bytes", "peak_reserved_bytes", "peak_allocated_above_baseline_bytes")})
        return values

    metric_keys = list(metrics(results[0]))
    medians = {mode: {key: statistics.median(metrics(r)[key] for r in results if r["mode"] == mode)
                      for key in metric_keys} for mode in MODES}
    ranges = {mode: {key: {"min": min(metrics(r)[key] for r in results if r["mode"] == mode),
                           "max": max(metrics(r)[key] for r in results if r["mode"] == mode)}
                     for key in metric_keys} for mode in MODES}
    differences = {"modules_minus_" + baseline: {key: medians["modules"][key] - medians[baseline][key]
                                                for key in metric_keys} for baseline in ("none", "base")}
    report = {"benchmark": "Isolated optimizer telemetry, not full training", "results": results,
              "median_by_mode": medians, "range_by_mode": ranges, "median_differences": differences,
              "model_and_optimizer_states_equal_all_modes": True,
              "notes": ["Separate process per mode/repetition; deterministic gradients; all trainable parameters active.",
                        "Synthetic gradients and 100% active parameters do not predict graph/EFS training memory or throughput.",
                        "Subtracting process RSS medians is noisy; use baseline-subtracted sampled RSS and ranges.",
                        "GPU peaks are from the PyTorch allocator and cover setup plus all updates after warmup."]}
    write_json(output / "summary.json", report)
    print(json.dumps({"summary": str(output / "summary.json"), "median_differences": differences}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--observer-warmup-updates", type=int, default=0,
                        help="Optional discarded observer session before memory baseline, to separate first-use costs")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--repetition", type=int, default=1)
    args = parser.parse_args()
    if min(args.repetitions, args.updates, args.warmup, args.threads) < 1:
        parser.error("repetitions, updates, warmup, threads must be positive")
    if args.observer_warmup_updates < 0:
        parser.error("observer-warmup-updates must be nonnegative")
    if args.worker and args.mode is None:
        parser.error("--worker requires --mode")
    worker(args) if args.worker else orchestrate(args)


if __name__ == "__main__":
    main()
