"""Actual-Lightning emergency checkpoint resume timing at mid/last batch."""
import json
from pathlib import Path
import shutil
import time

from test_lightning_200_epoch_schedule import HERE, TinyPotential, entry, run, torch


def main():
    torch.set_num_threads(1)
    entry.build_potential = lambda *args: TinyPotential()
    root = HERE / f"emergency_resume_schedule_{time.time_ns()}"
    root.mkdir()
    cases = {}
    for label, stop, every, expected_source_step in [("last_batch", 1, 2, 2), ("mid_epoch", 2, 3, 3)]:
        folder = root / label
        first_model, first_rows, first, _ = run(folder, stop, emergency_every=every)
        assert first["aligned_resume_scope"] == "epoch-boundary"
        path = folder / "checkpoints/emergency/step-latest.ckpt"
        source_path = folder / "source-emergency.ckpt"
        shutil.copy2(path, source_path)
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        assert source["global_step"] == expected_source_step
        assert source["aligned_resume_scope"] == "diagnostic-only"
        source_loop = source["loops"]["fit_loop"]
        record = {"source_epoch": source["epoch"], "source_global_step": source["global_step"],
                  "source_scheduler_last_epoch": source["lr_schedulers"][0]["last_epoch"],
                  "source_epoch_progress": source_loop["epoch_progress"]["current"],
                  "source_batch_progress": source_loop["epoch_loop.batch_progress"],
                  "source_scheduler_progress": source_loop["epoch_loop.scheduler_progress"]}
        try:
            resumed_model, rows, final, best = run(folder, 3, source_path, emergency_every=every)
            record.update(resume_success=True, epoch_start_rows=rows,
                          resumed_scheduler_calls=resumed_model.scheduler_calls,
                          final_epoch=final["epoch"], final_global_step=final["global_step"],
                          final_scheduler_last_epoch=final["lr_schedulers"][0]["last_epoch"],
                          final_lr=final["optimizer_states"][0]["param_groups"][0]["lr"],
                          best_checkpoint=best)
        except Exception as error:
            record.update(resume_success=False, error=repr(error))
            assert isinstance(error, ValueError) and "diagnostic-only" in str(error), repr(error)
        else:
            raise AssertionError("Incomplete-epoch diagnostic snapshot was accepted for training resume")
        record["passed_explicit_rejection"] = True
        cases[label] = record
    (root / "result.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(root), "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
