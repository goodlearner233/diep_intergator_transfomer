"""Targeted actual-Lightning resume edge cases; reuses the 200-epoch audit output."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from test_lightning_200_epoch_schedule import (
    HERE, TinyPotential, TracedModule, entry, run, assert_same, torch,
)


def main():
    root = Path(sys.argv[1]).resolve()
    entry.build_potential = lambda *args: TinyPotential()
    torch.set_num_threads(1)
    old = torch.load(root / "uninterrupted/checkpoints/last.ckpt", map_location="cpu", weights_only=False)
    model, rows, complete, selected = run(root / "uninterrupted", 200,
                                          root / "uninterrupted/checkpoints/last.ckpt")
    assert rows == [] and model.scheduler_calls == []
    assert_same(old["optimizer_states"], complete["optimizer_states"])
    assert_same(old["lr_schedulers"], complete["lr_schedulers"])
    assert_same(old["state_dict"], complete["state_dict"])
    assert Path(selected).is_file()
    try:
        run(root / "uninterrupted", 199, root / "uninterrupted/checkpoints/last.ckpt")
    except ValueError as error:
        past_limit_error = str(error)
        assert "Resume stop limit 199" in past_limit_error and "200 completed epochs" in past_limit_error
    else:
        raise AssertionError("Production on_load_checkpoint failed to reject a past stop limit")
    # The rolling save is inside epoch 199. It is now explicitly diagnostic-only,
    # rather than pretending its missing partial-epoch metrics can be resumed.
    emergency = torch.load(root / "uninterrupted/checkpoints/emergency/step-latest.ckpt",
                           map_location="cpu", weights_only=False)
    probe = TracedModule()
    probe._trainer = SimpleNamespace(state=SimpleNamespace(fn="fit"), max_epochs=199)
    try:
        probe.on_load_checkpoint(emergency)
    except ValueError as error:
        assert "diagnostic-only" in str(error)
    else:
        raise AssertionError("Incomplete-epoch snapshot was accepted for training resume")
    probe._trainer = SimpleNamespace(state=SimpleNamespace(fn="test"), max_epochs=1)
    probe.on_load_checkpoint(old)  # Validation/test restore is independent of a fit stop limit.
    probe.on_load_checkpoint(emergency)  # The diagnostic snapshot remains usable for inspection/evaluation.

    original = TracedModule.on_validation_epoch_end
    def worsening_validation(self):
        stats = self.phase_stats["val"]
        stats[21] = stats[22] * (self.current_epoch + 1)
        return original(self)
    TracedModule.on_validation_epoch_end = worsening_validation
    early_folder = root / f"early_best_boundaries_{__import__('time').time_ns()}"
    _, _, first, early_best = run(early_folder, 2)
    best_cp = torch.load(early_best, map_location="cpu", weights_only=False)
    assert best_cp["epoch"] == 0
    continued, continue_rows, final, kept_best = run(early_folder, 4,
                                                   early_folder / "checkpoints/last.ckpt")
    assert kept_best == early_best and continue_rows[0]["epoch"] == 2
    assert final["lr_schedulers"][0]["last_epoch"] == 4
    # Explicitly resuming a prior best is supported by Lightning's counters.
    # This is a rewind, so the same output's append-only epoch history contains
    # repeated epoch numbers; callers must use last for ordinary continuation.
    rewound, rewind_rows, rewind_final, rewind_best = run(early_folder, 3, Path(early_best))
    assert rewind_rows[0]["epoch"] == 1
    assert rewind_rows[0]["scheduler_last_epoch"] == 1
    assert rewind_best == early_best
    result = {"passed": True, "completed_200_resume": "No updates or extra scheduler advance; prior best restored",
              "entry_rejects_stop_limit_before_completed_epochs": past_limit_error,
              "mid_epoch_emergency_rejected_as_diagnostic_only": True,
              "test_restore_is_independent_of_fit_stop_limit": True,
              "early_best_retained_when_future_validation_worsens": True,
              "explicit_early_best_resume_first_epoch": rewind_rows[0],
              "caveat": "Resuming a historical checkpoint into an already more-advanced output rewinds training and appends duplicate epoch history; ordinary continuation should use last.ckpt."}
    (root / "resume_boundaries_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
