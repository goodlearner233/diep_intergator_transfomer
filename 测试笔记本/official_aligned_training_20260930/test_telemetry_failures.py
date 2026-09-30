"""Failure evidence survives early guards without changing training tensors."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from test_telemetry import Telemetry, Toy, collate, rows


class BrokenWrite:
    def __init__(self, stream):
        self.stream = stream

    def write(self, text):
        raise OSError("injected full disk")

    def flush(self):
        self.stream.flush()

    def fileno(self):
        return self.stream.fileno()

    def close(self):
        self.stream.close()


class TelemetryFailureTests(unittest.TestCase):
    def fixture(self, directory):
        torch.manual_seed(42)
        model = Toy()
        optimizer = model.configure_optimizers()
        trainer = SimpleNamespace(global_rank=0, world_size=1, precision="32-true", optimizers=[optimizer],
                                  current_epoch=3, global_step=7, sanity_checking=False)
        observer = Telemetry(directory)
        observer.setup(trainer, model, "fit")
        observer.on_fit_start(trainer, model)
        batch = collate([0])
        observer.on_train_batch_start(trainer, model, batch, 4)
        return model, optimizer, trainer, observer, batch

    def test_gradient_failure_preserves_pending_scalars_not_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer, trainer, observer, batch = self.fixture(directory)
            before = copy.deepcopy(model.state_dict())
            for parameter in model.parameters():
                parameter.grad = torch.ones_like(parameter)
            next(model.parameters()).grad.flatten()[0] = float("inf")
            observer.on_before_optimizer_step(trainer, model, optimizer)
            observer.on_exception(trainer, model, FloatingPointError("Nonfinite gradient: refusing update"))
            events = rows(observer.session_dir / "events.jsonl.gz")
            failure = next(event for event in events if event["event"] == "exception")
            pending = failure["pending_optimizer_update"]
            self.assertNotIn("old", pending)
            self.assertEqual(pending["statistics"]["gradient"]["nonfinite_elements"], 1)
            self.assertEqual(pending["modules"]["linear"]["gradient_nonfinite_elements"], 1)
            self.assertIn(None, pending["coordinates"]["gradient"])
            self.assertEqual(failure["active_contexts"]["train"]["sample_ids"], [0, 1])
            self.assertEqual(pending["global_step_before"], 7)
            self.assertFalse(rows(observer.session_dir / "optimizer_updates.jsonl.gz"))
            self.assertIsNone(observer._pending)
            self.assertFalse(observer._files)
            self.assertIsNone(observer._module)
            self.assertEqual(len(optimizer._optimizer_step_post_hooks), 0)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_failed_loss_batch_has_per_structure_errors_and_reported_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer, trainer, observer, batch = self.fixture(directory)
            outputs = model.calculate(batch)
            outputs["preds"][0].detach()[0] = float("inf")
            outputs["loss"] = torch.tensor(float("inf"))
            rng_before = torch.get_rng_state().clone()
            observer.record_failed_batch(trainer, model, "train", batch, outputs, "Nonfinite train loss")
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            observer.on_exception(trainer, model, FloatingPointError("Nonfinite train loss"))
            record = rows(observer.session_dir / "batches.jsonl.gz")[0]
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["failure_reason"], "Nonfinite train loss")
            self.assertEqual(record["batch_idx"], 4)
            self.assertEqual(record["sample_ids"], [0, 1])
            self.assertFalse(record["finite"])
            self.assertIsNone(record["reported_total_loss"])
            self.assertIsNone(record["energy_abs_error_per_atom"][0])
            self.assertEqual(len(record["force_mae_per_structure"]), 2)
            self.assertEqual(len(record["stress_mae_per_structure"]), 2)

    def test_bad_batch_logging_keeps_context_and_cannot_mask_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer, trainer, observer, batch = self.fixture(directory)
            observer.record_failed_batch(trainer, model, "train", batch, {}, "original training failure")
            self.assertIn("train", observer._contexts)
            self.assertIn("ValueError", observer._logging_error)
            observer.on_exception(trainer, model, FloatingPointError("original training failure"))
            failure = next(event for event in rows(observer.session_dir / "events.jsonl.gz") if event["event"] == "exception")
            self.assertEqual(failure["message"], "original training failure")
            self.assertEqual(failure["active_contexts"]["train"]["batch_idx"], 4)
            self.assertIn("ValueError", failure["logging_error"])

    def test_io_failure_still_closes_handles_and_cannot_mask_training_error(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer, trainer, observer, batch = self.fixture(directory)
            streams = list(observer._files.values())
            observer._files["events"] = BrokenWrite(observer._files["events"])
            observer.on_exception(trainer, model, FloatingPointError("original training failure"))
            self.assertTrue(all(stream.closed for stream in streams))
            self.assertEqual(len(optimizer._optimizer_step_post_hooks), 0)
            self.assertIsNone(observer._module)
            self.assertFalse(observer._files)
            observer.close()  # Repeated teardown is harmless.


if __name__ == "__main__":
    unittest.main(verbosity=2)
