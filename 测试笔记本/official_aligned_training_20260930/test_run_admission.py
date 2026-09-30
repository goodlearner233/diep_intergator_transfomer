"""Output ownership tests; no model training or original file changes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "training_official_aligned"))
from train import run_admission

RESULTS = HERE / f"run_admission_{time.time_ns()}"
RESULTS.mkdir()
LAUNCH_VARS = ("RANK", "LOCAL_RANK", "DIEP_ALIGNED_RUN_TOKEN", "TORCHELASTIC_RUN_ID",
               "TORCHELASTIC_ERROR_FILE", "WORLD_SIZE")


def clean_env(**updates):
    env = os.environ.copy()
    for name in LAUNCH_VARS:
        env.pop(name, None)
    env.update(updates)
    return env


class AdmissionTests(unittest.TestCase):
    def child(self, output, env, expected=0):
        completed = subprocess.run([sys.executable, "-X", "utf8", __file__, "--worker", str(output)],
                                   env=env, text=True, encoding="utf-8", capture_output=True, timeout=45)
        self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
        return completed.stdout.strip().splitlines()[-1]

    def test_fresh_uuid_for_static_torchrun_and_stale_inherited_token(self):
        output = RESULTS / "fresh_token"
        with patch.dict(os.environ, clean_env(TORCHELASTIC_RUN_ID="none", DIEP_ALIGNED_RUN_TOKEN="stale"), clear=True):
            with run_admission(output) as first:
                self.assertNotIn(first, ("none", "stale"))
            with run_admission(output) as second:
                self.assertNotEqual(first, second)
            self.assertFalse((output / ".active_run.json").exists())

    def test_reject_second_rank_zero_but_allow_inherited_lightning_worker(self):
        output = RESULTS / "lightning"
        with patch.dict(os.environ, clean_env(), clear=True):
            with run_admission(output) as token:
                self.child(output, clean_env(), expected=3)
                answer = self.child(output, clean_env(LOCAL_RANK="1", DIEP_ALIGNED_RUN_TOKEN=token))
                self.assertEqual(answer, token)

    def test_external_torchrun_workers_join_unique_attempt_with_static_id(self):
        output = RESULTS / "torchrun"
        attempt = RESULTS / "unique_torchrun_attempt"
        root_env = clean_env(RANK="0", LOCAL_RANK="0", WORLD_SIZE="2", TORCHELASTIC_RUN_ID="none",
                             TORCHELASTIC_ERROR_FILE=str(attempt / "0" / "error.json"))
        with patch.dict(os.environ, root_env, clear=True):
            with run_admission(output) as token:
                env = clean_env(RANK="1", LOCAL_RANK="1", WORLD_SIZE="2", TORCHELASTIC_RUN_ID="none",
                                TORCHELASTIC_ERROR_FILE=str(attempt / "1" / "error.json"))
                self.assertEqual(self.child(output, env), token)
                env["TORCHELASTIC_ERROR_FILE"] = str(RESULTS / "other_attempt" / "1" / "error.json")
                self.child(output, env, expected=3)

    def test_stale_descriptor_does_not_admit_worker_without_held_lock(self):
        output = RESULTS / "stale"
        output.mkdir()
        (output / ".active_run.json").write_text(json.dumps({"token": "stale", "launcher": None}))
        with patch.dict(os.environ, clean_env(LOCAL_RANK="1", DIEP_ALIGNED_RUN_TOKEN="stale"), clear=True):
            with self.assertRaisesRegex(RuntimeError, "could not join"):
                with run_admission(output, worker_timeout=0.1):
                    self.fail("Admitted stale worker")

    def test_release_after_failure_allows_future_owner(self):
        output = RESULTS / "exception"
        with patch.dict(os.environ, clean_env(), clear=True):
            with self.assertRaisesRegex(ValueError, "synthetic failure"):
                with run_admission(output):
                    raise ValueError("synthetic failure")
            self.assertFalse((output / ".active_run.json").exists())
            with run_admission(output):
                pass


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        try:
            with run_admission(sys.argv[2], worker_timeout=0.2) as token:
                print(token, flush=True)
        except RuntimeError as exc:
            print(str(exc), flush=True)
            raise SystemExit(3)
    else:
        result = unittest.main(exit=False, verbosity=2).result
        (RESULTS / "result.json").write_text(json.dumps({"successful": result.wasSuccessful(),
                "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors)}, indent=2))
        print("RESULT_DIRECTORY", RESULTS)
        raise SystemExit(0 if result.wasSuccessful() else 1)
