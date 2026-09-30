"""Independent formulas for streaming module summaries and optional observer."""
from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import torch

from test_telemetry import Telemetry, rows, telemetry_module


class GroupedToy(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        self.model.model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)])
        self.model.model.unused = torch.nn.Parameter(torch.tensor([4.0, 5.0]))
        self.loss_params = {"delta": 1.0}


class ModuleTelemetryTests(unittest.TestCase):
    def fixture(self, directory, device="cpu", module_statistics=True):
        torch.manual_seed(42)
        model = GroupedToy().to(device)
        parameters = list(model.named_parameters())
        optimizer = torch.optim.AdamW([
            {"params": [p for i, (_, p) in enumerate(parameters) if i % 2 == 0],
             "betas": (0.8, 0.99), "amsgrad": True},
            {"params": [p for i, (_, p) in enumerate(parameters) if i % 2],
             "betas": (0.9, 0.999), "amsgrad": False}], lr=0.001, weight_decay=0.00001)
        trainer = SimpleNamespace(global_rank=0, world_size=1, precision="32-true", optimizers=[optimizer],
                                  current_epoch=0, global_step=0)
        observer = Telemetry(directory, module_statistics=module_statistics)
        observer.setup(trainer, model, "fit")
        observer.on_fit_start(trainer, model)
        return model, optimizer, observer, trainer

    def test_group_name_compatibility(self):
        self.assertEqual(telemetry_module._parameter_group("model.model.blocks.2.layers.0.weight"), "model.model.blocks.2")
        self.assertEqual(telemetry_module._parameter_group("model.model.embedding.weight"), "model.model.embedding")
        self.assertEqual(telemetry_module._parameter_group("linear.weight"), "linear")

    def test_mixed_steps_betas_amsgrad_and_missing_states(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer, observer, _ = self.fixture(directory)
            names = {id(p): name for name, p in model.named_parameters()}
            expected = {}
            for gi, group in enumerate(optimizer.param_groups):
                for pi, parameter in enumerate(group["params"]):
                    name = names[id(parameter)]
                    module = telemetry_module._parameter_group(name)
                    expected.setdefault(module, [])
                    if "unused" in name:
                        continue
                    step = 3 + pi + gi
                    values = torch.arange(1, parameter.numel() + 1, device=parameter.device).reshape_as(parameter).float()
                    state = optimizer.state[parameter]
                    state.update(step=torch.tensor(float(step)), exp_avg=values * (-0.2 if gi else 0.3), exp_avg_sq=values * 0.04)
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = values * 0.08
                    parameter.grad = None if pi % 2 else torch.full_like(parameter, 0.2)
                    beta1, beta2 = group["betas"]
                    for j in range(parameter.numel()):
                        m_hat = float(state["exp_avg"].flatten()[j]) / (1 - beta1 ** step)
                        v_hat = float(state["exp_avg_sq"].flatten()[j]) / (1 - beta2 ** step)
                        source = state["max_exp_avg_sq"] if group["amsgrad"] else state["exp_avg_sq"]
                        denominator = math.sqrt(float(source.flatten()[j]) / (1 - beta2 ** step)) + group["eps"]
                        expected[module].append((m_hat, v_hat, math.sqrt(v_hat), denominator, 1 / denominator, step))
            state_before = copy.deepcopy(optimizer.state_dict())
            rng_before = torch.get_rng_state().clone()
            actual = observer._module_snapshot(optimizer)
            self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
            for module, samples in expected.items():
                adam = actual[module]["adam"]
                self.assertEqual(adam["initialized_elements"], len(samples))
                if not samples:
                    self.assertEqual(adam["uninitialized_elements"], 2)
                    self.assertIsNone(adam["m_hat_l2"])
                    self.assertIsNone(adam["denominator_min"])
                    continue
                for key, column in (("m_hat_l2", 0), ("v_hat_l2", 1), ("sqrt_v_hat_l2", 2), ("inverse_denominator_l2", 4)):
                    target = math.sqrt(sum(sample[column] ** 2 for sample in samples))
                    self.assertAlmostEqual(adam[key], target, places=10, msg=f"{module}/{key}")
                self.assertAlmostEqual(adam["denominator_min"], min(sample[3] for sample in samples), places=12)
                self.assertAlmostEqual(adam["denominator_max"], max(sample[3] for sample in samples), places=12)
                self.assertAlmostEqual(adam["inverse_denominator_max"], max(sample[4] for sample in samples), places=12)
                self.assertEqual(adam["state_step_min"], min(sample[5] for sample in samples))
                self.assertEqual(adam["state_step_max"], max(sample[5] for sample in samples))
            for key, states in state_before["state"].items():
                for field, value in states.items():
                    self.assertTrue(torch.equal(value, optimizer.state_dict()["state"][key][field]))
            observer.close()

    def test_module_delta_matches_direct_measurement_and_inactive_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer, observer, trainer = self.fixture(directory)
            before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            for name, parameter in model.named_parameters():
                parameter.grad = None if "unused" in name else torch.full_like(parameter, 10.0)
            observer.on_before_optimizer_step(trainer, model, optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            observer.close()
            record = rows(observer.session_dir / "optimizer_updates.jsonl.gz")[0]
            self.assertTrue(record["finite"])
            for module, stat in record["module_statistics"].items():
                params = [(name, p) for name, p in model.named_parameters() if telemetry_module._parameter_group(name) == module]
                target = math.sqrt(sum(float((p.detach() - before[name]).double().square().sum()) for name, p in params))
                self.assertAlmostEqual(stat["parameter_update_l2"], target, places=12)
                self.assertEqual(stat["before"]["adam"]["initialized_elements"], 0)
                self.assertIsNone(stat["before"]["adam"]["denominator_min"])
                if stat["after"]["active_gradient_elements"] == 0:
                    self.assertEqual(stat["after"]["adam"]["uninitialized_elements"], 2)
                    self.assertEqual(stat["after"]["active_gradient_elements"], 0)
                    self.assertEqual(stat["parameter_update_l2"], 0)
                else:
                    self.assertGreater(stat["gradient_l2_before_clip"], stat["gradient_l2_after_clip"])
                    self.assertEqual(stat["after"]["adam"]["state_step_min"], 1)
            metadata = rows(observer.session_dir / "events.jsonl.gz")
            self.assertEqual(metadata[0]["metadata"]["schema_version"], 2)
            self.assertTrue(metadata[1]["parameter_name_groups"])

    def test_nonfinite_and_disabled_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            model, optimizer, observer, _ = self.fixture(directory)
            name, parameter = next(iter(model.named_parameters()))
            with torch.no_grad():
                parameter.flatten()[0] = float("nan")
            parameter.grad = torch.ones_like(parameter)
            parameter.grad.flatten()[0] = float("inf")
            state = optimizer.state[parameter]
            state.update(step=torch.tensor(1.0), exp_avg=torch.zeros_like(parameter),
                         exp_avg_sq=torch.zeros_like(parameter), max_exp_avg_sq=torch.zeros_like(parameter))
            state["exp_avg"].flatten()[0] = float("nan")
            stat = observer._module_snapshot(optimizer)[telemetry_module._parameter_group(name)]
            self.assertEqual(stat["parameter_nonfinite_elements"], 1)
            self.assertEqual(stat["gradient_nonfinite_elements"], 1)
            self.assertEqual(stat["adam"]["nonfinite_elements"], 1)
            self.assertFalse(telemetry_module._finite(stat))
            observer.module_statistics = False
            self.assertEqual(observer._module_snapshot(optimizer), {})
            observer.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
