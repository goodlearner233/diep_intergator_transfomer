"""Independent data and batch invariants for the new training entry only.

Run with the repository's PyTorch/PyG environment. Synthetic references below
are deliberate fixtures, NOT scientific MatPES reference energies.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import sys
import time
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch
from pymatgen.core import Lattice, Structure
from torch.utils.data import random_split

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))
from training_official_aligned.data_pipeline import (MaxAtomsBatchSampler, iter_matpes_records,
    load_element_references, load_prepared, make_splits, prepare_data, distribution_versions)
from matgl.graph.data import collate_fn_pes, split_dataset

ROOT = Path(__file__).parent / f"data_test_results_{time.time_ns()}"
ROOT.mkdir(parents=True)


def records_fixture():
    structure = Structure(Lattice.cubic(5.5), ["Li", "O"], [[0, 0, 0], [0.4, 0.4, 0.4]])
    records = []
    for index in range(24):
        records.append({"structure": structure.as_dict(), "energy": -10.0 - index,
                        "forces": [[index + 1., 2., 3.], [-1., -2., -3.]],
                        "stress": [10., 20., 30., 40., 50., 60.]})
    return records


class DataPipelineTests(unittest.TestCase):
    def test_core_only_pymatgen_metadata(self):
        from importlib.metadata import PackageNotFoundError
        def version(name):
            if name == "pymatgen":
                raise PackageNotFoundError(name)
            return "2026.5.18"
        with patch("training_official_aligned.data_pipeline.importlib.metadata.version", side_effect=version):
            self.assertEqual(distribution_versions(("pymatgen", "pymatgen-core")),
                             {"pymatgen": None, "pymatgen-core": "2026.5.18"})

    @classmethod
    def setUpClass(cls):
        cls.records = records_fixture()
        cls.source = ROOT / "samples.json"
        cls.source.write_text(json.dumps(cls.records), encoding="utf-8")
        cls.refs = ROOT / "synthetic_refs.json"
        cls.refs.write_text(json.dumps({"Li": -1.25, "O": -2.5}), encoding="utf-8")
        cls.output = ROOT / "base"
        cls.metadata = prepare_data(cls.source, cls.output, element_refs_path=cls.refs)
        cls.loaded = load_prepared(cls.output)

    def test_streaming_array_jsonl_and_malformed(self):
        parsed = list(iter_matpes_records(self.source, chunk_size=17))
        self.assertEqual(parsed, json.loads(json.dumps(self.records)))
        jsonl = ROOT / "samples.jsonl"
        jsonl.write_text("\n\n" + "\n".join(map(json.dumps, self.records)) + "\n", encoding="utf-8")
        self.assertEqual(list(iter_matpes_records(jsonl, chunk_size=11)), json.loads(json.dumps(self.records)))
        for num, malformed in enumerate(('[{},]', '[{}] ignored', '[{},', '[42]')):
            path = ROOT / f"malformed-{num}.json"
            path.write_text(malformed)
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                list(iter_matpes_records(path, chunk_size=2))

    def test_split_matches_actual_official_and_legacy_torch(self):
        raw = list(range(41))
        retained = [x for x in raw if x not in {1, 8, 11, 39}]
        got = make_splits(retained, len(raw), 42, "official")
        n = len(retained)
        expected = random_split(range(n), [n - 2 * round(n * .05), round(n * .05), round(n * .05)],
                                generator=torch.Generator().manual_seed(42))
        self.assertEqual(list(got.values()), [subset.indices for subset in expected])
        legacy = make_splits(retained, len(raw), 42, "legacy-membership")
        old = split_dataset(raw, [.9, .05, .05], shuffle=True, random_state=42)
        raw_to_local = {i: k for k, i in enumerate(retained)}
        self.assertEqual(list(legacy.values()), [[raw_to_local[i] for i in part.indices if i in raw_to_local] for part in old])

    def test_rms_training_only_force_vector_and_stress_once(self):
        loaded = self.loaded
        train = loaded["splits"]["train"]
        expected = np.concatenate([self.records[i]["forces"] for i in train])
        correct_rms = np.sqrt(np.square(expected).sum(axis=1).mean())
        self.assertAlmostEqual(loaded["force_rms"], correct_rms, places=12)
        component_rms = np.sqrt(np.square(expected).mean())
        self.assertAlmostEqual(loaded["force_rms"] / component_rms, math.sqrt(3))
        self.assertEqual(loaded["dataset"][0][0].sample_idx.item(), 0)
        stress = loaded["dataset"][0][3]["stresses"].numpy()
        np.testing.assert_allclose(stress, [[-1., -6., -5.], [-6., -2., -4.], [-5., -4., -3.]])
        reloaded = load_prepared(self.output)
        np.testing.assert_array_equal(reloaded["dataset"][0][3]["stresses"].numpy(), stress)
        batch = collate_fn_pes([loaded["dataset"][0], loaded["dataset"][1]], include_stress=True)
        self.assertEqual(batch[0].sample_idx.tolist(), [0, 1])
        self.assertEqual(batch[4].shape, (4, 3))
        self.assertEqual(batch[5].shape, (6, 3))

    def test_cache_reuse_and_same_path_changed_labels_rejected(self):
        before = self.output.joinpath("prepared_data/manifest.json").read_bytes()
        again = prepare_data(self.source, self.output, element_refs_path=self.refs)
        self.assertEqual(again["cache_key"], self.metadata["cache_key"])
        self.assertEqual(before, self.output.joinpath("prepared_data/manifest.json").read_bytes())
        changed = ROOT / "changed.json"
        changed_records = copy.deepcopy(self.records)
        changed_records[0]["energy"] += 123
        changed.write_text(json.dumps(changed_records))
        with self.assertRaisesRegex(ValueError, "identity changed"):
            prepare_data(changed, self.output, element_refs_path=self.refs)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            prepare_data(self.source, self.output, cutoff=4.9, element_refs_path=self.refs)

        # Same path, byte length and timestamp still cannot bypass content hash.
        same_path = ROOT / "same_path.json"
        original_text = self.source.read_text()
        same_path.write_text(original_text)
        same_output = ROOT / "same_path_case"
        prepare_data(same_path, same_output, element_refs_path=self.refs)
        old_stat = same_path.stat()
        self.assertIn('"energy": -10.0', original_text)
        same_path.write_text(original_text.replace('"energy": -10.0', '"energy": -90.0', 1))
        os.utime(same_path, ns=(old_stat.st_atime_ns, old_stat.st_mtime_ns))
        self.assertEqual(same_path.stat().st_size, old_stat.st_size)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            prepare_data(same_path, same_output, element_refs_path=self.refs)

    def test_same_size_cached_label_tamper_rejected(self):
        cache = self.output / "prepared_data" / self.metadata["cache_relative_path"]
        labels_path = cache / "labels.json"
        original = labels_path.read_bytes()
        self.assertIn(b"-10.0", original)
        modified = original.replace(b"-10.0", b"-90.0", 1)
        self.assertEqual(len(modified), len(original))
        try:
            labels_path.write_bytes(modified)
            with self.assertRaisesRegex(ValueError, "cache checksum mismatch: labels.json"):
                prepare_data(self.source, self.output, element_refs_path=self.refs)
        finally:
            labels_path.write_bytes(original)

    def test_reference_validation(self):
        missing = ROOT / "missing_refs.json"
        missing.write_text('{"Li": -1.0}')
        with self.assertRaisesRegex(ValueError, "Missing isolated-atom"):
            prepare_data(self.source, ROOT / "missing_reference_case", element_refs_path=missing)
        official = ROOT / "official_shape_refs.json"
        official.write_text(json.dumps([{"elements": ["Li"], "energy": -1.25}, {"elements": ["O"], "energy": -2.5}]))
        self.assertEqual(load_element_references(official), {"Li": -1.25, "O": -2.5})
        official.write_text(json.dumps([{"elements": ["Li"], "energy": -1}, {"elements": ["Li"], "energy": -2}]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            load_element_references(official)
        # A validation-only element must not silently get a zero reference.
        val_only = copy.deepcopy(self.records)
        validation_index = self.loaded["splits"]["val"][0]
        val_only[validation_index]["structure"] = Structure(
            Lattice.cubic(5.5), ["Na", "O"], [[0, 0, 0], [.4, .4, .4]]).as_dict()
        val_source = ROOT / "validation_only_element.json"
        val_source.write_text(json.dumps(val_only))
        with self.assertRaisesRegex(ValueError, "Na"):
            prepare_data(val_source, ROOT / "validation_missing_ref_case", element_refs_path=self.refs)

    def test_large_structure_removed_ids_preserved_and_jsonl_works(self):
        records = copy.deepcopy(self.records)
        large = Structure(Lattice.cubic(100.), ["Li"] * 151, [[i / 151, 0, 0] for i in range(151)])
        records.insert(3, {"structure": large.as_dict(), "energy": 0., "forces": [[0., 0., 0.]] * 151,
                           "stress": np.eye(3).tolist()})
        records[0]["stress"] = [[10., 60., 50.], [60., 20., 40.], [50., 40., 30.]]
        source = ROOT / "with_large.jsonl"
        source.write_text("\n".join(map(json.dumps, records)))
        output = ROOT / "filtered"
        metadata = prepare_data(source, output, element_refs_path=self.refs)
        loaded = load_prepared(output)
        self.assertEqual(metadata["excluded_count"], 1)
        self.assertEqual(loaded["original_indices"], [i for i in range(25) if i != 3])
        self.assertEqual(loaded["dataset"][3][0].sample_idx.item(), 4)
        np.testing.assert_allclose(loaded["dataset"][0][3]["stresses"].numpy(),
                                   self.loaded["dataset"][0][3]["stresses"].numpy())

    def test_sampler_budget_ddp_padding_eval_exact_and_reproducibility(self):
        counts = [150, 149, 130, 120, 100, 95, 80, 72, 40, 1, 1, 1, 1, 1, 1, 1]
        one = MaxAtomsBatchSampler(counts, max_atoms=300)
        batches = list(one)
        self.assertEqual(sorted(i for batch in batches for i in batch), list(range(len(counts))))
        self.assertTrue(all(sum(counts[i] for i in batch) <= 300 for batch in batches))
        twin = MaxAtomsBatchSampler(counts, max_atoms=300)
        self.assertEqual(list(one), list(twin))
        one.set_epoch(7)
        twin.set_epoch(7)
        self.assertEqual(list(one), list(twin))
        self.assertEqual(sorted(map(tuple, list(one))), sorted(map(tuple, batches)))
        ddp = [MaxAtomsBatchSampler(counts, max_atoms=300, rank=r, num_replicas=4) for r in range(4)]
        self.assertEqual(len({len(s) for s in ddp}), 1)
        evals = [MaxAtomsBatchSampler(counts, max_atoms=300, shuffle=False, rank=r, num_replicas=4, pad=False)
                 for r in range(4)]
        self.assertEqual(sorted(i for sampler in evals for batch in sampler for i in batch), list(range(len(counts))))
        tiny = [MaxAtomsBatchSampler([1], num_replicas=4, rank=r) for r in range(4)]
        self.assertTrue(all(list(s) == [[0]] for s in tiny))
        with self.assertRaises(ValueError):
            MaxAtomsBatchSampler([1001])


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2).result
    (ROOT / "result.json").write_text(json.dumps({"tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "successful": result.wasSuccessful()}, indent=2))
    print("RESULT_DIRECTORY", ROOT)
    raise SystemExit(0 if result.wasSuccessful() else 1)
