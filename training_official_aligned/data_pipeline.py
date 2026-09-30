"""Portable MatPES preparation and whole-structure atom-budget batching.

Only ``prepare_data`` writes files, and must be called on global rank zero.
``load_prepared`` is read-only on every rank. Original repository code and raw
input files are never modified. Stress input is raw MatPES kbar, not GPa.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from ase.stress import voigt_6_to_full_3x3_stress
from pymatgen.core import Structure
from torch.utils.data import Dataset, Sampler

import matgl
from matgl.config import DEFAULT_ELEMENTS
from matgl.ext.pymatgen import Structure2Graph
from matgl.graph.data import MGLDataset

FORMAT_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def distribution_versions(names) -> dict:
    """Record distribution metadata without requiring an optional umbrella package.

    New pymatgen-core installations provide pymatgen.core without installing
    the separate pymatgen distribution. Imports already verify runtime modules;
    missing distribution metadata is explicitly recorded as null.
    """
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _json_write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def iter_matpes_records(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Stream JSON array or JSONL without keeping a second full raw JSON copy."""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        prefix = ""
        while not prefix.strip():
            prefix = handle.read(chunk_size)
            if not prefix:
                raise ValueError("MatPES input is empty")
        handle.seek(0)
        if prefix.lstrip()[0] != "[":
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise ValueError(f"JSONL line {line_number} is not a record object")
                    yield item
            return

        decoder = json.JSONDecoder()
        buffer = ""
        eof = False

        def read_more():
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            buffer += chunk
            eof = not chunk

        def require_char():
            nonlocal buffer
            buffer = buffer.lstrip()
            while not buffer and not eof:
                read_more()
                buffer = buffer.lstrip()
            if not buffer:
                raise ValueError("Unexpected end of MatPES JSON array")

        require_char()
        buffer = buffer[1:]  # Opening '[' established above.
        require_char()
        if buffer[0] == "]":
            buffer = buffer[1:]
        else:
            while True:
                while True:
                    try:
                        item, end = decoder.raw_decode(buffer)
                        break
                    except json.JSONDecodeError:
                        if eof:
                            raise
                        read_more()
                if not isinstance(item, dict):
                    raise ValueError("Every MatPES array entry must be a record object")
                yield item
                buffer = buffer[end:]
                require_char()
                delimiter, buffer = buffer[0], buffer[1:]
                if delimiter == "]":
                    break
                if delimiter != ",":
                    raise ValueError("Expected ',' or ']' after MatPES record")
                require_char()
                if buffer[0] == "]":
                    raise ValueError("Trailing comma in MatPES JSON array")
        if buffer.strip() or handle.read().strip():
            raise ValueError("Unexpected content after MatPES JSON array")


def load_element_references(path: Path) -> dict[str, float]:
    """Read the official MatPES atoms list, or an explicit symbol->energy map."""
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        pairs = list(payload.items())
    elif isinstance(payload, list):
        pairs = []
        for row in payload:
            elements = row.get("elements")
            symbol = elements[0] if elements and len(elements) == 1 else row.get("chemsys")
            if not isinstance(symbol, str):
                raise ValueError("Each isolated-atom reference must identify one element")
            pairs.append((symbol, row["energy"]))
    else:
        raise ValueError("Element references must be a JSON object or list")
    result = {}
    for symbol, value in pairs:
        if symbol in result:
            raise ValueError(f"Duplicate reference energy for {symbol}")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Nonfinite reference energy for {symbol}")
        result[symbol] = number
    if not result:
        raise ValueError("Element reference file is empty")
    return result


def make_splits(original_indices: list[int], raw_count: int, seed: int, policy: str) -> dict[str, list[int]]:
    """Return filtered-local indices; preserve raw IDs separately in the manifest."""
    generator = torch.Generator().manual_seed(seed)
    if policy == "official":
        n = len(original_indices)
        n_val = int(round(n * 0.05))
        n_test = int(round(n * 0.05))
        n_train = n - n_val - n_test
        permutation = torch.randperm(n, generator=generator).tolist()
        return {"train": permutation[:n_train], "val": permutation[n_train:n_train + n_val],
                "test": permutation[n_train + n_val:]}
    if policy == "legacy-membership":
        permutation = torch.randperm(raw_count, generator=generator).tolist()
        n_train, n_val = int(raw_count * 0.90), int(raw_count * 0.05)
        lookup = {raw: local for local, raw in enumerate(original_indices)}
        original_splits = {"train": permutation[:n_train], "val": permutation[n_train:n_train + n_val],
                           "test": permutation[n_train + n_val:]}
        return {name: [lookup[i] for i in indices if i in lookup] for name, indices in original_splits.items()}
    raise ValueError(f"Unknown split policy: {policy}")


def _implementation_fingerprint() -> dict:
    from matgl.graph import converters
    files = [Path(__file__), Path(inspect.getfile(Structure2Graph)),
             Path(inspect.getfile(MGLDataset)), Path(inspect.getfile(converters))]
    return {"source_hashes": {f.name: sha256_file(f) for f in files},
            "versions": distribution_versions(("torch", "torch-geometric", "pymatgen",
                                                "pymatgen-core", "numpy", "ase"))}


def prepare_data(data_path: Path, output_dir: Path, max_atoms: int = 150, cutoff: float = 5.0,
                 seed: int = 42, split_policy: str = "official", *,
                 element_refs_path: Path, threebody_cutoff: float = 4.0) -> dict:
    """Rank-zero preparation. Reusing an output with changed inputs is rejected.

    The complete input and reference files are hashed on preparation/resume;
    the graph cache key includes conversion code, elements, cutoffs and filter.
    Source arrays are streamed but the existing MGLDataset graph builder still
    requires retained structures/labels in host memory once on rank zero.
    """
    data_path, output_dir, element_refs_path = map(Path, (data_path, output_dir, element_refs_path))
    if max_atoms <= 0 or cutoff <= 0 or not 0 < threebody_cutoff <= cutoff:
        raise ValueError("Require max_atoms > 0 and 0 < threebody_cutoff <= cutoff")
    if split_policy not in {"official", "legacy-membership"}:
        raise ValueError("split_policy must be official or legacy-membership")
    before_stat = data_path.stat()
    identity = {"format_version": FORMAT_VERSION, "source_sha256": sha256_file(data_path),
                "reference_sha256": sha256_file(element_refs_path), "max_atoms": max_atoms,
                "cutoff": cutoff, "threebody_cutoff": threebody_cutoff, "seed": seed,
                "split_policy": split_policy, "element_types": list(DEFAULT_ELEMENTS),
                "stress_conversion": "raw MatPES kbar * -0.1 -> GPa; 6-Voigt or 3x3 -> 3x3",
                "implementation": _implementation_fingerprint()}
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    prepared_dir = output_dir / "prepared_data"
    manifest_path = prepared_dir / "manifest.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata["cache_key"] != cache_key:
            raise ValueError("Prepared data identity changed (data, refs, split, filter or code). Use a new output directory.")
        _verify_cache(prepared_dir, metadata, verify_hashes=True)
        metadata["manifest_sha256"] = sha256_file(manifest_path)
        return metadata
    prepared_dir.mkdir(parents=True, exist_ok=True)
    references = load_element_references(element_refs_path)
    structures, original_indices, atom_counts, excluded = [], [], [], []
    labels = {"energies": [], "forces": [], "stresses": []}
    raw_count = 0
    for raw_index, raw in enumerate(iter_matpes_records(data_path)):
        raw_count += 1
        structure = Structure.from_dict(raw["structure"])
        count = len(structure)
        if count > max_atoms:
            excluded.append(raw_index)
            continue
        if count <= 0:
            raise ValueError(f"Empty structure at raw sample {raw_index}")
        unknown = set(structure.composition.get_el_amt_dict()) - set(DEFAULT_ELEMENTS)
        if unknown:
            raise ValueError(f"Unsupported model element(s) {sorted(unknown)} at raw sample {raw_index}")
        energy = float(raw["energy"])
        forces = np.asarray(raw["forces"], dtype=np.float64)
        stress = np.asarray(raw["stress"], dtype=np.float64)
        if forces.shape != (count, 3):
            raise ValueError(f"Invalid force shape {forces.shape} at raw sample {raw_index}")
        if stress.shape == (6,):
            stress = voigt_6_to_full_3x3_stress(stress)
        if stress.shape != (3, 3):
            raise ValueError(f"Invalid stress shape {stress.shape} at raw sample {raw_index}")
        if not math.isfinite(energy) or not np.isfinite(forces).all() or not np.isfinite(stress).all():
            raise ValueError(f"Nonfinite E/F/S label at raw sample {raw_index}")
        structures.append(structure)
        original_indices.append(raw_index)
        atom_counts.append(count)
        labels["energies"].append(energy)
        labels["forces"].append(forces.tolist())
        labels["stresses"].append((stress * -0.1).tolist())
    after_stat = data_path.stat()
    if (before_stat.st_size, before_stat.st_mtime_ns) != (after_stat.st_size, after_stat.st_mtime_ns):
        raise RuntimeError("Input data changed during preparation")
    splits = make_splits(original_indices, raw_count, seed, split_policy)
    if any(not indices for indices in splits.values()):
        raise ValueError("Filtering/splitting left an empty train, validation or test split")
    train_elements = set()
    used_elements = set()
    for structure in structures:
        used_elements.update(structure.composition.get_el_amt_dict())
    force_square_sum, train_atom_count = 0.0, 0
    for local in splits["train"]:
        train_elements.update(structures[local].composition.get_el_amt_dict())
        force_square_sum += float(np.square(np.asarray(labels["forces"][local], dtype=np.float64)).sum())
        train_atom_count += atom_counts[local]
    missing_used_refs = sorted(used_elements - references.keys())
    if missing_used_refs:
        raise ValueError(f"Missing isolated-atom reference energies for retained train/validation/test elements: {missing_used_refs}")
    force_rms = math.sqrt(force_square_sum / train_atom_count)
    if not math.isfinite(force_rms) or force_rms <= 0:
        raise ValueError("Training force-vector RMS must be finite and greater than zero")

    cache_name = f"cache-{cache_key[:20]}"
    staging = prepared_dir / f".building-{uuid.uuid4().hex}"
    converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=cutoff)
    dataset = MGLDataset(structures=structures, labels=labels, converter=converter,
                         root=str(staging), clear_processed=True, save_cache=True)
    for graph, raw_index in zip(dataset.graphs, original_indices, strict=True):
        graph.sample_idx = torch.tensor([raw_index], dtype=torch.long)
    dataset.save()
    split_payload = {"local": splits, "original": {name: [original_indices[i] for i in indices]
                                                   for name, indices in splits.items()},
                     "original_indices": original_indices, "atom_counts": atom_counts,
                     "excluded_original_indices": excluded}
    _json_write(staging / "split_indices.json", split_payload)
    metadata = {"identity": identity, "cache_key": cache_key, "cache_relative_path": cache_name,
                "source_path_at_preparation": str(data_path.resolve()),
                "reference_path_at_preparation": str(element_refs_path.resolve()),
                "source_sha256": identity["source_sha256"], "raw_count": raw_count,
                "retained_count": len(original_indices), "excluded_count": len(excluded),
                "split_counts": {name: len(indices) for name, indices in splits.items()},
                "element_types": list(DEFAULT_ELEMENTS), "force_rms": force_rms,
                "force_rms_definition": "sqrt(sum_train_atoms(Fx^2+Fy^2+Fz^2) / train_atom_count)",
                "train_atom_count": train_atom_count,
                "element_refs": [references.get(symbol, 0.0) for symbol in DEFAULT_ELEMENTS],
                "training_elements": sorted(train_elements),
                "used_elements": sorted(used_elements),
                "zero_reference_elements_unused_in_all_splits": [el for el in DEFAULT_ELEMENTS if el not in references],
                "cache_files": {p.name: p.stat().st_size for p in staging.iterdir() if p.is_file()},
                "cache_sha256": {p.name: sha256_file(p) for p in staging.iterdir() if p.is_file()},
                "split_indices_sha256": sha256_file(staging / "split_indices.json")}
    final_cache = prepared_dir / cache_name
    if final_cache.exists():
        raise FileExistsError(f"Uncommitted cache already exists: {final_cache}; choose a fresh output directory")
    staging.rename(final_cache)
    temporary_manifest = prepared_dir / f"manifest-{uuid.uuid4().hex}.tmp"
    _json_write(temporary_manifest, metadata)
    os.replace(temporary_manifest, manifest_path)
    metadata["manifest_sha256"] = sha256_file(manifest_path)
    return metadata


def _verify_cache(prepared_dir: Path, metadata: dict, *, verify_hashes: bool = False) -> Path:
    cache = prepared_dir / metadata["cache_relative_path"]
    if cache.resolve().parent != prepared_dir.resolve():
        raise ValueError("Invalid prepared cache path")
    if verify_hashes and set(metadata.get("cache_sha256", {})) != set(metadata["cache_files"]):
        raise ValueError("Prepared cache checksums are incomplete; use a new output directory")
    for filename, expected_size in metadata["cache_files"].items():
        path = cache / filename
        if path.parent.resolve() != cache.resolve() or not path.is_file() or path.stat().st_size != expected_size:
            raise ValueError(f"Missing or changed prepared cache file: {filename}")
        if verify_hashes and sha256_file(path) != metadata["cache_sha256"][filename]:
            raise ValueError(f"Prepared cache checksum mismatch: {filename}")
    if sha256_file(cache / "split_indices.json") != metadata["split_indices_sha256"]:
        raise ValueError("Prepared split/index manifest checksum mismatch")
    return cache


class PreparedDataset(Dataset):
    """Read-only equivalent of MGLDataset's item access, safe for DDP startup."""
    def __init__(self, cache: Path, metadata: dict):
        # Private memory maps share unchanged cache pages between DDP ranks.
        # PyG collation creates new batch tensors; cached tensors stay read-only.
        self.graphs = torch.load(str(cache / "pyg_graph.pt"), map_location="cpu", weights_only=False, mmap=True)
        self.lattices = torch.load(str(cache / "lattice.pt"), map_location="cpu", weights_only=False, mmap=True)
        self.state_attr = torch.load(str(cache / "state_attr.pt"), map_location="cpu", weights_only=False, mmap=True)
        self.labels = json.loads((cache / "labels.json").read_text(encoding="utf-8"))
        self.element_types = tuple(metadata["element_types"])
        n = metadata["retained_count"]
        if not (len(self.graphs) == len(self.lattices) == len(self.state_attr) == n):
            raise ValueError("Prepared graph cache count mismatch")
        if any(len(values) != n for values in self.labels.values()):
            raise ValueError("Prepared label count mismatch")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return (self.graphs[index], self.lattices[index], self.state_attr[index],
                {key: torch.tensor(values[index], dtype=matgl.float_th) for key, values in self.labels.items()})


def load_prepared(output_dir: Path) -> dict:
    """Load only committed prepared artifacts; never write or reconvert stress."""
    prepared_dir = Path(output_dir) / "prepared_data"
    manifest_path = prepared_dir / "manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = _verify_cache(prepared_dir, metadata)
    indices = json.loads((cache / "split_indices.json").read_text(encoding="utf-8"))
    dataset = PreparedDataset(cache, metadata)
    for graph, raw_index in zip(dataset.graphs, indices["original_indices"], strict=True):
        if graph.sample_idx.numel() != 1 or graph.sample_idx.item() != raw_index:
            raise ValueError("Graph sample ID does not match immutable original index")
    metadata["manifest_sha256"] = sha256_file(manifest_path)
    return {"dataset": dataset, "metadata": metadata, "splits": indices["local"],
            "original_splits": indices["original"], "atom_counts": indices["atom_counts"],
            "original_indices": indices["original_indices"], "force_rms": metadata["force_rms"],
            "element_refs": metadata["element_refs"]}


class MaxAtomsBatchSampler(Sampler[list[int]]):
    """Fixed greedy whole-structure packs; shuffle pack order each epoch.

    DDP training pads by repeating whole packs so all ranks have equal steps.
    Set pad=False for evaluation: rank shards then cover each item exactly once
    (and can have different lengths; the caller must avoid per-batch collectives).
    Indices refer to the supplied subset, not the source file.
    """
    def __init__(self, atom_counts, max_atoms: int = 1000, shuffle: bool = True,
                 rank: int = 0, num_replicas: int = 1, seed: int = 42, pad: bool = True):
        self.atom_counts = [int(n) for n in atom_counts]
        if max_atoms <= 0 or num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError("Invalid atom budget or rank/world size")
        if any(n <= 0 or n > max_atoms for n in self.atom_counts):
            raise ValueError("Each structure must have 1..max_atoms atoms; structures cannot be split")
        self.max_atoms, self.shuffle, self.rank = max_atoms, shuffle, rank
        self.num_replicas, self.seed, self.pad, self.epoch = num_replicas, seed, pad, 0
        indices = (torch.randperm(len(self.atom_counts), generator=torch.Generator().manual_seed(seed)).tolist()
                   if shuffle else list(range(len(self.atom_counts))))
        # Lightning calls sampler.set_epoch for custom batch samplers.
        self.sampler = self
        self.batches = []
        current, total = [], 0
        for index in indices:
            count = self.atom_counts[index]
            if current and total + count > max_atoms:
                self.batches.append(current)
                current, total = [], 0
            current.append(index)
            total += count
        if current:
            self.batches.append(current)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _ordered_ids(self):
        indices = list(range(len(self.batches)))
        if self.shuffle:
            indices = torch.randperm(len(self.batches), generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        if self.pad and indices:
            extra = (-len(indices)) % self.num_replicas
            indices += (indices * math.ceil(extra / len(indices)))[:extra]
        return indices

    def __iter__(self):
        return iter([self.batches[i][:] for i in self._ordered_ids()[self.rank::self.num_replicas]])

    def __len__(self):
        n = len(self.batches)
        if self.pad:
            return math.ceil(n / self.num_replicas)
        return len(range(self.rank, n, self.num_replicas))

    def diagnostics(self) -> dict:
        order = self._ordered_ids()
        repeated = order[len(self.batches):]
        return {"epoch": self.epoch, "rank": self.rank, "num_replicas": self.num_replicas,
                "global_unique_batches": len(self.batches), "local_batches": len(self),
                "global_padding_batches": len(repeated),
                "global_padding_structures": sum(len(self.batches[i]) for i in repeated),
                "repeated_batch_ids": repeated,
                "local_batch_atoms": [sum(self.atom_counts[j] for j in self.batches[i])
                                      for i in order[self.rank::self.num_replicas]]}
