"""Leakage-safe reader for hierarchical frozen visual targets."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class HierarchicalTargetBank(Dataset[dict[str, object]]):
    """Memory-mapped target view restricted to exactly one declared split.

    A caller cannot accidentally iterate over validation/test rows while asking
    for training targets.  Arrays remain memory mapped and are copied only for
    the requested sample.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        teachers: Sequence[str] = ("synclr", "clip_b32", "vae_sdxl"),
        *,
        verify_checksums: bool = False,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise RuntimeError("unsupported hierarchical target schema")

        index_path = self.root / self.manifest["index"]
        if verify_checksums and _sha256(index_path) != self.manifest["index_sha256"]:
            raise RuntimeError("hierarchical target index checksum mismatch")
        with index_path.open(newline="", encoding="utf-8") as handle:
            all_rows = list(csv.DictReader(handle))
        self.rows = [row for row in all_rows if row["split"] == split]
        if len(self.rows) != int(self.manifest["split_counts"][split]):
            raise RuntimeError(f"manifest/index row-count mismatch for split {split}")

        available = self.manifest["teachers"]
        missing = sorted(set(teachers).difference(available))
        if missing:
            raise KeyError(f"target bank does not contain teachers: {missing}")
        self.arrays: dict[str, np.ndarray] = {}
        for teacher in teachers:
            metadata = available[teacher]
            path = self.root / metadata["array"]
            if verify_checksums and _sha256(path) != metadata["array_sha256"]:
                raise RuntimeError(f"{teacher} target checksum mismatch")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if tuple(array.shape) != tuple(metadata["shape"]):
                raise RuntimeError(f"{teacher} target shape does not match manifest")
            self.arrays[teacher] = array

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.rows[item]
        position = int(row["position"])
        targets = {
            key: torch.from_numpy(np.array(array[position], dtype=np.float32, copy=True))
            for key, array in self.arrays.items()
        }
        return {
            "filename": row["filename"],
            "image_index": int(row["image_index"]),
            "split": row["split"],
            "targets": targets,
        }
