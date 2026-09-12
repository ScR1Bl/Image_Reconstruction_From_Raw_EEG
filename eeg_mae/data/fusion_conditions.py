"""Disk-backed, split-safe EEG condition bank for the fusion prior."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CONDITION_SHAPES = {
    "semantic_fused": (1024,),
    "semantic_synclr": (1024,),
    "semantic_clip_b32": (1024,),
    "semantic_vae_sdxl": (1024,),
    "color_grid16": (16, 16, 192),
    "edge_grid16": (16, 16, 192),
}


class FusionConditionBank(Dataset[dict[str, object]]):
    def __init__(self, root: str | Path, split: str) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1 or not self.manifest.get("complete"):
            raise RuntimeError("fusion condition bank is incomplete or incompatible")
        with (self.root / self.manifest["index"]).open(newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if len(self.rows) != int(self.manifest["split_counts"][split]):
            raise RuntimeError("fusion condition split count mismatch")
        self.arrays = {
            name: np.load(self.root / metadata["array"], mmap_mode="r", allow_pickle=False)
            for name, metadata in self.manifest["conditions"].items()
        }

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: int) -> dict[str, object]:
        row = self.rows[item]
        position = int(row["position"])
        return {
            "filename": row["filename"],
            "image_index": int(row["image_index"]),
            "conditions": {
                name: torch.from_numpy(np.array(array[position], dtype=np.float32, copy=True))
                for name, array in self.arrays.items()
            },
        }


__all__ = ["CONDITION_SHAPES", "FusionConditionBank"]
