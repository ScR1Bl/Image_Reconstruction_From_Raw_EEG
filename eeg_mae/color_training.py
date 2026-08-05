"""Shared data and checkpoint utilities for the final color encoder."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from .advanced_color import DINOColorOracleV2
from .semantic_edge import EEGDINOGridPredictor


def load_color_oracle(path: Path, device: torch.device) -> tuple[DINOColorOracleV2, dict]:
    """Load and freeze the color oracle used as a training teacher."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DINOColorOracleV2(**payload["model_config"])
    model.load_state_dict(payload.get("ema", payload["model"]), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def semantic_ensemble(predictor: EEGDINOGridPredictor, eeg: torch.Tensor) -> torch.Tensor:
    """Average normalized semantic evidence across the canonical subjects."""

    batch, subjects = eeg.shape[:2]
    subject_ids = torch.arange(subjects, device=eeg.device).repeat(batch)
    with torch.no_grad():
        values = predictor.encoder(eeg.flatten(0, 1), subject_ids)
    return F.normalize(values.reshape(batch, subjects, -1).mean(1), dim=-1)


def split_records(index_path: Path, split: str, filenames: list[str]) -> list[tuple[int, int, str]]:
    """Match an image-disjoint EEG split to rows in a derived color bank."""

    lookup = {name: index for index, name in enumerate(filenames)}
    records: dict[int, tuple[int, int, str]] = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) != 0 or row["split"] != split:
                continue
            name, condition = row["image_file"], int(row["image_index"])
            if name in lookup:
                records[condition] = (condition, lookup[name], name)
    return [records[key] for key in sorted(records)]


class EEGColorDataset(Dataset):
    """Aligned EEG, full-color latent, and RGB targets for one image split."""

    def __init__(self, arrays, records, latent_root: Path, rgb_root: Path) -> None:
        self.arrays, self.records = arrays, records
        self.grid = np.load(latent_root / "grid.npy", mmap_mode="r")
        self.global_feature = np.load(latent_root / "global.npy", mmap_mode="r")
        manifest = json.loads((rgb_root / "manifest.json").read_text(encoding="utf-8"))
        rgb_lookup = {name: index for index, name in enumerate(manifest["filenames"])}
        self.rgb_rows = [rgb_lookup[record[2]] for record in records]
        self.rgb = np.load(rgb_root / "rgb.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        condition, latent_index, name = self.records[index]
        eeg = np.stack(
            [np.asarray(array[condition], dtype=np.float32).mean(0) for array in self.arrays]
        )
        return {
            "eeg": torch.from_numpy(eeg),
            "grid": torch.from_numpy(np.array(self.grid[latent_index], copy=True)),
            "global": torch.from_numpy(np.array(self.global_feature[latent_index], copy=True)),
            "rgb": torch.from_numpy(np.array(self.rgb[self.rgb_rows[index]], copy=True)).float()
            / 255,
            "filename": name,
        }


def training_rgb_mean(dataset: EEGColorDataset) -> torch.Tensor:
    """Compute the training-image mean without loading the RGB bank at once."""

    total = np.zeros(dataset.rgb.shape[1:], np.float64)
    for start in range(0, len(dataset.rgb_rows), 256):
        rows = dataset.rgb_rows[start : start + 256]
        total += np.asarray(dataset.rgb[rows], dtype=np.float32).sum(0, dtype=np.float64)
    return torch.from_numpy(total / (255.0 * len(dataset.rgb_rows))).float()[None]


__all__ = [
    "EEGColorDataset",
    "load_color_oracle",
    "semantic_ensemble",
    "split_records",
    "training_rgb_mean",
]
