"""Frozen image splits and aligned, single-participant experimental data.

The 6/2/2 split is image-disjoint, NOT certified sequence-disjoint: the supplied
preprocessed arrays do not contain original presentation events. Keep this
qualification in every report until an event-level audit is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PROTOCOL_VERSION = 1


def make_protocol(rows: list[dict], seed: int = 20260912) -> dict:
    groups = defaultdict(list)
    names = set()
    indices = set()
    for row in rows:
        name, index = row["filename"], int(row["image_index"])
        if name in names or index in indices:
            raise ValueError("duplicate image name or EEG condition")
        names.add(name)
        indices.add(index)
        groups[name.rsplit("_", 1)[0]].append(dict(row))
    result = []
    for concept in sorted(groups):
        group = sorted(groups[concept], key=lambda x: x["filename"])
        if len(group) != 10:
            raise ValueError(f"expected 10 images for {concept}, got {len(group)}")
        local_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{concept}".encode()).digest()[:8], "big"
        )
        order = np.random.default_rng(local_seed).permutation(10)
        for position, number in enumerate(order):
            row = group[int(number)]
            result.append(
                {
                    "filename": row["filename"],
                    "image_index": int(row["image_index"]),
                    "visual_position": int(row["position"]),
                    "concept": concept,
                    "split": "train" if position < 6 else "val" if position < 8 else "test",
                    "legacy_split": row["split"],
                }
            )
    result.sort(key=lambda x: x["image_index"])
    return {
        "version": PROTOCOL_VERSION,
        "seed": seed,
        "split_per_concept": {"train": 6, "val": 2, "test": 2},
        "image_disjoint": True,
        "sequence_disjoint_verified": False,
        "temporal_audit": "pending original session/sequence event metadata",
        "test_policy": "explicit final evaluation only; never checkpoint selection",
        "rows": result,
    }


def save_protocol(index: Path, output: Path, seed: int = 20260912) -> dict:
    with index.open(newline="", encoding="utf-8") as handle:
        protocol = make_protocol(list(csv.DictReader(handle)), seed)
    protocol["source_index_sha256"] = hashlib.sha256(index.read_bytes()).hexdigest()
    encoded = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"refusing to replace frozen protocol: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return protocol


def load_protocol(path: str | Path) -> dict:
    path = Path(path)
    result = json.loads(path.read_text(encoding="utf-8"))
    result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


class SingleSubjectBank:
    """Aligned targets; no EEG from a second participant is loaded."""

    def __init__(
        self,
        protocol: str | Path,
        subject: int = 0,
        root: str | Path = "data/derived",
        *,
        load_edges: bool = True,
        load_rgb: bool = False,
        edge_keys: tuple[str, ...] | None = None,
    ):
        if subject not in range(10):
            raise ValueError("subject must be a zero-based THINGS-EEG2 participant ID")
        root = Path(root)
        self.protocol = load_protocol(protocol)
        self.rows = self.protocol["rows"]
        self.subject = subject
        self.eeg = np.load(
            root / "eeg_float32_cache" / f"sub-{subject + 1:02d}__preprocessed_eeg_training.npy",
            mmap_mode="r",
        )
        if self.eeg.shape != (16540, 4, 17, 100):
            raise ValueError(f"unexpected EEG shape: {self.eeg.shape}")
        self.indices = np.asarray([r["image_index"] for r in self.rows])
        self.positions = np.asarray([r["visual_position"] for r in self.rows])
        self.names = [r["filename"] for r in self.rows]
        hierarchy = root / "hierarchical_visual_targets"
        with (hierarchy / "index.csv").open(newline="", encoding="utf-8") as handle:
            image_rows = list(csv.DictReader(handle))
        if [image_rows[p]["filename"] for p in self.positions] != self.names:
            raise ValueError("hierarchical visual bank and protocol disagree")
        self.targets = {
            name: np.asarray(
                np.load(hierarchy / f"{name}.float16.npy", mmap_mode="r")[self.positions],
                dtype=np.float32,
            )
            for name in ("clip_b32", "synclr", "vae_sdxl")
        }
        dino = torch.load(
            root / "visual_targets_dinov2s_192.pt", map_location="cpu", weights_only=False
        )
        lookup = {name: i for i, name in enumerate(dino["filenames"])}
        self.targets["dino"] = dino["dino_global"][[lookup[n] for n in self.names]].float().numpy()
        del dino
        self.valid = np.ones(len(self.rows), dtype=bool)
        if load_edges:
            edge = torch.load(
                root / "pyramid_edge_targets_v1.pt", map_location="cpu", weights_only=False
            )
            lookup = {name: i for i, name in enumerate(edge["filenames"])}
            ordered = [lookup[n] for n in self.names]
            self.valid = edge["valid"][ordered].numpy().astype(bool)
            from .build_pyramid_edge_targets import PYRAMID_KEYS

            for key in edge_keys or PYRAMID_KEYS:
                self.targets[key] = edge[key][ordered].float().numpy()
            del edge
        self.rgb = None
        if load_rgb:
            rgb_root = root / "rich_dino_color_v2"
            manifest = json.loads((rgb_root / "manifest.json").read_text(encoding="utf-8"))
            lookup = {name: i for i, name in enumerate(manifest["filenames"])}
            self.rgb_positions = np.asarray([lookup[n] for n in self.names])
            self.rgb = np.load(rgb_root / "rgb.npy", mmap_mode="r")

    def split(self, split: str, *, valid_edges: bool = False) -> np.ndarray:
        if split not in {"train", "val", "test"}:
            raise ValueError(split)
        return np.asarray(
            [
                i
                for i, row in enumerate(self.rows)
                if row["split"] == split and (not valid_edges or self.valid[i])
            ],
            dtype=np.int64,
        )

    def mean_eeg(self, repeats: int = 4, end_ms: int = 800) -> np.ndarray:
        if repeats not in (1, 2, 4) or end_ms not in (200, 400, 800):
            raise ValueError("supported repeats 1/2/4; windows 200/400/800 ms")
        return np.asarray(self.eeg[self.indices, :repeats, :, 20 : 20 + end_ms // 10]).mean(1)


class SingleSubjectDataset(Dataset):
    def __init__(
        self, bank, split, task, *, end_ms=800, repeats=4, repeat_augmentation=False, limit=0
    ):
        if end_ms not in (200, 400, 800) or repeats not in (1, 2, 4):
            raise ValueError("invalid EEG window or repetitions")
        self.bank, self.task, self.end_ms = bank, task, end_ms
        self.repeats, self.repeat_augmentation = repeats, repeat_augmentation
        self.rows = bank.split(split, valid_edges=task == "edge")
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        position = int(self.rows[item])
        count = self.repeats
        if self.repeat_augmentation:
            count = (1, 2, 4)[int(torch.randint(3, ()))]
            chosen = torch.randperm(4)[:count].numpy()
        else:
            chosen = np.arange(count)
        eeg = np.array(
            self.bank.eeg[self.bank.indices[position], chosen, :, : 20 + self.end_ms // 10]
        ).mean(0)
        result = {"eeg": torch.from_numpy(eeg), "row": position, "subject_id": self.bank.subject}
        if self.task in ("dino", "clip_b32", "synclr"):
            result["target"] = torch.from_numpy(self.bank.targets[self.task][position].copy())
        elif self.task == "edge":
            from .build_pyramid_edge_targets import PYRAMID_KEYS

            result.update(
                {k: torch.from_numpy(self.bank.targets[k][position].copy()) for k in PYRAMID_KEYS}
            )
        elif self.task == "color":
            if self.bank.rgb is None:
                raise ValueError("RGB targets were not loaded")
            rgb = np.array(self.bank.rgb[self.bank.rgb_positions[position]], copy=True)
            result["rgb"] = torch.from_numpy(rgb).float() / 255
        else:
            raise ValueError(self.task)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=Path, default=Path("data/derived/hierarchical_visual_targets/index.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/derived/single_subject_geometry/protocol.json")
    )
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    protocol = save_protocol(args.index, args.output, args.seed)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "images": len(protocol["rows"]),
                "sequence_disjoint_verified": False,
            }
        )
    )


if __name__ == "__main__":
    main()
