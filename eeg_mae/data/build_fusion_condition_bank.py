"""Cache frozen EEG encoder outputs once before training the fusion prior."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from ..evaluation.official_200way_benchmark import cached_array
from ..models.color_encoder import load_color_encoder
from ..models.edge_encoder import load_edge_encoder
from ..models.hierarchical_semantic import BRANCH_DIMS, load_hierarchical_semantic_encoder
from .fusion_conditions import CONDITION_SHAPES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class EnsembleEEGDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], arrays: list[np.ndarray], pending: list[int]):
        self.rows, self.arrays, self.pending = rows, arrays, pending

    def __len__(self) -> int:
        return len(self.pending)

    def __getitem__(self, item: int):
        position = self.pending[item]
        image_index = int(self.rows[position]["image_index"])
        eeg = np.stack(
            [np.asarray(value[image_index], dtype=np.float32).mean(0) for value in self.arrays]
        )
        return position, torch.from_numpy(eeg)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build frozen EEG conditions for fusion-prior training"
    )
    parser.add_argument("--target-bank", default="data/derived/hierarchical_visual_targets")
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--eeg-cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--semantic-checkpoint", default="runs/eeg_hierarchical_encoder/best.pt")
    parser.add_argument("--color-root", default="runs/eeg_color_encoder_direct")
    parser.add_argument("--edge-root", default="runs/eeg_edge_encoder_direct")
    parser.add_argument("--output", default="data/derived/eeg_fusion_conditions")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    target_root, output = Path(args.target_bank), Path(args.output)
    manifest = json.loads((target_root / "manifest.json").read_text(encoding="utf-8"))
    with (target_root / manifest["index"]).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["position"]) for row in rows] != list(range(len(rows))):
        raise RuntimeError("target index positions must be contiguous")
    output.mkdir(parents=True, exist_ok=True)
    index_output = output / "index.csv"
    if not index_output.exists():
        index_output.write_bytes((target_root / manifest["index"]).read_bytes())

    arrays = {}
    for name, shape in CONDITION_SHAPES.items():
        path = output / f"{name}.float16.npy"
        if path.exists():
            value = np.load(path, mmap_mode="r+")
            if value.shape != (len(rows), *shape) or value.dtype != np.float16:
                raise RuntimeError(f"invalid resumable array: {path}")
        else:
            value = np.lib.format.open_memmap(
                path, mode="w+", dtype=np.float16, shape=(len(rows), *shape)
            )
        arrays[name] = value
    completed_path = output / "completed.npy"
    if completed_path.exists():
        completed = np.load(completed_path, mmap_mode="r+")
    else:
        completed = np.lib.format.open_memmap(
            completed_path, mode="w+", dtype=np.uint8, shape=(len(rows),)
        )
        completed[:] = 0
        completed.flush()
    pending = np.flatnonzero(np.asarray(completed) == 0).tolist()
    if pending:
        eeg_arrays = [
            cached_array(Path(args.archives), Path(args.eeg_cache), subject, "training")
            for subject in range(10)
        ]
        semantic, _, _ = load_hierarchical_semantic_encoder(args.semantic_checkpoint, device)
        color, _ = load_color_encoder(args.color_root, device)
        edge, _ = load_edge_encoder(args.edge_root, device)
        semantic.eval()
        loader = DataLoader(
            EnsembleEEGDataset(rows, eeg_arrays, pending),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        for number, (positions, eeg) in enumerate(loader, 1):
            eeg = eeg.to(device, non_blocking=True)
            batch, subjects = eeg.shape[:2]
            ids = torch.arange(subjects, device=device).repeat(batch)
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                semantic_output = semantic(eeg.flatten(0, 1), ids)
                color_output = color(eeg)
                edge_output = edge(eeg)

            def ensemble(
                value: torch.Tensor, batch_size: int = batch, subject_count: int = subjects
            ) -> torch.Tensor:
                value = F.normalize(
                    value.float().reshape(batch_size, subject_count, -1), dim=-1
                ).mean(1)
                return F.normalize(value, dim=-1)

            values = {
                "semantic_fused": ensemble(semantic_output.fused),
                **{
                    f"semantic_{name}": ensemble(semantic_output.branches[name])
                    for name in BRANCH_DIMS
                },
                "color_grid16": color_output.condition.grid16.float(),
                "edge_grid16": edge_output.condition.grid16.float(),
            }
            location = positions.numpy()
            for name, value in values.items():
                arrays[name][location] = value.cpu().numpy().astype(np.float16)
                arrays[name].flush()
            completed[location] = 1
            completed.flush()
            print(f"conditions {int(completed.sum())}/{len(rows)}", flush=True)

    condition_metadata = {}
    for name, value in arrays.items():
        path = output / f"{name}.float16.npy"
        condition_metadata[name] = {
            "array": path.name,
            "shape": list(value.shape),
            "dtype": "float16",
            "sha256": _sha256(path),
        }
    result = {
        "schema_version": 1,
        "complete": bool(np.asarray(completed).all()),
        "images": len(rows),
        "index": index_output.name,
        "index_sha256": _sha256(index_output),
        "split_counts": manifest["split_counts"],
        "source_target_fingerprint": manifest["source_fingerprint"],
        "conditions": condition_metadata,
        "checkpoints": {
            "semantic": {
                "path": args.semantic_checkpoint,
                "sha256": _sha256(Path(args.semantic_checkpoint)),
            },
            "color": {
                "path": str(Path(args.color_root) / "best.pt"),
                "sha256": _sha256(Path(args.color_root) / "best.pt"),
            },
            "edge": {
                "path": str(Path(args.edge_root) / "best.pt"),
                "sha256": _sha256(Path(args.edge_root) / "best.pt"),
            },
        },
        "inference_inputs": ["preprocessed_eeg", "known_subject_id"],
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"complete fusion condition bank: {output}")


if __name__ == "__main__":
    main()
