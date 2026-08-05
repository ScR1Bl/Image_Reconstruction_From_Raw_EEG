"""Export full 16x16 antiring-oracle latents for EEG supervision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .train_color_oracle_v2 import RichDINOColorDataset
from .train_compact_color_tokens import load_teacher


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build full color latent bank from oracle_v2_antiring"
    )
    parser.add_argument("--rich-bank", default="data/derived/rich_dino_color_v2")
    parser.add_argument("--teacher", default="runs/dino_color_oracle_v2_antiring/best.pt")
    parser.add_argument("--output", default="data/derived/full_color_latents_antiring")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    source = Path(args.rich_bank)
    output = Path(args.output)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    filenames = source_manifest["filenames"][: args.limit]
    teacher, payload = load_teacher(
        Path(args.teacher),
        torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"),
    )
    config = payload["model_config"]
    if config["input_dim"] != source_manifest["hidden_dim"] or config["layers"] != len(
        source_manifest["layers"]
    ):
        raise ValueError("teacher and rich DINO bank are incompatible")
    output.mkdir(parents=True, exist_ok=True)
    count, width = len(filenames), config["width"]
    manifest_path = output / "manifest.json"
    expected = {
        "format": 1,
        "complete": False,
        "processed": 0,
        "images": count,
        "filenames": filenames,
        "grid_shape": [16, 16, width],
        "global_dim": width,
        "dtype": "float16",
        "teacher_checkpoint": str(Path(args.teacher)),
        "teacher_epoch": payload.get("epoch"),
        "source_rich_bank": str(source),
    }
    start = 0
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        immutable = (
            "format",
            "images",
            "filenames",
            "grid_shape",
            "global_dim",
            "dtype",
            "teacher_checkpoint",
        )
        if any(current.get(key) != expected.get(key) for key in immutable):
            raise RuntimeError("existing latent-bank manifest is incompatible")
        if current.get("complete"):
            print(f"full color latent bank already complete: {output}")
            return
        start = int(current.get("processed", 0))
        expected.update(current)
    mode = "r+" if start else "w+"
    grid = np.lib.format.open_memmap(
        output / "grid.npy", mode=mode, dtype=np.float16, shape=(count, 16, 16, width)
    )
    global_feature = np.lib.format.open_memmap(
        output / "global.npy", mode=mode, dtype=np.float16, shape=(count, width)
    )
    expected["complete"] = False
    expected["processed"] = start
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    dataset = RichDINOColorDataset(source, filenames[start:])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    device = next(teacher.parameters()).device
    cursor = start
    with torch.inference_mode():
        for batch in loader:
            with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
                latent = teacher.encode_full_latent(
                    batch["patches"].to(device, non_blocking=True),
                    batch["cls"].to(device, non_blocking=True),
                )
            size = len(latent.grid)
            grid[cursor : cursor + size] = latent.grid.float().cpu().numpy().astype(np.float16)
            global_feature[cursor : cursor + size] = (
                latent.global_feature.float().cpu().numpy().astype(np.float16)
            )
            cursor += size
            if cursor % 256 < size or cursor == count:
                grid.flush()
                global_feature.flush()
                expected["processed"] = cursor
                manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
                print(f"latent bank {cursor}/{count}", flush=True)
    grid.flush()
    global_feature.flush()
    if (
        not np.isfinite(np.asarray(grid[:: max(1, count // 32)])).all()
        or not np.isfinite(np.asarray(global_feature[:: max(1, count // 32)])).all()
    ):
        raise RuntimeError("non-finite values in exported latent bank")
    expected["processed"] = count
    expected["complete"] = True
    manifest_path.write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"full color latent bank complete: {output}")


if __name__ == "__main__":
    main()
