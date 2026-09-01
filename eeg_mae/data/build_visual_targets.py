from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build deterministic visual targets for stage 3")
    p.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    p.add_argument("--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip")
    p.add_argument("--output", default="data/derived/visual_targets_dinov2s.pt")
    p.add_argument("--teacher", default="facebook/dinov2-small")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--projected-dim", type=int, default=64)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-images", type=int, default=0)
    return p.parse_args()


def color_descriptor(image: Image.Image) -> torch.Tensor:
    array = (
        np.asarray(
            image.convert("RGB").resize((64, 64), Image.Resampling.BICUBIC), dtype=np.float32
        )
        / 255.0
    )
    pixels = array.reshape(-1, 3)
    statistics = np.concatenate(
        [
            pixels.mean(0),
            pixels.std(0),
            np.quantile(pixels, 0.10, axis=0),
            np.quantile(pixels, 0.50, axis=0),
            np.quantile(pixels, 0.90, axis=0),
        ]
    )
    grid = array.reshape(4, 16, 4, 16, 3).mean((1, 3)).reshape(-1)
    return torch.from_numpy(np.concatenate([statistics, grid]).astype(np.float32))


def unique_index_images(index_path: Path) -> list[str]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        names = {row["image_file"] for row in csv.DictReader(handle)}
    if "" in names:
        raise ValueError("index contains an empty image_file")
    return sorted(names)


def main() -> None:
    from transformers import AutoImageProcessor, AutoModel

    args = arguments()
    index_path, zip_path, output = Path(args.index), Path(args.images_zip), Path(args.output)
    names = unique_index_images(index_path)
    if args.max_images:
        names = names[: args.max_images]
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    processor = AutoImageProcessor.from_pretrained(args.teacher)
    teacher = AutoModel.from_pretrained(args.teacher).to(device).eval()
    hidden_dim = teacher.config.hidden_size
    generator = torch.Generator().manual_seed(args.seed)
    projection = torch.randn(hidden_dim, args.projected_dim, generator=generator)
    projection = torch.linalg.qr(projection, mode="reduced").Q.to(device)

    with zipfile.ZipFile(zip_path) as archive:
        by_basename: dict[str, list[str]] = {}
        for member in archive.namelist():
            if member.lower().endswith((".jpg", ".jpeg", ".png")):
                by_basename.setdefault(Path(member).name, []).append(member)
        missing = [name for name in names if name not in by_basename]
        ambiguous = {
            name: by_basename[name] for name in names if len(by_basename.get(name, [])) != 1
        }
        if missing or ambiguous:
            raise RuntimeError(
                f"image pairing failed: missing={missing[:5]}, ambiguous={list(ambiguous)[:5]}"
            )

        all_grid, all_global, all_color = [], [], []
        for start in range(0, len(names), args.batch_size):
            batch_names = names[start : start + args.batch_size]
            images = []
            for name in batch_names:
                with Image.open(io.BytesIO(archive.read(by_basename[name][0]))) as image:
                    images.append(image.convert("RGB"))
            all_color.append(torch.stack([color_descriptor(image) for image in images]))
            inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ),
            ):
                output_features = teacher(pixel_values=inputs).last_hidden_state
                global_features = F.normalize(output_features[:, 0].float(), dim=-1)
                patches = output_features[:, 1:].float()
                side = round(patches.shape[1] ** 0.5)
                if side * side != patches.shape[1]:
                    raise RuntimeError(f"teacher patch count is not square: {patches.shape[1]}")
                patches = patches.reshape(len(images), side, side, hidden_dim).permute(0, 3, 1, 2)
                patches = F.adaptive_avg_pool2d(patches, (args.grid_size, args.grid_size)).permute(
                    0, 2, 3, 1
                )
                patches = F.normalize(patches @ projection, dim=-1)
            all_global.append(global_features.cpu().half())
            all_grid.append(patches.cpu().half())
            print(f"targets {min(start + len(images), len(names))}/{len(names)}", flush=True)

    bank = {
        "filenames": names,
        "color": torch.cat(all_color),
        "dino_global": torch.cat(all_global),
        "dino_grid": torch.cat(all_grid),
        "projection": projection.cpu(),
        "metadata": {
            "teacher": args.teacher,
            "grid_size": args.grid_size,
            "projected_dim": args.projected_dim,
            "seed": args.seed,
            "complete": not args.max_images,
            "source_zip": str(zip_path),
            "color_layout": "rgb mean/std/q10/q50/q90 (15) + RGB 4x4 grid (48)",
        },
    }
    if len(set(names)) != len(names) or any(
        t.shape[0] != len(names) for t in (bank["color"], bank["dino_grid"])
    ):
        raise RuntimeError("target bank integrity check failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    output.with_suffix(".json").write_text(
        json.dumps({**bank["metadata"], "images": len(names)}, indent=2)
    )
    print(f"saved {len(names)} verified targets to {output}")


if __name__ == "__main__":
    main()
