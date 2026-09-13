"""Render explicit conditioning variants with paired seeds and a frozen PixArt.

No text label or target image is fed to the EEG-conditioned variants.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from ..data.dit_ten_subjects import ARTIFACTS, OUTPUT
from ..lightning.common import sha256_of_file
from ..lightning.dit_adapter import DiTAdapterLightning


def validate_render_signature(output, signature):
    path = output / "render_config.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != signature:
            raise ValueError("render settings or weights changed; choose a fresh output directory")
    elif any(output.glob("*/*.png")):
        raise ValueError(
            "existing images have no verified render configuration; choose a fresh output"
        )
    else:
        path.write_text(json.dumps(signature, indent=2), encoding="utf-8")


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, default=OUTPUT / "baseline_conditions.pt")
    parser.add_argument(
        "--checkpoint", type=Path, default=ARTIFACTS / "dit_adapter/faza1-t8c/checkpoints/best.ckpt"
    )
    parser.add_argument("--null", type=Path, default=ARTIFACTS / "dit_adapter/t5_embeddings.pt")
    parser.add_argument("--output", type=Path, default=OUTPUT / "baseline_generation")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.conditions, map_location="cpu", weights_only=False)
    count = min(args.count, len(bundle["rows"]))
    signature = {
        "checkpoint_sha256": sha256_of_file(args.checkpoint),
        "conditions_sha256": sha256_of_file(args.conditions),
        "null_sha256": sha256_of_file(args.null),
        "count": count,
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
    }
    validate_render_signature(args.output, signature)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    hparams = {**state["hyper_parameters"], "null_embeddings_file": str(args.null)}
    module = DiTAdapterLightning(**hparams).eval().requires_grad_(False)
    module.load_state_dict(state["state_dict"], strict=True)
    del state
    module = module.cuda()
    variants = bundle["variants"]
    started = time.monotonic()
    for name, condition in variants.items():
        directory = args.output / name
        directory.mkdir(exist_ok=True)
        for i in range(count):
            destination = directory / f"{i:03d}.png"
            if destination.exists():
                continue
            latents = module.sample_images(
                condition[i : i + 1].cuda(),
                steps=args.steps,
                guidance=args.guidance,
                seed=args.seed + i,
            )
            module.decode_and_grid(latents).save(destination)
            print(
                json.dumps(
                    {
                        "variant": name,
                        "image": i + 1,
                        "count": count,
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
    width = 256
    label_width = 150
    grid = Image.new(
        "RGB", (label_width + count * width, (len(variants) + 1) * (width + 25)), "white"
    )
    draw = ImageDraw.Draw(grid)
    labels = ["Stimulus"] + list(variants)
    with zipfile.ZipFile("data/things_eeg2_osf/image_set/training_images.zip") as z:
        for line, label in enumerate(labels):
            y = line * (width + 25)
            draw.text((5, y + 30), label, fill="black")
            for i in range(count):
                row = bundle["rows"][i]
                if line == 0:
                    img = Image.open(io.BytesIO(z.read(row["zip_member"]))).convert("RGB")
                else:
                    img = Image.open(args.output / label / f"{i:03d}.png").convert("RGB")
                grid.paste(img.resize((width, width)), (label_width + i * width, y + 25))
                draw.text((label_width + i * width + 3, y + 3), row["filename"], fill="black")
    grid.save(args.output / "comparison.png")
    report = {
        "complete": True,
        "metadata": bundle["metadata"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_of_file(args.checkpoint),
        "conditions_sha256": sha256_of_file(args.conditions),
        "steps": args.steps,
        "guidance": args.guidance,
        "seed_per_image": [args.seed + i for i in range(count)],
        "variants": list(variants),
        "count": count,
        "seconds": time.monotonic() - started,
        "test_evaluated": False,
        "comparison": "same initial noise for every variant of an image",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
