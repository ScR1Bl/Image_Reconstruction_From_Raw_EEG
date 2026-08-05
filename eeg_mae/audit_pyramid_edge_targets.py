from __future__ import annotations

import argparse
import io
import random
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def arguments():
    parser = argparse.ArgumentParser(description="Render a pyramid-edge target quality panel")
    parser.add_argument("--bank", default="data/derived/pyramid_edge_targets_v1.pt")
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--output", default="runs/pyramid_edge_target_audit.png")
    parser.add_argument("--images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def gray(value, size):
    array = (value.squeeze().float().clamp(0, 1) * 255).to(torch.uint8).numpy()
    return Image.fromarray(array, "L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def main():
    args = arguments()
    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    valid = torch.where(bank["valid"])[0].tolist()
    selected = random.Random(args.seed).sample(valid, min(args.images, len(valid)))
    headers = (
        "original",
        "shape 16",
        "distance 16",
        "semantic 32",
        "perceptual 64",
        "crisp 64",
        "confidence",
        "uncertainty",
    )
    tile, label = 160, 24
    canvas = Image.new("RGB", (len(headers) * tile, (len(selected) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    with zipfile.ZipFile(args.images_zip) as archive:
        members = {
            Path(member).name: member
            for member in archive.namelist()
            if member.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        for row, index in enumerate(selected, 1):
            name = bank["filenames"][index]
            y = row * (tile + label)
            draw.text((3, y - label + 3), name, fill="black")
            with Image.open(io.BytesIO(archive.read(members[name]))) as image:
                original = image.convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            images = (
                original,
                gray(bank["shape16"][index], tile),
                gray(bank["distance16"][index], tile),
                gray(bank["semantic32"][index], tile),
                gray(bank["perceptual64"][index], tile),
                gray(bank["crisp64"][index], tile),
                gray(bank["confidence64"][index], tile),
                gray(bank["uncertainty64"][index], tile),
            )
            for column, image in enumerate(images):
                canvas.paste(image, (column * tile, y))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"saved {len(selected)} target examples to {output}", flush=True)


if __name__ == "__main__":
    main()
