"""Paged audit panel: stimulus, selected training reference and both text modes."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from ..data.dit_ten_subjects import OUTPUT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, default=OUTPUT / "text_comparison_conditions.pt")
    parser.add_argument("--generation", type=Path, default=OUTPUT / "text_generation")
    args = parser.parse_args()
    bundle = torch.load(args.conditions, map_location="cpu", weights_only=False)
    status = json.loads((args.generation / "report.json").read_text())
    if not status["complete"]:
        raise ValueError("generation is incomplete")
    records = bundle["metadata"]["predictions"]
    try:
        font = ImageFont.truetype("arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    width, cell_h, left, top = 240, 280, 180, 70
    labels = ["Stimulus", "TRAIN reference", "EEG category text", "Text + TRAIN prior"]
    with zipfile.ZipFile("data/things_eeg2_osf/image_set/training_images.zip") as archive:
        for start in range(0, status["count"], 8):
            n = min(8, status["count"] - start)
            panel = Image.new("RGB", (left + n * width, top + len(labels) * cell_h), "white")
            draw = ImageDraw.Draw(panel)
            draw.text(
                (left, 8),
                "10 subjects | 4 repetitions | 0-800 ms | predicted category, not recovered instance geometry",
                fill="black",
                font=font,
            )
            draw.text(
                (left, 33),
                "First 8 fixed examples, then 16 random examples (seed 20260913); incorrect categories are retained.",
                fill="black",
                font=font,
            )
            for line, label in enumerate(labels):
                y = top + line * cell_h
                draw.text((6, y + 65), label, fill="black", font=font)
                for col in range(n):
                    i = start + col
                    row, record = bundle["rows"][i], records[i]
                    predicted = record["predicted_category"].split("_", 1)[1]
                    if line < 2:
                        member = (
                            row["zip_member"] if line == 0 else record["reference"]["zip_member"]
                        )
                        image = Image.open(io.BytesIO(archive.read(member))).convert("RGB")
                    else:
                        variant = "category_text_only" if line == 2 else "category_text_and_prior"
                        image = Image.open(args.generation / variant / f"{i:03d}.png").convert(
                            "RGB"
                        )
                    caption = row["filename"] if line == 0 else f"EEG: {predicted}"
                    color = (
                        "black" if line == 0 else ("darkgreen" if record["correct"] else "darkred")
                    )
                    draw.text((left + col * width + 3, y + 4), caption, fill=color, font=font)
                    panel.paste(image.resize((width - 4, width - 4)), (left + col * width, y + 30))
            panel.save(args.generation / f"audit_page_{start // 8 + 1}.png")
    print(args.generation)


if __name__ == "__main__":
    main()
