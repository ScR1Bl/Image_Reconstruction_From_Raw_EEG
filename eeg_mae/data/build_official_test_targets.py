from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from .build_visual_targets import color_descriptor


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DINO targets for the official 200-image THINGS-EEG2 test set"
    )
    parser.add_argument("--training-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--images-zip", default="data/things_eeg2_osf/image_set/test_images.zip")
    parser.add_argument("--output", default="data/derived/official_test_targets_dinov2s_192.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def ordered_test_members(archive: zipfile.ZipFile) -> list[str]:
    members = [
        name for name in archive.namelist() if name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    members.sort()  # numbered folders 00001..00200 define EEG condition order
    if len(members) != 200:
        raise RuntimeError(f"official test archive must contain 200 images, found {len(members)}")
    folders = [Path(name).parent.name.split("_", 1)[0] for name in members]
    expected = [f"{index:05d}" for index in range(1, 201)]
    if folders != expected:
        raise RuntimeError("test-image folders are not the official 00001..00200 condition order")
    return members


def main() -> None:
    from transformers import AutoImageProcessor, AutoModel

    args = arguments()
    training = torch.load(args.training_bank, map_location="cpu", weights_only=False)
    metadata = training.get("metadata", {})
    teacher_name = metadata.get("teacher")
    projection = training.get("projection")
    if not teacher_name or projection is None:
        raise RuntimeError("training bank lacks teacher/projection metadata")
    grid_size = int(metadata.get("grid_size", training["dino_grid"].shape[1]))
    projected_dim = int(metadata.get("projected_dim", projection.shape[1]))
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    processor = AutoImageProcessor.from_pretrained(teacher_name)
    teacher = AutoModel.from_pretrained(teacher_name).to(device).eval()
    projection = projection.float().to(device)

    names: list[str] = []
    concepts: list[str] = []
    all_global: list[torch.Tensor] = []
    all_grid: list[torch.Tensor] = []
    all_color: list[torch.Tensor] = []
    with zipfile.ZipFile(args.images_zip) as archive:
        members = ordered_test_members(archive)
        for start in range(0, len(members), args.batch_size):
            batch_members = members[start : start + args.batch_size]
            images: list[Image.Image] = []
            for member in batch_members:
                with Image.open(io.BytesIO(archive.read(member))) as image:
                    images.append(image.convert("RGB"))
                names.append(Path(member).name)
                concepts.append(Path(member).parent.name.split("_", 1)[1])
            all_color.append(torch.stack([color_descriptor(image) for image in images]))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            with (
                torch.no_grad(),
                torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"),
            ):
                hidden = teacher(pixel_values=pixels).last_hidden_state.float()
                global_features = F.normalize(hidden[:, 0], dim=-1)
                patches = hidden[:, 1:]
                side = round(patches.shape[1] ** 0.5)
                if side * side != patches.shape[1]:
                    raise RuntimeError(f"teacher patch count is not square: {patches.shape[1]}")
                patches = patches.reshape(len(images), side, side, hidden.shape[-1]).permute(
                    0, 3, 1, 2
                )
                patches = F.adaptive_avg_pool2d(patches, (grid_size, grid_size)).permute(0, 2, 3, 1)
                patches = F.normalize(patches @ projection, dim=-1)
            all_global.append(global_features.cpu().half())
            all_grid.append(patches.cpu().half())
            print(f"official targets {min(start + len(images), 200)}/200", flush=True)

    bank = {
        "filenames": names,
        "concepts": concepts,
        "condition_index": torch.arange(200),
        "color": torch.cat(all_color),
        "dino_global": torch.cat(all_global),
        "dino_grid": torch.cat(all_grid),
        "projection": projection.cpu(),
        "metadata": {
            "teacher": teacher_name,
            "grid_size": grid_size,
            "projected_dim": projected_dim,
            "official_conditions": 200,
            "condition_order": "numeric test-image folder 00001..00200",
            "source_zip": str(args.images_zip),
            "training_bank": str(args.training_bank),
            "complete": True,
        },
    }
    if bank["dino_global"].shape != (200, training["dino_global"].shape[1]):
        raise RuntimeError("official global target dimensions do not match the training bank")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    output.with_suffix(".json").write_text(
        json.dumps({**bank["metadata"], "filenames": names, "concepts": concepts}, indent=2),
        encoding="utf-8",
    )
    print(f"saved official 200-condition target bank to {output}", flush=True)


if __name__ == "__main__":
    main()
