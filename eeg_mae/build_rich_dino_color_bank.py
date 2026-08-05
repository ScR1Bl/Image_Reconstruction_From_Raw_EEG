"""Build a resumable, memory-mapped multi-layer DINO bank for color oracle v2.

Unlike the legacy bank this preserves the native 16x16 spatial grid, all 384
channels, feature magnitude, and several transformer depths.
"""

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


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build lossless multi-layer DINO color features")
    p.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    p.add_argument("--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip")
    p.add_argument("--output", default="data/derived/rich_dino_color_v2")
    p.add_argument("--teacher", default="facebook/dinov2-small")
    p.add_argument("--revision", default="main")
    p.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9, 12])
    p.add_argument("--rgb-size", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def names_from_index(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        names = sorted({row["image_file"] for row in csv.DictReader(handle)})
    if not names or "" in names:
        raise ValueError("index contains no valid image filenames")
    return names


def rgb_uint8(image: Image.Image, size: int) -> np.ndarray:
    return (
        np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.uint8
        )
        .transpose(2, 0, 1)
        .copy()
    )


def write_manifest(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = arguments()
    if sorted(set(args.layers)) != args.layers or min(args.layers) < 1:
        raise ValueError("--layers must be unique, sorted, positive transformer layer numbers")
    names = names_from_index(Path(args.index))
    if args.max_images:
        names = names[: args.max_images]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    configuration = {
        "format": 2,
        "teacher": args.teacher,
        "revision": args.revision,
        "layers": args.layers,
        "rgb_size": args.rgb_size,
        "images": len(names),
        "filenames": names,
        "normalized": False,
        "projected": False,
    }
    processed = 0
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        comparable = {key: previous.get(key) for key in configuration}
        if comparable != configuration:
            raise RuntimeError(f"incompatible existing bank at {output}")
        processed = int(previous.get("processed", 0))

    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError(
            'Missing visual teacher dependencies. Run: python -m pip install -e ".[visual-teachers]"'
        ) from error
    processor = AutoImageProcessor.from_pretrained(args.teacher, revision=args.revision)
    model = AutoModel.from_pretrained(args.teacher, revision=args.revision).eval()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    dim = int(model.config.hidden_size)
    if max(args.layers) > int(model.config.num_hidden_layers):
        raise ValueError(f"teacher has only {model.config.num_hidden_layers} layers")
    crop_size = getattr(processor, "crop_size", None) or getattr(processor, "size", None)
    if hasattr(crop_size, "height"):
        crop_size = crop_size.height
    elif hasattr(crop_size, "get"):
        crop_size = crop_size.get("height") or crop_size.get("shortest_edge")
    if not isinstance(crop_size, int):
        raise TypeError(f"cannot determine processor crop size from {processor}")
    patch_side = int(crop_size // getattr(model.config, "patch_size", 14))
    shapes = {
        "patches": (len(names), len(args.layers), patch_side, patch_side, dim),
        "cls": (len(names), len(args.layers), dim),
        "rgb": (len(names), 3, args.rgb_size, args.rgb_size),
    }
    modes = "r+" if processed else "w+"
    patches = np.lib.format.open_memmap(
        output / "patches.npy", mode=modes, dtype=np.float16, shape=shapes["patches"]
    )
    cls = np.lib.format.open_memmap(
        output / "cls.npy", mode=modes, dtype=np.float16, shape=shapes["cls"]
    )
    rgb = np.lib.format.open_memmap(
        output / "rgb.npy", mode=modes, dtype=np.uint8, shape=shapes["rgb"]
    )

    with zipfile.ZipFile(args.images_zip) as archive:
        members: dict[str, list[str]] = {}
        for member in archive.namelist():
            if member.lower().endswith((".jpg", ".jpeg", ".png")):
                members.setdefault(Path(member).name, []).append(member)
        invalid = [name for name in names if len(members.get(name, [])) != 1]
        if invalid:
            raise RuntimeError(f"image pairing failed for {invalid[:5]}")
        for start in range(processed, len(names), args.batch_size):
            batch_names = names[start : start + args.batch_size]
            images = []
            for name in batch_names:
                with Image.open(io.BytesIO(archive.read(members[name][0]))) as image:
                    images.append(image.convert("RGB"))
            pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            with (
                torch.inference_mode(),
                torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"),
            ):
                hidden = model(pixel_values=pixels, output_hidden_states=True).hidden_states
            stop = start + len(images)
            for destination, layer in enumerate(args.layers):
                value = hidden[layer].float()
                tokens = value[:, 1 : 1 + patch_side * patch_side]
                if tokens.shape[1] != patch_side * patch_side:
                    raise RuntimeError(f"unexpected token count at layer {layer}: {value.shape}")
                patches[start:stop, destination] = (
                    tokens.reshape(len(images), patch_side, patch_side, dim)
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
                cls[start:stop, destination] = value[:, 0].cpu().numpy().astype(np.float16)
            rgb[start:stop] = np.stack([rgb_uint8(image, args.rgb_size) for image in images])
            processed = stop
            if (start // args.batch_size + 1) % args.checkpoint_every == 0 or processed == len(
                names
            ):
                patches.flush()
                cls.flush()
                rgb.flush()
                write_manifest(
                    manifest_path,
                    {
                        **configuration,
                        "processed": processed,
                        "complete": processed == len(names),
                        "hidden_dim": dim,
                        "patch_side": patch_side,
                        "storage_gib": round(
                            sum(
                                np.prod(shape)
                                * np.dtype(np.float16 if key != "rgb" else np.uint8).itemsize
                                for key, shape in shapes.items()
                            )
                            / 2**30,
                            3,
                        ),
                    },
                )
            print(f"rich DINO {processed}/{len(names)}", flush=True)
    print(f"saved complete rich DINO bank to {output}", flush=True)


if __name__ == "__main__":
    main()
