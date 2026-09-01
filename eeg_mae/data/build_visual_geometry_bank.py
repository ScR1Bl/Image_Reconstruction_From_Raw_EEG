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
    parser = argparse.ArgumentParser(description="Build dense RGB/edge/saliency/depth teachers")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--output", default="data/derived/rich_visual_targets_32.pt")
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--teacher-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--depth-model", default="depth-anything/Depth-Anything-V2-Small-hf")
    parser.add_argument("--depth-revision", default="5426e4f0f36572d16453bbda7a8389317b1bef99")
    parser.add_argument("--saliency-model", default="ZhengPeng7/BiRefNet_lite")
    parser.add_argument("--saliency-revision", default="7838f1c3472f827cd8ce13ab5ccc2ce48077360f")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-saliency", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=250,
        help="Save resumable progress every N batches (0 disables it)",
    )
    return parser.parse_args()


def unique_names(index: Path) -> list[str]:
    with index.open(newline="", encoding="utf-8") as handle:
        names = sorted({row["image_file"] for row in csv.DictReader(handle)})
    if not names or "" in names:
        raise ValueError("index contains no valid image_file values")
    return names


def image_tensor(image: Image.Image, size: int) -> torch.Tensor:
    array = (
        np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32
        ).copy()
        / 255.0
    )
    return torch.from_numpy(array).permute(2, 0, 1)


def sobel_edges(rgb: torch.Tensor) -> torch.Tensor:
    gray = (rgb * rgb.new_tensor([0.299, 0.587, 0.114])[:, None, None]).sum(0, keepdim=True)
    kernels = rgb.new_tensor(
        [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]
    )[:, None]
    gradient = F.conv2d(gray[None], kernels, padding=1).square().sum(1, keepdim=True).sqrt()[0]
    # A robust per-image scale preserves weak contours without allowing one outlier to dominate.
    scale = torch.quantile(gradient.flatten(), 0.95).clamp_min(1e-4)
    return (gradient / scale).clamp(0, 1)


def robust_depth(depth: torch.Tensor, size: int) -> torch.Tensor:
    depth = F.interpolate(
        depth[:, None].float(), (size, size), mode="bicubic", align_corners=False
    ).squeeze(1)
    flat = depth.flatten(1)
    low = torch.quantile(flat, 0.02, dim=1)[:, None, None]
    high = torch.quantile(flat, 0.98, dim=1)[:, None, None]
    return ((depth - low) / (high - low).clamp_min(1e-6)).clamp(0, 1)[:, None]


def metadata(args: argparse.Namespace) -> dict:
    return {
        "size": args.size,
        "teacher_size": args.teacher_size,
        "depth_model": None if args.no_depth else args.depth_model,
        "depth_revision": None if args.no_depth else args.depth_revision,
        "saliency_model": None if args.no_saliency else args.saliency_model,
        "saliency_revision": None if args.no_saliency else args.saliency_revision,
        "depth_is_relative_per_image": True,
        "saliency_is_teacher_pseudolabel": True,
        "complete": not args.max_images,
    }


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = arguments()
    if args.size < 8 or args.size % 8:
        raise ValueError("--size must be a multiple of 8 and at least 8")
    names = unique_names(Path(args.index))
    if args.max_images:
        names = names[: args.max_images]
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output.with_suffix(output.suffix + ".partial")
    run_metadata = metadata(args)

    rgb_all, edge_all, saliency_all, depth_all = [], [], [], []
    processed = 0
    if partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        processed = len(partial["filenames"])
        if partial["filenames"] != names[:processed] or partial["metadata"] != run_metadata:
            raise RuntimeError(f"incompatible partial target bank: {partial_path}")
        if processed > len(names):
            raise RuntimeError(f"partial target bank has too many entries: {processed}")
        rgb_all.append(partial["rgb"])
        edge_all.append(partial["edge"])
        saliency_all.append(partial["saliency"])
        depth_all.append(partial["depth"])
        print(f"resuming rich targets at {processed}/{len(names)} from {partial_path}", flush=True)

    depth_processor = depth_model = None
    if not args.no_depth:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        depth_processor = AutoImageProcessor.from_pretrained(
            args.depth_model, revision=args.depth_revision
        )
        depth_model = (
            AutoModelForDepthEstimation.from_pretrained(
                args.depth_model, revision=args.depth_revision
            )
            .to(device)
            .eval()
        )

    saliency_model = None
    if not args.no_saliency:
        from transformers import AutoModelForImageSegmentation

        saliency_model = (
            AutoModelForImageSegmentation.from_pretrained(
                args.saliency_model,
                revision=args.saliency_revision,
                trust_remote_code=True,
            )
            .to(device)
            .eval()
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
            rgb = torch.stack([image_tensor(image, args.size) for image in images])
            rgb_all.append(rgb.half())
            edge_all.append(torch.stack([sobel_edges(item) for item in rgb]).half())

            if saliency_model is not None:
                large = torch.stack(
                    [image_tensor(image, args.teacher_size) for image in images]
                ).to(device)
                mean = large.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                std = large.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"),
                ):
                    saliency = saliency_model((large - mean) / std)[-1].sigmoid().float()
                saliency = F.interpolate(
                    saliency, (args.size, args.size), mode="bilinear", align_corners=False
                )
                saliency_all.append(saliency.cpu().half())
            else:
                saliency_all.append(
                    torch.zeros(len(images), 1, args.size, args.size, dtype=torch.float16)
                )

            if depth_model is not None and depth_processor is not None:
                pixels = depth_processor(images=images, return_tensors="pt")["pixel_values"].to(
                    device
                )
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"),
                ):
                    predicted = depth_model(pixel_values=pixels).predicted_depth.float()
                depth_all.append(robust_depth(predicted, args.size).cpu().half())
            else:
                depth_all.append(
                    torch.zeros(len(images), 1, args.size, args.size, dtype=torch.float16)
                )
            print(f"rich targets {min(start + len(images), len(names))}/{len(names)}", flush=True)
            completed_batches = (start // args.batch_size) + 1
            if args.checkpoint_every and completed_batches % args.checkpoint_every == 0:
                partial_bank = {
                    "filenames": names[: start + len(images)],
                    "rgb": torch.cat(rgb_all),
                    "edge": torch.cat(edge_all),
                    "saliency": torch.cat(saliency_all),
                    "depth": torch.cat(depth_all),
                    "metadata": run_metadata,
                }
                atomic_save(partial_bank, partial_path)
                print(f"checkpointed targets to {partial_path}", flush=True)

    bank = {
        "filenames": names,
        "rgb": torch.cat(rgb_all),
        "edge": torch.cat(edge_all),
        "saliency": torch.cat(saliency_all),
        "depth": torch.cat(depth_all),
        "metadata": run_metadata,
    }
    expected = (len(names), args.size, args.size)
    for key, channels in (("rgb", 3), ("edge", 1), ("saliency", 1), ("depth", 1)):
        if tuple(bank[key].shape) != (expected[0], channels, expected[1], expected[2]):
            raise RuntimeError(f"invalid {key} shape: {tuple(bank[key].shape)}")
        if not torch.isfinite(bank[key]).all():
            raise RuntimeError(f"non-finite values in {key}")
    atomic_save(bank, output)
    if partial_path.exists():
        partial_path.unlink()
    output.with_suffix(".json").write_text(
        json.dumps({**bank["metadata"], "images": len(names)}, indent=2)
    )
    print(f"saved {len(names)} rich visual targets to {output}", flush=True)


if __name__ == "__main__":
    main()
