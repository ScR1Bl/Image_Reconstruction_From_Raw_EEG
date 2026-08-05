from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt
from torch.nn import functional as F

from .build_rich_visual_targets import atomic_save, image_tensor


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build clean semantic-edge, detail-edge and signed-distance targets"
    )
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--output", default="data/derived/edge_targets_64_v2.pt")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--teacher-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--saliency-model", default="ZhengPeng7/BiRefNet_lite")
    parser.add_argument("--saliency-revision", default="7838f1c3472f827cd8ce13ab5ccc2ce48077360f")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--min-foreground", type=float, default=0.015)
    parser.add_argument("--max-foreground", type=float, default=0.90)
    parser.add_argument("--min-semantic-density", type=float, default=0.025)
    parser.add_argument("--max-semantic-density", type=float, default=0.50)
    parser.add_argument("--max-components", type=int, default=5)
    parser.add_argument("--min-component-pixels", type=int, default=8)
    return parser.parse_args()


def unique_names(index: Path) -> list[str]:
    with index.open(newline="", encoding="utf-8") as handle:
        return sorted({row["image_file"] for row in csv.DictReader(handle)})


def soft_boundary(mask: torch.Tensor, kernels: tuple[int, int] = (3, 5)) -> torch.Tensor:
    boundaries = []
    for kernel in kernels:
        dilation = F.max_pool2d(mask, kernel, stride=1, padding=kernel // 2)
        erosion = -F.max_pool2d(-mask, kernel, stride=1, padding=kernel // 2)
        boundaries.append((dilation - erosion).clamp(0, 1))
    return (0.7 * boundaries[0] + 0.3 * boundaries[1]).clamp(0, 1)


def downsample_mask_preserve_thin(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Reduce a soft foreground mask without deleting confident thin structures."""
    if mask.shape[-2:] == (size, size):
        return mask.clamp(0, 1)
    linear = F.interpolate(mask, (size, size), mode="bilinear", align_corners=False)
    mean = F.adaptive_avg_pool2d(mask, (size, size))
    peak = F.adaptive_max_pool2d(mask, (size, size))
    # A high-confidence one-pixel structure survives, while weak teacher noise does not
    # automatically become foreground in the signed-distance target.
    return torch.maximum(torch.maximum(linear, mean), 0.75 * peak).clamp(0, 1)


def downsample_boundary_preserve_thin(boundary: torch.Tensor, size: int) -> torch.Tensor:
    """Conservative max/mean reduction used after extracting edges at teacher scale."""
    if boundary.shape[-2:] == (size, size):
        return boundary.clamp(0, 1)
    mean = F.adaptive_avg_pool2d(boundary, (size, size))
    peak = F.adaptive_max_pool2d(boundary, (size, size))
    return torch.maximum(mean, 0.85 * peak).clamp(0, 1)


def sparse_detail_edges(rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = []
    for image, foreground in zip(rgb, mask):
        array = (image.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.1)
        # Conservative thresholds intentionally discard texture and retain stable contours.
        edge = cv2.Canny(gray, 70, 170, L2gradient=True).astype(np.float32) / 255.0
        gate = cv2.dilate(
            (foreground[0].numpy() > 0.08).astype(np.uint8), np.ones((5, 5), np.uint8)
        )
        edge *= gate
        result.append(torch.from_numpy(edge)[None])
    return torch.stack(result)


def signed_distance_targets(
    mask: torch.Tensor,
    radius: float = 12.0,
    min_foreground: float = 0.005,
    max_foreground: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    fields, valid = [], []
    for value in mask:
        foreground = value[0].numpy() >= 0.5
        fraction = float(foreground.mean())
        good = min_foreground <= fraction <= max_foreground
        valid.append(good)
        if good:
            inside = distance_transform_edt(foreground)
            outside = distance_transform_edt(~foreground)
            signed = np.clip(inside - outside, -radius, radius) / radius
            field = (signed + 1.0) * 0.5
        else:
            field = np.zeros_like(foreground, dtype=np.float32)
        fields.append(torch.from_numpy(field.astype(np.float32))[None])
    return torch.stack(fields), torch.tensor(valid, dtype=torch.bool)


def quality_validity(
    mask: torch.Tensor,
    semantic: torch.Tensor,
    *,
    min_foreground: float,
    max_foreground: float,
    min_semantic_density: float,
    max_semantic_density: float,
    max_components: int,
    min_component_pixels: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reject blank, nearly full, excessively dense and fragmented pseudo-labels."""
    foreground = mask[:, 0].numpy() >= 0.5
    contour = semantic[:, 0].numpy() >= 0.30
    fg_fraction = torch.from_numpy(foreground.mean(axis=(1, 2)).astype(np.float32))
    semantic_density = torch.from_numpy(contour.mean(axis=(1, 2)).astype(np.float32))
    component_counts = []
    for value in contour:
        count, _, stats, _ = cv2.connectedComponentsWithStats(value.astype(np.uint8), 8)
        component_counts.append(
            sum(int(area) >= min_component_pixels for area in stats[1:count, cv2.CC_STAT_AREA])
        )
    components = torch.tensor(component_counts, dtype=torch.int16)
    valid = (
        (fg_fraction >= min_foreground)
        & (fg_fraction <= max_foreground)
        & (semantic_density >= min_semantic_density)
        & (semantic_density <= max_semantic_density)
        & (components <= max_components)
    )
    return valid, {
        "foreground_fraction": fg_fraction,
        "semantic_density": semantic_density,
        "components": components,
    }


def main() -> None:
    args = arguments()
    if args.size != 64:
        raise ValueError("the VQ edge prior expects --size 64")
    names = unique_names(Path(args.index))
    if args.max_images:
        names = names[: args.max_images]
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output.with_suffix(output.suffix + ".partial")
    metadata = {
        "version": 2,
        "size": args.size,
        "teacher_size": args.teacher_size,
        "saliency_model": args.saliency_model,
        "saliency_revision": args.saliency_revision,
        "semantic_definition": "BiRefNet boundary at teacher resolution, max/mean reduced to preserve thin structures",
        "foreground_definition": "bilinear/mean/0.75*peak fusion of the soft teacher mask",
        "detail_definition": "foreground-gated conservative Canny plus semantic contour",
        "distance_definition": "signed foreground distance clipped at 12px and mapped to [0,1]",
        "quality_filter": {
            "min_foreground": args.min_foreground,
            "max_foreground": args.max_foreground,
            "min_semantic_density": args.min_semantic_density,
            "max_semantic_density": args.max_semantic_density,
            "max_components": args.max_components,
            "min_component_pixels": args.min_component_pixels,
        },
        "complete": not bool(args.max_images),
    }
    semantic_all, detail_all, distance_all, valid_all = [], [], [], []
    quality_all = {"foreground_fraction": [], "semantic_density": [], "components": []}
    processed = 0
    if partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if (
            partial["metadata"] != metadata
            or partial["filenames"] != names[: len(partial["filenames"])]
        ):
            raise RuntimeError(f"incompatible partial edge bank: {partial_path}")
        processed = len(partial["filenames"])
        for key, destination in (
            ("semantic", semantic_all),
            ("detail", detail_all),
            ("distance", distance_all),
            ("valid", valid_all),
        ):
            destination.append(partial[key])
        for key, values in quality_all.items():
            values.append(partial["quality"][key])
        print(f"resuming edge targets at {processed}/{len(names)}", flush=True)

    from transformers import AutoModelForImageSegmentation

    teacher = (
        AutoModelForImageSegmentation.from_pretrained(
            args.saliency_model, revision=args.saliency_revision, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    with zipfile.ZipFile(args.images_zip) as archive:
        members = {}
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
            large = torch.stack([image_tensor(image, args.teacher_size) for image in images]).to(
                device
            )
            mean = large.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = large.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            with (
                torch.inference_mode(),
                torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"),
            ):
                mask = teacher((large - mean) / std)[-1].sigmoid().float()
            mask = mask.cpu()
            # Extract topology before 512 -> 64 reduction. The old order erased thin
            # high-confidence structures before an edge could be formed.
            semantic_high = soft_boundary(mask, kernels=(5, 11))
            semantic = downsample_boundary_preserve_thin(semantic_high, args.size)
            mask = downsample_mask_preserve_thin(mask, args.size)
            detail = torch.maximum(sparse_detail_edges(rgb, mask), (semantic > 0.30).float())
            distance, distance_valid = signed_distance_targets(
                mask, min_foreground=args.min_foreground, max_foreground=args.max_foreground
            )
            valid, quality = quality_validity(
                mask,
                semantic,
                min_foreground=args.min_foreground,
                max_foreground=args.max_foreground,
                min_semantic_density=args.min_semantic_density,
                max_semantic_density=args.max_semantic_density,
                max_components=args.max_components,
                min_component_pixels=args.min_component_pixels,
            )
            valid &= distance_valid
            semantic_all.append(semantic.half())
            detail_all.append(detail.half())
            distance_all.append(distance.half())
            valid_all.append(valid)
            for key, values in quality_all.items():
                values.append(quality[key])
            done = start + len(images)
            print(
                f"edge targets {done}/{len(names)} valid={int(valid.sum())}/{len(valid)}",
                flush=True,
            )
            if (
                args.checkpoint_every
                and ((start // args.batch_size) + 1) % args.checkpoint_every == 0
            ):
                atomic_save(
                    {
                        "filenames": names[:done],
                        "semantic": torch.cat(semantic_all),
                        "detail": torch.cat(detail_all),
                        "distance": torch.cat(distance_all),
                        "valid": torch.cat(valid_all),
                        "quality": {key: torch.cat(value) for key, value in quality_all.items()},
                        "metadata": metadata,
                    },
                    partial_path,
                )

    bank = {
        "filenames": names,
        "semantic": torch.cat(semantic_all),
        "detail": torch.cat(detail_all),
        "distance": torch.cat(distance_all),
        "valid": torch.cat(valid_all),
        "quality": {key: torch.cat(value) for key, value in quality_all.items()},
        "metadata": metadata,
    }
    expected = (len(names), 1, args.size, args.size)
    for key in ("semantic", "detail", "distance"):
        if tuple(bank[key].shape) != expected or not torch.isfinite(bank[key]).all():
            raise RuntimeError(f"invalid {key}: {tuple(bank[key].shape)}")
    atomic_save(bank, output)
    if partial_path.exists():
        partial_path.unlink()
    output.with_suffix(".json").write_text(
        json.dumps(
            {**metadata, "images": len(names), "valid_images": int(bank["valid"].sum())}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"saved {len(names)} edge targets to {output}", flush=True)


if __name__ == "__main__":
    main()
