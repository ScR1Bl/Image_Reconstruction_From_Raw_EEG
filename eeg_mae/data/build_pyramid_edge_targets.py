from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

PYRAMID_KEYS = (
    "shape16",
    "distance16",
    "semantic32",
    "perceptual64",
    "crisp64",
    "confidence64",
    "uncertainty64",
    "orientation64",
)


def image_tensor(image: Image.Image, size: int) -> torch.Tensor:
    array = (
        np.asarray(
            image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ).copy()
        / 255.0
    )
    return torch.from_numpy(array).permute(2, 0, 1)


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build uncertainty-aware multi-level edge targets")
    parser.add_argument("--edge-bank", default="data/derived/edge_targets_64_v2.pt")
    parser.add_argument("--rich-bank", default="data/derived/rich_visual_targets_32.pt")
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--pidinet-root", default="data/external/pidinet")
    parser.add_argument(
        "--pidinet-checkpoint", default="data/external/pidinet/trained_models/table5_pidinet.pth"
    )
    parser.add_argument("--output", default="data/derived/pyramid_edge_targets_v1.pt")
    parser.add_argument("--teacher-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_pidinet(root: Path, checkpoint: Path, device: torch.device):
    sys.path.insert(0, str(root.resolve()))
    try:
        from models.pidinet import pidinet
    finally:
        sys.path.pop(0)
    model = pidinet(SimpleNamespace(config="carv4", sa=True, dil=True))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


@torch.no_grad()
def perceptual_edges(model, rgb: torch.Tensor, output_size: int = 64) -> torch.Tensor:
    mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    predictions = []
    for scale in (0.75, 1.0, 1.25):
        value = F.interpolate(rgb, scale_factor=scale, mode="bilinear", align_corners=False)
        predictions.append(
            F.interpolate(
                model((value - mean) / std)[-1],
                (output_size, output_size),
                mode="bilinear",
                align_corners=False,
            )
        )
    prediction = torch.stack(predictions).mean(0)
    # PiDiNet probabilities have image-dependent calibration. Robust contrast
    # normalization retains ranking while avoiding a fixed brittle threshold.
    flat = prediction.flatten(1)
    low = torch.quantile(flat, 0.50, dim=1)[:, None, None, None]
    high = torch.quantile(flat, 0.98, dim=1)[:, None, None, None]
    return ((prediction - low) / (high - low).clamp_min(1e-5)).clamp(0, 1)


def depth_boundaries(depth: torch.Tensor, size: int = 64) -> torch.Tensor:
    value = F.interpolate(depth.float(), (size, size), mode="bicubic", align_corners=False)
    dx = value[..., :, 1:] - value[..., :, :-1]
    dy = value[..., 1:, :] - value[..., :-1, :]
    dx = F.pad(dx, (0, 1))
    dy = F.pad(dy, (0, 0, 0, 1))
    gradient = (dx.square() + dy.square() + 1e-8).sqrt()
    scale = torch.quantile(gradient.flatten(1), 0.95, dim=1)[:, None, None, None]
    return (gradient / scale.clamp_min(1e-5)).clamp(0, 1)


def orientation_from_distance(distance: torch.Tensor) -> torch.Tensor:
    dx = F.pad(distance[..., :, 1:] - distance[..., :, :-1], (0, 1))
    dy = F.pad(distance[..., 1:, :] - distance[..., :-1, :], (0, 0, 0, 1))
    norm = (dx.square() + dy.square() + 1e-6).sqrt()
    return torch.cat((dx / norm, dy / norm), 1)


def skeleton_batch(value: torch.Tensor) -> torch.Tensor:
    """Return a batched morphological skeleton without a scikit-image dependency."""
    current = (value >= 0.42).float()
    skeleton = torch.zeros_like(current)
    # Repeated erosion terminates within half the image width. Subtracting the
    # opening at every scale leaves the medial-axis pixels.
    for _ in range(max(value.shape[-2:])):
        eroded = -F.max_pool2d(-current, 3, stride=1, padding=1)
        opened = F.max_pool2d(eroded, 3, stride=1, padding=1)
        skeleton = torch.maximum(skeleton, (current - opened).clamp_min(0))
        current = eroded
        if not bool(current.any()):
            break
    return skeleton


def derive_targets(
    semantic: torch.Tensor, distance: torch.Tensor, perceptual: torch.Tensor, geometry: torch.Tensor
) -> dict[str, torch.Tensor]:
    semantic, distance = semantic.float(), distance.float()
    foreground = (distance >= 0.5).float()
    shape16 = F.adaptive_avg_pool2d(foreground, (16, 16))
    distance16 = F.interpolate(distance, (16, 16), mode="bilinear", align_corners=False)
    semantic32 = torch.maximum(
        F.adaptive_avg_pool2d(semantic, (32, 32)), 0.80 * F.adaptive_max_pool2d(semantic, (32, 32))
    ).clamp(0, 1)

    # Compare cues in a small tolerance band because independent teachers do
    # not place a one-pixel contour at exactly the same coordinate.
    cues = torch.cat((semantic, perceptual, geometry), 1)
    supported = F.max_pool2d(cues, 3, stride=1, padding=1)
    mean = supported.mean(1, keepdim=True)
    disagreement = supported.std(1, keepdim=True, unbiased=False)
    confidence = (mean * (1 - disagreement)).clamp(0, 1)
    uncertainty = (mean * disagreement * 2).clamp(0, 1)
    fused = (0.50 * semantic + 0.35 * perceptual + 0.15 * geometry).clamp(0, 1)
    crisp = skeleton_batch(fused).to(fused.device)
    orientation = orientation_from_distance(distance)
    return {
        "shape16": shape16,
        "distance16": distance16,
        "semantic32": semantic32,
        "perceptual64": perceptual,
        "crisp64": crisp,
        "confidence64": confidence,
        "uncertainty64": uncertainty,
        "orientation64": orientation,
    }


def quality(targets: dict[str, torch.Tensor], original_valid: torch.Tensor) -> torch.Tensor:
    shape_fraction = targets["shape16"].mean((1, 2, 3))
    crisp_density = targets["crisp64"].mean((1, 2, 3))
    confidence = targets["confidence64"].mean((1, 2, 3))
    return (
        original_valid
        & (shape_fraction > 0.01)
        & (shape_fraction < 0.92)
        & (crisp_density > 0.002)
        & (crisp_density < 0.35)
        & (confidence > 0.01)
    )


def main() -> None:
    args = arguments()
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    edge = torch.load(args.edge_bank, map_location="cpu", weights_only=False)
    rich = torch.load(args.rich_bank, map_location="cpu", weights_only=False)
    if edge["filenames"] != rich["filenames"]:
        raise RuntimeError("edge and rich target banks use different image order")
    names = edge["filenames"][: args.max_images or None]
    model = load_pidinet(Path(args.pidinet_root), Path(args.pidinet_checkpoint), device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output.with_suffix(output.suffix + ".partial")
    metadata = {
        "version": 1,
        "teacher": "PiDiNet table5 + BiRefNet contour + Depth-Anything-V2",
        "pidinet_checkpoint": args.pidinet_checkpoint,
        "edge_bank": args.edge_bank,
        "rich_bank": args.rich_bank,
        "teacher_size": args.teacher_size,
        "complete": not bool(args.max_images),
        "definitions": {
            "shape16": "foreground occupancy derived from signed distance",
            "distance16": "coarse signed distance",
            "semantic32": "BiRefNet object boundary",
            "perceptual64": "multi-scale PiDiNet edge probability",
            "crisp64": "one-pixel skeleton of semantic/perceptual/depth consensus",
            "confidence64": "three-teacher agreement-weighted edge probability",
            "uncertainty64": "localized teacher disagreement",
            "orientation64": "normal field from signed-distance gradient",
        },
    }
    accumulated = {key: [] for key in PYRAMID_KEYS}
    valid_all = []
    processed = 0
    if partial_path.exists():
        partial = torch.load(partial_path, map_location="cpu", weights_only=False)
        if (
            partial["metadata"] != metadata
            or partial["filenames"] != names[: len(partial["filenames"])]
        ):
            raise RuntimeError(f"incompatible partial bank: {partial_path}")
        processed = len(partial["filenames"])
        for key in PYRAMID_KEYS:
            accumulated[key].append(partial[key])
        valid_all.append(partial["valid"])

    with zipfile.ZipFile(args.images_zip) as archive:
        members = {
            Path(member).name: member
            for member in archive.namelist()
            if member.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        for start in range(processed, len(names), args.batch_size):
            batch_names = names[start : start + args.batch_size]
            images = []
            for name in batch_names:
                with Image.open(io.BytesIO(archive.read(members[name]))) as image:
                    images.append(image.convert("RGB"))
            rgb = torch.stack([image_tensor(image, args.teacher_size) for image in images]).to(
                device
            )
            with (
                torch.inference_mode(),
                torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"),
            ):
                perceptual = perceptual_edges(model, rgb)
            stop = start + len(images)
            semantic = edge["semantic"][start:stop].float().to(device)
            distance = edge["distance"][start:stop].float().to(device)
            geometry = depth_boundaries(rich["depth"][start:stop].to(device))
            targets = derive_targets(semantic, distance, perceptual.float(), geometry)
            valid = quality(targets, edge["valid"][start:stop].to(device))
            for key in PYRAMID_KEYS:
                accumulated[key].append(targets[key].cpu().half())
            valid_all.append(valid.cpu())
            print(
                f"pyramid targets {stop}/{len(names)} valid={int(valid.sum())}/{len(valid)}",
                flush=True,
            )
            if (
                args.checkpoint_every
                and ((start // args.batch_size) + 1) % args.checkpoint_every == 0
            ):
                atomic_save(
                    {
                        "filenames": names[:stop],
                        "metadata": metadata,
                        **{key: torch.cat(value) for key, value in accumulated.items()},
                        "valid": torch.cat(valid_all),
                    },
                    partial_path,
                )

    bank = {
        "filenames": names,
        "metadata": metadata,
        **{key: torch.cat(value) for key, value in accumulated.items()},
        "valid": torch.cat(valid_all),
    }
    atomic_save(bank, output)
    if partial_path.exists():
        partial_path.unlink()
    output.with_suffix(".json").write_text(
        json.dumps(
            {**metadata, "images": len(names), "valid_images": int(bank["valid"].sum())}, indent=2
        ),
        encoding="utf-8",
    )
    print(f"saved pyramid bank to {output}", flush=True)


if __name__ == "__main__":
    main()
