"""Build the frozen visual-teacher bank used by the hierarchical EEG stage.

The default recipe reproduces the three complementary targets used by
Brain-HIVE: SynCLR ViT-B/16, CLIP ViT-B/32, and SDXL VAE latents.  Teachers are
run sequentially so the builder also works on small GPUs.  Every array is
written transactionally and accompanied by a pinned-model manifest.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

SYNCLR_REPO = "paulgavrikov/synclr-vit-base-patch16-224"
SYNCLR_FILE = "synclr_vit_b_16.pth"
SYNCLR_REVISION = "d8438a27995c69f16819cd58bff048387127bd09"
CLIP_REPO = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
CLIP_REVISION = "1a25a446712ba5ee05982a381eed697ef9b435cf"
SDXL_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
SDXL_REVISION = "462165984030d82259a11f4367a4eed129e94a7b"

TEACHER_KEYS = ("synclr", "clip_b32", "vae_sdxl")
EXPECTED_SHAPES = {
    "synclr": (768,),
    "clip_b32": (512,),
    "vae_sdxl": (4, 16, 16),
}


@dataclass(frozen=True)
class ImageRecord:
    position: int
    image_index: int
    filename: str
    split: str
    zip_member: str
    crc32: int
    file_size: int


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pinned SynCLR + CLIP-B/32 + SDXL-VAE targets for THINGS-EEG2"
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"),
    )
    parser.add_argument(
        "--images-zip",
        type=Path,
        default=Path("data/things_eeg2_osf/image_set/training_images.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/derived/hierarchical_visual_targets"),
    )
    parser.add_argument(
        "--teachers",
        nargs="+",
        choices=TEACHER_KEYS,
        default=list(TEACHER_KEYS),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0, help="Debug only; 0 means all images")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    # Virus scanners and file indexers can briefly retain a handle on a newly
    # written file on Windows. Keep the operation atomic, but tolerate that
    # transient lock instead of losing a long feature-extraction run.
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def collect_image_records(index_path: Path, images_zip: Path) -> list[ImageRecord]:
    """Resolve the image-disjoint index against the ZIP without reading pixels."""

    indexed: dict[str, tuple[int, str]] = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            filename = row["image_file"]
            value = (int(row["image_index"]), row["split"])
            previous = indexed.setdefault(filename, value)
            if previous != value:
                raise RuntimeError(f"inconsistent index rows for {filename}: {previous} != {value}")

    with zipfile.ZipFile(images_zip) as archive:
        by_basename: dict[str, list[zipfile.ZipInfo]] = {}
        for info in archive.infolist():
            if not info.is_dir() and info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                by_basename.setdefault(Path(info.filename).name, []).append(info)

    missing = sorted(set(indexed).difference(by_basename))
    ambiguous = sorted(name for name in indexed if len(by_basename.get(name, ())) != 1)
    if missing or ambiguous:
        raise RuntimeError(
            f"image pairing failed: missing={missing[:5]}, ambiguous={ambiguous[:5]}"
        )

    ordered = sorted(indexed.items(), key=lambda item: item[1][0])
    records: list[ImageRecord] = []
    for position, (filename, (image_index, split)) in enumerate(ordered):
        info = by_basename[filename][0]
        records.append(
            ImageRecord(
                position=position,
                image_index=image_index,
                filename=filename,
                split=split,
                zip_member=info.filename,
                crc32=info.CRC,
                file_size=info.file_size,
            )
        )
    if len({record.image_index for record in records}) != len(records):
        raise RuntimeError("duplicate image_index values in the visual target index")
    return records


def records_fingerprint(records: Iterable[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        line = (
            f"{record.position}\t{record.image_index}\t{record.filename}\t{record.split}\t"
            f"{record.zip_member}\t{record.crc32}\t{record.file_size}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _write_index(path: Path, records: Sequence[ImageRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    os.replace(temporary, path)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


class Teacher:
    key: str

    def __init__(self, device: torch.device, seed: int):
        self.device = device
        self.seed = seed
        self.model: torch.nn.Module

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        raise NotImplementedError

    def encode(self, pixels: torch.Tensor, image_indices: Sequence[int]) -> torch.Tensor:
        raise NotImplementedError

    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        self.model.cpu()
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SynCLRTeacher(Teacher):
    key = "synclr"

    def __init__(self, device: torch.device, seed: int):
        super().__init__(device, seed)
        from huggingface_hub import hf_hub_download
        from timm.models.vision_transformer import VisionTransformer
        from torchvision import transforms

        checkpoint = hf_hub_download(SYNCLR_REPO, filename=SYNCLR_FILE, revision=SYNCLR_REVISION)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "model" not in payload:
            raise KeyError(f"SynCLR checkpoint has no 'model' state: {checkpoint}")
        state = {
            key.removeprefix("module.visual."): value for key, value in payload["model"].items()
        }
        self.model = VisionTransformer(
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            num_classes=0,
        )
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        self.checkpoint_sha256 = _sha256(Path(checkpoint))

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(image) for image in images])

    def encode(self, pixels: torch.Tensor, image_indices: Sequence[int]) -> torch.Tensor:
        del image_indices
        with torch.autocast("cuda", torch.float16, enabled=self.device.type == "cuda"):
            return self.model.forward_features(pixels)[:, 0].float()

    def metadata(self) -> dict[str, Any]:
        return {
            "role": "high-level visual invariance",
            "repository": SYNCLR_REPO,
            "revision": SYNCLR_REVISION,
            "checkpoint_file": SYNCLR_FILE,
            "checkpoint_sha256": self.checkpoint_sha256,
            "feature": "raw CLS token from forward_features",
            "preprocessing": "resize-short-side 224, center-crop 224, OpenCLIP normalization",
            "normalized": False,
        }


class CLIPB32Teacher(Teacher):
    key = "clip_b32"

    def __init__(self, device: torch.device, seed: int):
        super().__init__(device, seed)
        from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

        self.processor = CLIPImageProcessor.from_pretrained(CLIP_REPO, revision=CLIP_REVISION)
        self.model = (
            CLIPVisionModelWithProjection.from_pretrained(CLIP_REPO, revision=CLIP_REVISION)
            .eval()
            .to(device)
        )
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or CLIP_REVISION

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return self.processor(images=list(images), return_tensors="pt")["pixel_values"]

    def encode(self, pixels: torch.Tensor, image_indices: Sequence[int]) -> torch.Tensor:
        del image_indices
        with torch.autocast("cuda", torch.float16, enabled=self.device.type == "cuda"):
            return self.model(pixel_values=pixels).image_embeds.float()

    def metadata(self) -> dict[str, Any]:
        return {
            "role": "image-language semantic alignment",
            "repository": CLIP_REPO,
            "revision": self.resolved_revision,
            "feature": "projected image_embeds from CLIPVisionModelWithProjection",
            "preprocessing": "checkpoint CLIPImageProcessor",
            "normalized": False,
        }


class SDXLVAETeacher(Teacher):
    key = "vae_sdxl"

    def __init__(self, device: torch.device, seed: int):
        super().__init__(device, seed)
        from diffusers import AutoencoderKL
        from torchvision import transforms

        self.model = AutoencoderKL.from_pretrained(
            SDXL_REPO,
            subfolder="vae",
            revision=SDXL_REVISION,
            torch_dtype=torch.float32,
        ).eval().to(device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(128),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.scaling_factor = float(self.model.config.scaling_factor)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None) or SDXL_REVISION

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        return torch.stack([self.transform(image) for image in images])

    def encode(self, pixels: torch.Tensor, image_indices: Sequence[int]) -> torch.Tensor:
        posterior = self.model.encode(pixels.float()).latent_dist
        # Brain-HIVE samples the posterior.  Per-image generators keep the same
        # distribution while making the target independent of batching/resume.
        noises = []
        sample_shape = posterior.mean.shape[1:]
        for image_index in image_indices:
            generator = torch.Generator(device="cpu").manual_seed(self.seed + image_index)
            noises.append(torch.randn(sample_shape, generator=generator, dtype=torch.float32))
        noise = torch.stack(noises).to(self.device)
        return (posterior.mean.float() + posterior.std.float() * noise) * self.scaling_factor

    def metadata(self) -> dict[str, Any]:
        return {
            "role": "low-level SDXL latent",
            "repository": SDXL_REPO,
            "subfolder": "vae",
            "revision": self.resolved_revision,
            "feature": "sampled posterior latent multiplied by VAE scaling_factor",
            "posterior_sampling": "deterministic per image: seed + image_index",
            "scaling_factor": self.scaling_factor,
            "preprocessing": "resize-short-side 128, center-crop 128, RGB mapped to [-1,1]",
            "normalized": False,
        }


def build_teacher(key: str, device: torch.device, seed: int) -> Teacher:
    classes = {
        "synclr": SynCLRTeacher,
        "clip_b32": CLIPB32Teacher,
        "vae_sdxl": SDXLVAETeacher,
    }
    return classes[key](device, seed)


def _load_images(archive: zipfile.ZipFile, records: Sequence[ImageRecord]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for record in records:
        with Image.open(io.BytesIO(archive.read(record.zip_member))) as image:
            images.append(image.convert("RGB"))
    return images


def _prepare_memmap(
    output: Path,
    key: str,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    fingerprint: str,
    overwrite: bool,
) -> tuple[np.memmap, int, Path, Path]:
    final_path = output / f"{key}.{np.dtype(dtype).name}.npy"
    partial_path = output / f"{key}.{np.dtype(dtype).name}.partial.npy"
    progress_path = output / f"{key}.progress.json"
    metadata_path = output / f"{key}.json"

    if final_path.exists():
        if not overwrite and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("source_fingerprint") == fingerprint
                and tuple(metadata.get("shape", ())) == shape
            ):
                raise FileExistsError(f"COMPLETE:{final_path}")
        if not overwrite:
            raise FileExistsError(f"existing target is incompatible: {final_path}")
        final_path.unlink()
        metadata_path.unlink(missing_ok=True)

    if partial_path.exists() and progress_path.exists() and not overwrite:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("source_fingerprint") != fingerprint
            or tuple(progress.get("shape", ())) != shape
        ):
            raise RuntimeError(f"incompatible interrupted target: {partial_path}")
        array = np.lib.format.open_memmap(partial_path, mode="r+", dtype=dtype, shape=shape)
        return array, int(progress["completed"]), partial_path, progress_path

    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    array = np.lib.format.open_memmap(partial_path, mode="w+", dtype=dtype, shape=shape)
    _atomic_json(
        progress_path,
        {"teacher": key, "source_fingerprint": fingerprint, "shape": shape, "completed": 0},
    )
    return array, 0, partial_path, progress_path


def build_one_teacher(
    *,
    key: str,
    records: Sequence[ImageRecord],
    images_zip: Path,
    output: Path,
    fingerprint: str,
    batch_size: int,
    storage_dtype: str,
    device: torch.device,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    expected = (len(records), *EXPECTED_SHAPES[key])
    dtype = np.dtype(storage_dtype)
    try:
        array, start, partial_path, progress_path = _prepare_memmap(
            output, key, expected, dtype, fingerprint, overwrite
        )
    except FileExistsError as error:
        if str(error).startswith("COMPLETE:"):
            print(f"{key}: verified complete target exists; skipping", flush=True)
            return json.loads((output / f"{key}.json").read_text(encoding="utf-8"))
        raise

    print(f"{key}: loading pinned teacher", flush=True)
    teacher = build_teacher(key, device, seed)
    try:
        with zipfile.ZipFile(images_zip) as archive:
            for offset in range(start, len(records), batch_size):
                batch_records = records[offset : offset + batch_size]
                images = _load_images(archive, batch_records)
                pixels = teacher.preprocess(images).to(device, non_blocking=True)
                with torch.inference_mode():
                    features = teacher.encode(
                        pixels, [record.image_index for record in batch_records]
                    )
                features = features.detach().float().cpu()
                if tuple(features.shape[1:]) != EXPECTED_SHAPES[key]:
                    raise RuntimeError(
                        f"{key} produced {tuple(features.shape[1:])}, "
                        f"expected {EXPECTED_SHAPES[key]}"
                    )
                if not torch.isfinite(features).all():
                    raise RuntimeError(f"{key} produced NaN or infinity")
                end = offset + len(batch_records)
                array[offset:end] = features.numpy().astype(dtype, copy=False)
                array.flush()
                _atomic_json(
                    progress_path,
                    {
                        "teacher": key,
                        "source_fingerprint": fingerprint,
                        "shape": expected,
                        "completed": end,
                    },
                )
                print(f"{key}: {end}/{len(records)}", flush=True)

        sample_std = float(np.asarray(array, dtype=np.float32).std())
        if not np.isfinite(sample_std) or sample_std < 1e-6:
            raise RuntimeError(f"{key} target bank collapsed (std={sample_std})")
        teacher_metadata = teacher.metadata()
    finally:
        teacher.close()

    array.flush()
    del array
    final_path = output / f"{key}.{dtype.name}.npy"
    os.replace(partial_path, final_path)
    progress_path.unlink(missing_ok=True)
    metadata = {
        "schema_version": 1,
        "teacher": key,
        "source_fingerprint": fingerprint,
        "shape": expected,
        "dtype": dtype.name,
        "array": final_path.name,
        "array_sha256": _sha256(final_path),
        "seed": seed,
        **teacher_metadata,
    }
    _atomic_json(output / f"{key}.json", metadata)
    return metadata


def split_counts(records: Sequence[ImageRecord]) -> dict[str, int]:
    return {
        split: sum(record.split == split for record in records)
        for split in ("train", "val", "test")
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    records = collect_image_records(args.index, args.images_zip)
    if args.max_images:
        records = records[: args.max_images]
    if not records:
        raise RuntimeError("visual target index is empty")

    fingerprint = records_fingerprint(records)
    _write_index(args.output / "index.csv", records)
    index_sha256 = _sha256(args.output / "index.csv")
    device = _device(args.device)
    teacher_manifests = {}
    for key in dict.fromkeys(args.teachers):
        teacher_manifests[key] = build_one_teacher(
            key=key,
            records=records,
            images_zip=args.images_zip,
            output=args.output,
            fingerprint=fingerprint,
            batch_size=args.batch_size,
            storage_dtype=args.storage_dtype,
            device=device,
            seed=args.seed,
            overwrite=args.overwrite,
        )

    manifest = {
        "schema_version": 1,
        "recipe": "Brain-HIVE-compatible SynCLR-B16 + CLIP-B32 + SDXL-VAE visual targets",
        "images": len(records),
        "complete_things_eeg2_training_partition": len(records) == 16_540,
        "split_counts": split_counts(records),
        "source_fingerprint": fingerprint,
        "index": "index.csv",
        "index_sha256": index_sha256,
        "images_zip": str(args.images_zip),
        "teachers": teacher_manifests,
        "training_policy": {
            "train": "optimization only",
            "val": "checkpoint selection only",
            "test": "untouched final evaluation only",
        },
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(
        f"saved verified hierarchical targets for {len(records)} images to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
