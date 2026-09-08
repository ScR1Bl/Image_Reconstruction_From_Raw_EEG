"""Build the VAE latent bank for PixArt-alpha adapter training (Step 3).

Precomputes the VAE posterior MOMENTS (mean + std, fp16, [N, 4+4, 64, 64]) for
every training image, so the VAE never occupies VRAM during adapter training and
each epoch can still SAMPLE fresh latents from the posterior. Multiply sampled
latents by the VAE scaling_factor (0.18215) before feeding the DiT — recorded in
metadata, applied at training time, NOT baked into the bank.

Preprocessing: resize shorter side to 512 (bicubic) + center crop 512x512,
RGB in [-1, 1]. Row order = sorted unique image_file from the holdout index
(same contract as every other bank; training_target_indices works unchanged).
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image

from .build_clip_targets import unique_index_images

REPO = "PixArt-alpha/PixArt-XL-2-512x512"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VAE latent bank for PixArt-alpha")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--output", default="data/derived/vae_latents_pixart512.pt")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--official-test",
        action="store_true",
        help="tryb testowy: test_images.zip, porzadek folderow 00001..00200",
    )
    return parser.parse_args()


def preprocess(image: Image.Image, resolution: int) -> torch.Tensor:
    width, height = image.size
    scale = resolution / min(width, height)
    image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (image.width - resolution) // 2
    top = (image.height - resolution) // 2
    image = image.crop((left, top, left + resolution, top + resolution))
    tensor = torch.from_numpy(__import__("numpy").asarray(image)).float() / 127.5 - 1.0
    return tensor.permute(2, 0, 1)


def main() -> None:
    from diffusers import AutoencoderKL

    args = arguments()
    zip_path, output = Path(args.images_zip), Path(args.output)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    vae = (
        AutoencoderKL.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.bfloat16)
        .to(device)
        .eval()
    )

    with zipfile.ZipFile(zip_path) as archive:
        if args.official_test:
            from .build_official_test_targets import ordered_test_members

            members = ordered_test_members(archive)
            names = [Path(member).name for member in members]
        else:
            names = unique_index_images(Path(args.index))
            by_basename: dict[str, list[str]] = {}
            for member in archive.namelist():
                if member.lower().endswith((".jpg", ".jpeg", ".png")):
                    by_basename.setdefault(Path(member).name, []).append(member)
            missing = [name for name in names if name not in by_basename]
            if missing or any(len(by_basename[name]) != 1 for name in names):
                raise RuntimeError(f"image pairing failed: missing={missing[:5]}")
            members = [by_basename[name][0] for name in names]
        if args.max_images:
            members, names = members[: args.max_images], names[: args.max_images]

        chunks = []
        for start in range(0, len(members), args.batch_size):
            batch_members = members[start : start + args.batch_size]
            batch = torch.stack(
                [
                    preprocess(
                        Image.open(io.BytesIO(archive.read(member))).convert("RGB"),
                        args.resolution,
                    )
                    for member in batch_members
                ]
            ).to(device, torch.bfloat16)
            with torch.no_grad():
                dist = vae.encode(batch).latent_dist
                moments = torch.cat((dist.mean, dist.std), dim=1).float()
            chunks.append(moments.cpu().half())
            print(
                f"vae latents {min(start + len(batch_members), len(members))}/{len(members)}",
                flush=True,
            )

    moments = torch.cat(chunks)
    if moments.shape[0] != len(names):
        raise RuntimeError("latent bank integrity check failed")
    bank = {
        "filenames": names,
        "latent_moments": moments,  # [N, 8, 64, 64]: kanaly 0-3 mean, 4-7 std
        "metadata": {
            "vae": f"{REPO}/vae",
            "resolution": args.resolution,
            "scaling_factor": vae.config.scaling_factor,
            "layout": "mean(0:4) + std(4:8); sampluj: mean + std*randn, potem *scaling_factor",
            "preprocessing": "resize krotszy bok->512 bicubic, center crop 512, RGB [-1,1]",
            "official_test": bool(args.official_test),
            "complete": not args.max_images,
            "source_zip": str(zip_path),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    output.with_suffix(".json").write_text(
        json.dumps({**bank["metadata"], "images": len(names)}, indent=2)
    )
    print(f"saved {len(names)} latent moments to {output}")


if __name__ == "__main__":
    main()
