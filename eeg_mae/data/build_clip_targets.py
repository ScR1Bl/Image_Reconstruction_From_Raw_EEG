"""Build CLIP-image target banks (Step 2), patterned after build_visual_targets.

Teacher: laion/CLIP-ViT-L-14-laion2B-s32B-b82K (OpenCLIP ViT-L/14, LAION-2B),
projected CLS embedding ``image_embeds`` (768-d) via CLIPVisionModelWithProjection.
Embeddings are stored RAW (unnormalized) on purpose: their norm distribution is a
Step-3 input (generation quality), and retrieval normalizes inside the metric.

Train bank rows follow the sorted-unique ``image_file`` order of the holdout
index (same contract as visual_targets_dinov2s_192.pt — training_target_indices
works unchanged). The official 200-condition test bank follows the numeric
00001..00200 folder order via ordered_test_members.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image

TEACHER = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
FAMILY_TEACHERS = {
    "clip": TEACHER,
    "siglip2": "google/siglip2-so400m-patch14-384",
    "eva02": "EVA02-L-14::merged2b_s4b_b131k",  # open_clip: model::pretrained
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build image-embedding target banks")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--output", default="data/derived/clip_targets_vitl14_laion2b.pt")
    parser.add_argument("--family", choices=tuple(FAMILY_TEACHERS), default="clip")
    parser.add_argument("--teacher", default=None, help="default zalezny od --family")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument(
        "--official-test",
        action="store_true",
        help="tryb testowy: --images-zip to test_images.zip, porzadek folderow 00001..00200",
    )
    args = parser.parse_args()
    if args.teacher is None:
        args.teacher = FAMILY_TEACHERS[args.family]
    if args.family != "clip" and args.output == parser.get_default("output"):
        parser.error(
            "dla --family innego niz clip podaj jawnie --output (ochrona przed nadpisaniem)"
        )
    return args


def load_embedder(family: str, teacher: str, device):
    """Return (embed_fn(list[PIL]) -> raw float tensor [B, dim], resolved_teacher)."""

    if family == "clip":
        from transformers import AutoImageProcessor, CLIPConfig, CLIPVisionModelWithProjection

        processor = AutoImageProcessor.from_pretrained(teacher)
        projection_dim = CLIPConfig.from_pretrained(teacher).projection_dim
        model = (
            CLIPVisionModelWithProjection.from_pretrained(teacher, projection_dim=projection_dim)
            .to(device)
            .eval()
        )

        def embed(images):
            inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            return model(pixel_values=inputs).image_embeds.float()

        return embed, teacher

    if family == "siglip2":
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(teacher)
        model = AutoModel.from_pretrained(teacher).to(device).eval()

        def embed(images):
            inputs = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            features = model.get_image_features(pixel_values=inputs)
            # transformers v5 zwraca BaseModelOutputWithPooling zamiast tensora
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            return features.float()

        return embed, teacher

    if family == "eva02":
        import open_clip

        name, pretrained = teacher.split("::")
        model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        model = model.to(device).eval()

        def embed(images):
            inputs = torch.stack([preprocess(image) for image in images]).to(device)
            return model.encode_image(inputs).float()

        return embed, teacher

    raise ValueError(f"unknown family: {family}")


def unique_index_images(index_path: Path) -> list[str]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        names = {row["image_file"] for row in csv.DictReader(handle)}
    if "" in names:
        raise ValueError("index contains an empty image_file")
    return sorted(names)


def main() -> None:
    args = arguments()
    zip_path, output = Path(args.images_zip), Path(args.output)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    embed, teacher_name = load_embedder(args.family, args.teacher, device)

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
            ambiguous = {
                name: by_basename[name] for name in names if len(by_basename.get(name, [])) != 1
            }
            if missing or ambiguous:
                raise RuntimeError(
                    f"image pairing failed: missing={missing[:5]}, ambiguous={list(ambiguous)[:5]}"
                )
            members = [by_basename[name][0] for name in names]
        if args.max_images:
            members, names = members[: args.max_images], names[: args.max_images]

        chunks = []
        for start in range(0, len(members), args.batch_size):
            batch = members[start : start + args.batch_size]
            images = []
            for member in batch:
                with Image.open(io.BytesIO(archive.read(member))) as image:
                    images.append(image.convert("RGB"))
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
                ),
            ):
                embeds = embed(images)
            chunks.append(embeds.cpu().half())
            print(
                f"clip targets {min(start + len(batch), len(members))}/{len(members)}", flush=True
            )

    clip_global = torch.cat(chunks)
    embed_dim = clip_global.shape[1]
    if len(set(names)) != len(names) or clip_global.shape != (len(names), embed_dim):
        raise RuntimeError("clip bank integrity check failed")
    norms = clip_global.float().norm(dim=-1)
    bank = {
        "filenames": names,
        "clip_global": clip_global,
        "metadata": {
            "teacher": teacher_name,
            "family": args.family,
            "embed_dim": embed_dim,
            "normalized": False,
            "official_test": bool(args.official_test),
            "complete": not args.max_images,
            "source_zip": str(zip_path),
            "norm_mean": float(norms.mean()),
            "norm_std": float(norms.std()),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output)
    output.with_suffix(".json").write_text(
        json.dumps({**bank["metadata"], "images": len(names)}, indent=2)
    )
    print(
        f"saved {len(names)} clip targets to {output} "
        f"(norm {norms.mean():.2f} +/- {norms.std():.2f})"
    )


if __name__ == "__main__":
    main()
