"""Score generated images against the fixed TRUE-image CLIP gallery."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from ..data.build_category_prior import category, training_labels
from ..data.build_category_text import category_prompt, load_concepts
from ..data.dit_ten_subjects import OUTPUT, legacy_rows, load_archive_features

TEACHER = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
REVISION = "1627032197142fbe2a7cfec626f4ced3ae60d07a"


def gallery_metrics(prediction, gallery, correct_indices):
    scores = F.normalize(prediction.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T
    correct = scores[torch.arange(len(scores)), torch.tensor(correct_indices)]
    greater = (scores > correct[:, None]).sum(1)
    equal = (scores == correct[:, None]).sum(1).clamp_min(1)
    return {
        "cosine_to_true_image": float(correct.mean()),
        "top1_true_gallery": float(((1 - greater).clamp_min(0) / equal).clamp_max(1).mean()),
        "top5_true_gallery": float(((5 - greater).clamp_min(0) / equal).clamp_max(1).mean()),
        "gallery_size": len(gallery),
        "images": len(prediction),
        "ranks_optimistic": (greater + 1).tolist(),
        "cosines": correct.tolist(),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation", type=Path, default=OUTPUT / "comparison_generation")
    parser.add_argument("--conditions", type=Path, default=OUTPUT / "comparison_conditions.pt")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="optional classifier predictions JSON for semantic adherence scoring",
    )
    parser.add_argument(
        "--concepts",
        type=Path,
        help="optional shared THINGS sense definitions for the CLIP text gallery",
    )
    args = parser.parse_args()
    torch.set_num_threads(2)
    from transformers import CLIPConfig, CLIPImageProcessor, CLIPVisionModelWithProjection

    processor = CLIPImageProcessor.from_pretrained(TEACHER, revision=REVISION)
    config = CLIPConfig.from_pretrained(TEACHER, revision=REVISION)
    model = (
        CLIPVisionModelWithProjection.from_pretrained(
            TEACHER,
            revision=REVISION,
            projection_dim=config.projection_dim,
            torch_dtype=torch.float16,
        )
        .cuda()
        .eval()
    )
    bundle = torch.load(args.conditions, map_location="cpu", weights_only=False)
    generation = json.loads((args.generation / "report.json").read_text())
    count = generation["count"]
    rows = legacy_rows()["val"]
    positions = {r["filename"]: i for i, r in enumerate(rows)}
    indices = [positions[r["filename"]] for r in bundle["rows"][:count]]
    gallery = load_archive_features()["y_val"].float()
    records = bundle["metadata"].get("predictions")
    if args.predictions:
        by_name = {r["filename"]: r for r in json.loads(args.predictions.read_text())}
        records = [by_name[r["filename"]] for r in bundle["rows"][:count]]
    text_features = None
    if records is not None:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

        names, _ = training_labels(legacy_rows()["train"])
        concepts = load_concepts(args.concepts) if args.concepts else None
        lookup = {name: i for i, name in enumerate(names)}
        tokenizer = CLIPTokenizerFast.from_pretrained(TEACHER, revision=REVISION)
        text_model = (
            CLIPTextModelWithProjection.from_pretrained(
                TEACHER,
                revision=REVISION,
                projection_dim=config.projection_dim,
                torch_dtype=torch.float16,
            )
            .cuda()
            .eval()
        )
        text_features = []
        for start in range(0, len(names), 64):
            tokens = tokenizer(
                [category_prompt(n, concepts) for n in names[start : start + 64]],
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            text_features.append(text_model(**tokens).text_embeds.float().cpu())
        text_features = torch.cat(text_features)
        predicted_ids = [lookup[r["predicted_category"]] for r in records[:count]]
        true_ids = [lookup[category(r)] for r in bundle["rows"][:count]]
        del text_model

    def embed(images):
        outputs = []
        for start in range(0, len(images), 8):
            pixels = (
                processor(images=images[start : start + 8], return_tensors="pt")["pixel_values"]
                .cuda()
                .half()
            )
            outputs.append(model(pixel_values=pixels).image_embeds.float().cpu())
        return torch.cat(outputs)

    originals = []
    with zipfile.ZipFile("data/things_eeg2_osf/image_set/training_images.zip") as z:
        for row in bundle["rows"][:count]:
            originals.append(Image.open(io.BytesIO(z.read(row["zip_member"]))).convert("RGB"))
    original_features = embed(originals)
    alignment = F.cosine_similarity(original_features, gallery[indices]).tolist()
    if min(alignment) < 0.995:
        raise ValueError(f"CLIP scorer and true feature bank disagree: {alignment}")
    metrics = {}
    for variant in generation["variants"]:
        features = embed(
            [
                Image.open(args.generation / variant / f"{i:03d}.png").convert("RGB")
                for i in range(count)
            ]
        )
        metrics[variant] = gallery_metrics(features, gallery, indices)
        if text_features is not None:
            metrics[variant]["category_text_gallery"] = {
                "agreement_with_predicted_category": gallery_metrics(
                    features, text_features, predicted_ids
                ),
                "agreement_with_stimulus_category": gallery_metrics(
                    features, text_features, true_ids
                ),
                "interpretation": "CLIP text semantic proxy, not a human object-realism or geometry assessment",
            }
    report = {
        "teacher": TEACHER,
        "revision": REVISION,
        "text_gallery_concepts": str(args.concepts) if args.concepts else "bare category names",
        "true_feature_alignment_cosines": alignment,
        "metrics": metrics,
        "test_evaluated": False,
        "sample_policy": bundle["metadata"].get(
            "sample_policy",
            "first validation categories, qualitative pilot; not a representative benchmark",
        ),
        "evaluation_reference": "true original image embeddings, never predicted or shuffled conditions",
        "not_independent_metric": "CLIP is also the adapter target; geometry needs separate validation",
    }
    (args.generation / "scores.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
