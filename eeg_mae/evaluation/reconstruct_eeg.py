"""Run the frozen ten-person category-assisted pipeline on preprocessed EEG.

No stimulus image, category label, or validation gallery is accepted as input.
Output photographs are generated using a retrieved TRAINING reference prior.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..data.build_category_prior import training_labels
from ..data.build_text_comparison import text_for_predictions
from ..data.dit_ten_subjects import ARTIFACTS, OUTPUT
from ..lightning.category_classifier import CategoryClassifier, select_reference
from ..lightning.common import sha256_of_file
from ..models.hierarchical_semantic import load_hierarchical_semantic_encoder
from ..models.semantic_encoder import SubjectAdaptiveSemanticEncoder


def validate_eeg(value):
    value = np.asarray(value)
    if value.ndim == 4:
        value = value[None]
    if value.ndim != 5 or value.shape[1:] != (10, 4, 17, 100) or len(value) == 0:
        raise ValueError("expected [N,10,4,17,100] or [10,4,17,100]; subject order sub-01..10")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError("EEG must contain finite preprocessed floating point samples")
    return value.astype(np.float32, copy=False)


@torch.no_grad()
def predict(eeg, bank, classifier_path, clip_checkpoint, batch_size=8):
    names, labels = training_labels(bank["rows"])
    if names != bank["category_names"] or not torch.equal(labels, bank["labels"]):
        raise ValueError("reference bank category order mismatch")
    semantic_path = Path(bank["metadata"]["checkpoint"])
    if sha256_of_file(semantic_path) != bank["metadata"]["checkpoint_sha256"]:
        raise ValueError("hierarchical encoder changed since reference bank creation")
    hierarchical, _, _ = load_hierarchical_semantic_encoder(semantic_path, "cuda")
    hierarchical.eval().requires_grad_(False)
    payload = torch.load(clip_checkpoint, weights_only=False, mmap=True, map_location="cpu")
    clip = SubjectAdaptiveSemanticEncoder(output_dim=768).cuda().eval().requires_grad_(False)
    clip.load_state_dict(
        {
            k.removeprefix("trunk."): v
            for k, v in payload["state_dict"].items()
            if k.startswith("trunk.")
        },
        strict=True,
    )
    classifier = CategoryClassifier.load_from_checkpoint(
        classifier_path, map_location="cpu", weights_only=False
    ).eval()
    queries, features = [], []
    for start in range(0, len(eeg), batch_size):
        value = torch.from_numpy(eeg[start : start + batch_size].mean(2)).cuda()
        ids = torch.arange(10, device="cuda").repeat(len(value))
        with torch.autocast("cuda", torch.float16):
            h = hierarchical(value.flatten(0, 1), ids).fused
            c = clip(value.flatten(0, 1), ids)
        h = F.normalize(
            F.normalize(h.float().reshape(len(value), 10, -1), dim=-1).mean(1), dim=-1
        ).cpu()
        c = F.normalize(c.float().reshape(len(value), 10, -1).mean(1), dim=-1).cpu()
        queries.append(h)
        features.append(torch.cat([h, c], 1))
    scores = classifier(torch.cat(features))
    selected = select_reference(scores, torch.cat(queries), bank)
    top = scores.topk(5, dim=1)
    records = [
        {
            "sample": i,
            "predicted_category": names[int(top.indices[i, 0])],
            "top5": [names[int(k)] for k in top.indices[i]],
            "uncalibrated_logit_margin": float(top.values[i, 0] - top.values[i, 1]),
            "training_reference": bank["rows"][int(selected[i])],
        }
        for i in range(len(eeg))
    ]
    return bank["clip"][selected].float(), records


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eeg",
        type=Path,
        required=True,
        help="npy in project preprocessing/channel order: all 100 cached samples; encoder uses samples 20:100 (0-800ms)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path, default=OUTPUT / "category_prior/training_bank.pt")
    parser.add_argument(
        "--classifier", type=Path, default=OUTPUT / "category_classifier/checkpoints/best.ckpt"
    )
    parser.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=ARTIFACTS / "eeg_encoder/clip2b-warm-lr2/checkpoints/best.ckpt",
    )
    parser.add_argument(
        "--adapter", type=Path, default=ARTIFACTS / "dit_adapter/faza1-t8c/checkpoints/best.ckpt"
    )
    parser.add_argument("--null", type=Path, default=ARTIFACTS / "dit_adapter/t5_embeddings.pt")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--mode", choices=("prior", "text", "text-and-prior"), default="prior")
    parser.add_argument("--text-bank", type=Path, default=OUTPUT / "category_text.pt")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("this validated pipeline requires a CUDA GPU")
    if args.steps < 1 or not np.isfinite(args.guidance) or args.guidance < 0:
        raise ValueError("invalid diffusion settings")
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("choose an empty output directory to preserve previous results")
    eeg = validate_eeg(np.load(args.eeg, allow_pickle=False))
    bank = torch.load(args.bank, weights_only=False, map_location="cpu")
    condition, records = predict(eeg, bank, args.classifier, args.clip_checkpoint)
    text = None
    if args.mode != "prior":
        text_bank = torch.load(args.text_bank, weights_only=False, map_location="cpu", mmap=True)
        text = text_for_predictions(records, text_bank)
        del text_bank
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "complete": False,
        "mode": "EEG predicted-category visualization",
        "geometry_warning": "Object pose/detail may come from the reference or generator; not demonstrated EEG geometry",
        "confidence_warning": "Top-1 is a hypothesis, top5 are alternatives; scores are not calibrated probabilities",
        "subjects": list(range(1, 11)),
        "repetitions": 4,
        "window_ms": [0, 800],
        "inputs": {
            k: {"path": str(p), "sha256": sha256_of_file(p)}
            for k, p in {
                "eeg": args.eeg,
                "bank": args.bank,
                "classifier": args.classifier,
                "clip_encoder": args.clip_checkpoint,
                "adapter": args.adapter,
                "null": args.null,
            }.items()
        },
        "steps": args.steps,
        "guidance": args.guidance,
        "seed": args.seed,
        "predictions": records,
        "rendered": not args.no_render,
        "conditioning_mode": args.mode,
        "training_reference_used_by_generator": args.mode != "text",
        "text_bank_sha256": sha256_of_file(args.text_bank) if text is not None else None,
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(condition, args.output / "conditions.pt")
    del bank
    gc.collect()
    torch.cuda.empty_cache()
    if not args.no_render:
        from ..lightning.dit_adapter import DiTAdapterLightning

        state = torch.load(args.adapter, map_location="cpu", weights_only=False, mmap=True)
        model = (
            DiTAdapterLightning(
                **{**state["hyper_parameters"], "null_embeddings_file": str(args.null)}
            )
            .eval()
            .requires_grad_(False)
        )
        model.load_state_dict(state["state_dict"], strict=True)
        model.cuda()
        for i, embedding in enumerate(condition):
            latents = model.sample_images(
                (torch.zeros_like(embedding) if args.mode == "text" else embedding)[None].cuda(),
                steps=args.steps,
                guidance=args.guidance,
                seed=args.seed + i,
                **(
                    {
                        "prompt_embeds": text["embeds"][i : i + 1].cuda(),
                        "prompt_mask": text["mask"][i : i + 1].cuda(),
                        "lambda_override": 0.0 if args.mode == "text" else None,
                    }
                    if text is not None
                    else {}
                ),
            )
            model.decode_and_grid(latents).save(args.output / f"{i:03d}.png")
            print(
                json.dumps({"sample": i, "predicted_category": records[i]["predicted_category"]}),
                flush=True,
            )
    report["complete"] = True
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Complete: {report_path}", flush=True)


if __name__ == "__main__":
    main()
