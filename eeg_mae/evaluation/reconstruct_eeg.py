"""Generate from preprocessed EEG, preserving true zero-based subject IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..lightning.common import sha256_of_file
from ..lightning.ensemble_trunk import EnsembleTrunkLightning


def eeg_conditions(model, array, subject_ids):
    if array.ndim != 5 or array.shape[-2:] != (17, 100) or any(size == 0 for size in array.shape):
        raise ValueError("expected EEG [N,S,R,17,100]")
    if not np.isfinite(array).all():
        raise ValueError("EEG contains nonfinite values")
    ids = torch.as_tensor(subject_ids, dtype=torch.long, device=model.device)
    if len(ids) != array.shape[1] or len(ids.unique()) != len(ids):
        raise ValueError("provide one distinct subject ID per S column")
    if not len(ids) or ids.min() < 0 or ids.max() >= model.trunk.subjects:
        raise ValueError("subject ID outside checkpoint roster")
    eeg = torch.as_tensor(np.asarray(array, dtype=np.float32).mean(axis=2), device=model.device)
    with torch.no_grad():
        return model(eeg, ids).float().cpu()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eeg", required=True)
    parser.add_argument("--subject-ids", required=True, type=int, nargs="+")
    parser.add_argument("--encoder", default="runs/eeg_encoder/best.ckpt")
    parser.add_argument("--adapter", default="runs/dit_adapter/best.ckpt")
    parser.add_argument("--null-embeddings", default="data/derived/pixart_null.pt")
    parser.add_argument("--repo", default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if args.steps < 1 or not np.isfinite(args.guidance):
        raise ValueError("steps must be positive and guidance finite")
    model = (
        EnsembleTrunkLightning.load_from_checkpoint(args.encoder, map_location="cpu")
        .to(args.device)
        .eval()
    )
    model.freeze()
    conditions = eeg_conditions(
        model, np.load(args.eeg, mmap_mode="r", allow_pickle=False), args.subject_ids
    )
    if conditions.shape[-1] != 768 or not torch.isfinite(conditions).all():
        raise ValueError("adapter requires finite raw 768-dimensional CLIP conditions")
    output.mkdir(parents=True, exist_ok=False)

    hashes = {"eeg": sha256_of_file(args.eeg), "encoder": sha256_of_file(args.encoder)}
    if not args.no_render:
        hashes["adapter"] = sha256_of_file(args.adapter)
        hashes["null_embeddings"] = sha256_of_file(args.null_embeddings)
    torch.save(
        {"clip_conditions": conditions, "subject_ids": args.subject_ids}, output / "conditions.pt"
    )
    report = {
        "parameters": vars(args),
        "samples": [],
        "hashes": hashes,
        "conditioning": "raw EEG-predicted CLIP; repetition mean then subject fusion",
    }
    if not args.no_render:
        from ..lightning.dit_adapter import DiTAdapterLightning

        adapter = (
            DiTAdapterLightning.load_from_checkpoint(
                args.adapter,
                map_location="cpu",
                repo=args.repo,
                null_embeddings_file=args.null_embeddings,
                strict=True,
            )
            .to(args.device)
            .eval()
        )
        adapter.freeze()
        for index, condition in enumerate(conditions):
            seed = args.seed + index
            latents = adapter.sample_images(
                condition[None].to(args.device), steps=args.steps, guidance=args.guidance, seed=seed
            )
            filename = f"{index:05d}.png"
            adapter.decode_and_grid(latents).save(output / filename)
            report["samples"].append({"index": index, "seed": seed, "image": filename})
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {len(conditions)} conditions to {output}")


if __name__ == "__main__":
    main()
