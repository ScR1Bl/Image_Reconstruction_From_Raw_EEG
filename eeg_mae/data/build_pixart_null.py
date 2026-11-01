"""Extract PixArt's learned null caption without loading a T5 text encoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

REVISION = "50f702106901db6d0f8b67eb88e814c56ded2692"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="PixArt-alpha/PixArt-XL-2-512x512")
    parser.add_argument("--output", default="data/derived/pixart_null.pt")
    parser.add_argument(
        "--reference", help="Optional preserved null artifact to verify shape and values"
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    local = Path(args.repo)
    if local.is_dir():
        weights = local / "transformer" / "diffusion_pytorch_model.safetensors"
    else:
        weights = hf_hub_download(
            args.repo, "transformer/diffusion_pytorch_model.safetensors", revision=REVISION
        )
    # Recent diffusers omits this historical field when constructing its module.
    # Read the pinned source weights, without instantiating transformer or T5.
    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        embeds = handle.get_tensor("caption_projection.y_embedding").float().unsqueeze(0)
    if embeds.shape != (1, 120, 4096):
        raise ValueError(f"unexpected null shape {embeds.shape}")
    mask = torch.ones(embeds.shape[:2], dtype=torch.long)
    if args.reference:
        ref = torch.load(args.reference, map_location="cpu", weights_only=False)["null_learned"]
        if ref["embeds"].shape != embeds.shape or ref["mask"].shape != mask.shape:
            raise ValueError("preserved null artifact shape differs")
        if not torch.allclose(ref["embeds"].float(), embeds, atol=0.02, rtol=0.01):
            raise ValueError("preserved learned null differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "null_learned": {"embeds": embeds, "mask": mask},
            "metadata": {
                "repo": args.repo,
                "revision": REVISION,
                "source": "caption_projection.y_embedding",
            },
        },
        output,
    )
    print(f"saved {output}: {tuple(embeds.shape)}")


if __name__ == "__main__":
    main()
