"""Encode the known category vocabulary with frozen NF4 T5 for PixArt.

Text depends only on category names. It contains no target image descriptions.
T5 is unloaded before diffusion so the full pipeline fits a 6GB GPU.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import torch

from .dit_ten_subjects import OUTPUT

REPO = "PixArt-alpha/PixArt-XL-2-512x512"
REVISION = "50f702106901db6d0f8b67eb88e814c56ded2692"
CONCEPTS_URL = "https://raw.githubusercontent.com/ViCCo-Group/thingsvision/57e91c9754dfea8579ffc58c0ae42d9dc60080af/data/files/things_concepts.tsv"
CONCEPTS_SHA256 = "f6eb4ec6fcad61e50a746bee820f98d1595fe655e9e210a9a407e9a17fd8cee4"


def load_concepts(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        return {row["uniqueID"]: row for row in csv.DictReader(handle, delimiter="\t")}


def category_prompt(name, concepts=None):
    key = re.sub(r"^\d+_", "", name)
    if concepts is not None:
        concept = concepts[key]
        label = concept["Word"].replace("_", " ")
        taxonomy = (
            concept["Top-down Category (manual selection)"] or concept["All Bottom-up Categories"]
        )
        definition = " ".join(
            concept["Definition (from WordNet, Google, or Wikipedia)"].split(";")[0].split()[:24]
        )
        return f"a photograph depicting {label}. {taxonomy}. {definition}"
    label = re.sub(r"^\d+_", "", name).replace("_", " ")
    label = re.sub(r"\d+$", "", label).strip()
    return f"a photograph of a {label}"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, default=OUTPUT / "category_prior/training_bank.pt")
    parser.add_argument("--output", type=Path, default=OUTPUT / "category_text.pt")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=32)
    parser.add_argument(
        "--concepts", type=Path, help="THINGS concepts TSV for vocabulary-wide sense disambiguation"
    )
    parser.add_argument(
        "--fetch-concepts",
        action="store_true",
        help="fetch the pinned THINGS vocabulary when --concepts is missing",
    )
    parser.add_argument("--cache-dir", type=Path, default=OUTPUT / "hf_cache")
    parser.add_argument(
        "--text-source",
        type=Path,
        help="optional local PixArt snapshot with text_encoder and tokenizer",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("text bank exists; choose a fresh output")
    torch.set_num_threads(2)
    if args.fetch_concepts:
        import urllib.request

        if args.concepts is None:
            args.concepts = Path("data/derived/things_concepts.tsv")
        if not args.concepts.exists():
            with urllib.request.urlopen(CONCEPTS_URL, timeout=30) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != CONCEPTS_SHA256:
                raise ValueError("THINGS vocabulary checksum mismatch")
            args.concepts.parent.mkdir(parents=True, exist_ok=True)
            args.concepts.write_bytes(content)
    from transformers import BitsAndBytesConfig, T5EncoderModel, T5Tokenizer

    bank = torch.load(args.bank, map_location="cpu", weights_only=False)
    names = bank["category_names"]
    del bank
    concepts = load_concepts(args.concepts) if args.concepts else None
    prompts = [category_prompt(name, concepts) for name in names]
    source = str(args.text_source) if args.text_source else REPO
    tokenizer = T5Tokenizer.from_pretrained(
        source, subfolder="tokenizer", revision=REVISION, cache_dir=args.cache_dir
    )
    if max(len(ids) for ids in tokenizer(prompts, truncation=False).input_ids) > args.max_length:
        raise ValueError("a category prompt would be truncated; increase --max-length")
    model = T5EncoderModel.from_pretrained(
        source,
        subfolder="text_encoder",
        revision=REVISION,
        cache_dir=args.cache_dir,
        variant="fp16",
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    ).eval()
    embeddings, masks = [], []
    for start in range(0, len(prompts), args.batch_size):
        tokens = tokenizer(
            prompts[start : start + args.batch_size],
            padding="max_length",
            max_length=args.max_length,
            truncation=True,
            return_tensors="pt",
        ).to("cuda")
        hidden = model(
            input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
        ).last_hidden_state
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("nonfinite T5 embeddings")
        embeddings.append(hidden.half().cpu())
        masks.append(tokens.attention_mask.cpu())
        if start % 80 == 0:
            print(f"T5 categories {start + len(hidden)}/{len(prompts)}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "category_names": names,
            "prompts": prompts,
            "embeds": torch.cat(embeddings),
            "mask": torch.cat(masks),
            "metadata": {
                "repo": REPO,
                "revision": REVISION,
                "quantization": "NF4 double quantization, bfloat16 compute",
                "prompt_policy": "category vocabulary only, optional THINGS definitions; no image captions or instance orientation",
                "concepts_sha256": hashlib.sha256(args.concepts.read_bytes()).hexdigest()
                if args.concepts
                else None,
                "concepts_reference_url": CONCEPTS_URL if args.concepts else None,
                "max_length": args.max_length,
            },
        },
        args.output,
    )
    print(
        json.dumps({"complete": True, "output": str(args.output), "categories": len(names)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
