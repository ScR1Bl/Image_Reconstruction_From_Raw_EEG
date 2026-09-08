"""Side-by-side grid: original stimulus (top) vs image generated from its CLIP
embedding (bottom), for held-out images the adapter never saw (Step 3, Faza 2).

This is the visual-inspection artifact of the DoD: it makes the adapter's
behavior judgeable by eye, not only through retrieval numbers.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .official_200way_benchmark import training_target_indices


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original vs generated comparison grid")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip-bank", default="data/derived/clip_targets_vitl14_laion2b.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0, help="ktore obrazy val (przesuniecie)")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--cfg", type=float, default=4.5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--output", default="artifacts/dit_adapter/faza2/grids/original_vs_generated.png"
    )
    return parser.parse_args()


def main() -> None:
    from eeg_mae.lightning.dit_adapter import DiTAdapterLightning

    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")

    clip = torch.load(args.clip_bank, map_location="cpu", weights_only=False)
    ti, _, val_idx = training_target_indices(Path(args.index), clip)
    rows = torch.from_numpy(ti[val_idx][args.offset : args.offset + args.count]).long()
    names = [clip["filenames"][int(r)] for r in rows]
    cond = clip["clip_global"][rows].float().to(device)

    # oryginaly z archiwum (te same pliki, ktore dostal teacher CLIP)
    with zipfile.ZipFile(args.images_zip) as archive:
        by_name = {}
        for member in archive.namelist():
            if member.lower().endswith((".jpg", ".jpeg", ".png")):
                by_name.setdefault(Path(member).name, member)
        originals = [Image.open(io.BytesIO(archive.read(by_name[n]))).convert("RGB") for n in names]

    module = (
        DiTAdapterLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", strict=False)
        .to(device)
        .eval()
    )

    generated = []
    with torch.no_grad():
        for start in range(0, len(cond), 4):
            chunk = cond[start : start + 4]
            latents = module.sample_images(
                chunk, steps=args.steps, guidance=args.cfg, seed=args.seed + start
            )
            for i in range(len(chunk)):
                generated.append(module.decode_and_grid(latents[i : i + 1]))

    tile, label = args.tile, 22
    grid = Image.new("RGB", (args.count * tile, 2 * tile + 2 * label), "white")
    draw = ImageDraw.Draw(grid)
    for i, (orig, gen, name) in enumerate(zip(originals, generated, names)):
        concept = name.rsplit("_", 1)[0]
        draw.text((i * tile + 4, 4), concept[:30], fill="black")
        grid.paste(orig.resize((tile, tile)), (i * tile, label))
        draw.text((i * tile + 4, label + tile + 4), "wygenerowany", fill="black")
        grid.paste(gen.resize((tile, tile)), (i * tile, 2 * label + tile))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    print(f"siatka: {out}")
    print("koncepty:", ", ".join(n.rsplit("_", 1)[0] for n in names))


if __name__ == "__main__":
    main()
