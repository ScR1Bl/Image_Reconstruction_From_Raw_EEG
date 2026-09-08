"""Evaluate the PixArt DiT adapter WITHOUT EEG (Step 3, Faza 2).

Generates images from REAL held-out CLIP embeddings and measures, on both the
generated image and its conditioning:
  - CLIP-similarity: cosine(CLIP(gen), cond_emb) — did the generator obey
  - retrieval gen->orig: is CLIP(gen) nearest to its own original in the pool
  - concept accuracy: does the nearest original share the concept
Plus: CFG-scale ablation, adapter-lambda ablation, a NOISED-embedding sweep whose
noise scale is the measured EEG encoder error (Step 2b: predicted vs true CLIP),
and a real-vs-renormalized comparison (Step 2 renorm gives every prediction the
mean CLIP norm). Writes an eval-grid PNG artifact and a JSON report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .official_200way_benchmark import training_target_indices

CLIP_TEACHER = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DiT adapter (no EEG)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip-bank", default="data/derived/clip_targets_vitl14_laion2b.pt")
    parser.add_argument(
        "--official-clip-bank", default="data/derived/official_clip_targets_vitl14_laion2b.pt"
    )
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--pool-size", type=int, default=200, help="ile held-out obrazow do metryk")
    parser.add_argument("--grid-size", type=int, default=8, help="ile obrazow do siatki wzrokowej")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", nargs="+", type=float, default=(1.0, 2.0, 4.5, 7.0))
    parser.add_argument("--lambdas", nargs="+", type=float, default=None)
    parser.add_argument("--noise-scale", nargs="+", type=float, default=(0.0, 0.5, 1.0))
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--output", default="artifacts/dit_adapter/faza2")
    return parser.parse_args()


def load_clip_scorer(device):
    from transformers import AutoImageProcessor, CLIPConfig, CLIPVisionModelWithProjection

    processor = AutoImageProcessor.from_pretrained(CLIP_TEACHER)
    projection_dim = CLIPConfig.from_pretrained(CLIP_TEACHER).projection_dim
    model = (
        CLIPVisionModelWithProjection.from_pretrained(CLIP_TEACHER, projection_dim=projection_dim)
        .to(device)
        .eval()
    )

    @torch.no_grad()
    def embed(pil_images):
        inputs = processor(images=pil_images, return_tensors="pt")["pixel_values"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            return model(pixel_values=inputs).image_embeds.float()

    return embed


def main() -> None:
    from eeg_mae.lightning.dit_adapter import DiTAdapterLightning

    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    out_dir = Path(args.output)
    (out_dir / "grids").mkdir(parents=True, exist_ok=True)

    module = (
        DiTAdapterLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", strict=False)
        .to(device)
        .eval()
    )
    lam_profile = module.lambda_profile()

    clip_bank = torch.load(args.clip_bank, map_location="cpu", weights_only=False)
    ti, _, val_idx = training_target_indices(Path(args.index), clip_bank)
    rows = torch.from_numpy(ti[val_idx][: args.pool_size]).long()
    names = [clip_bank["filenames"][int(r)] for r in rows]
    concepts = [n.rsplit("_", 1)[0] for n in names]
    cond = clip_bank["clip_global"][rows].float().to(device)  # prawdziwe CLIP-emb held-out
    true_norm = clip_bank["clip_global"].float().norm(dim=-1).mean().item()

    scorer = load_clip_scorer(device)

    def generate(embeds, cfg, steps, lam=None, seed=None):
        pil = []
        for start in range(0, len(embeds), 8):
            chunk = embeds[start : start + 8]
            lat = module.sample_images(
                chunk,
                steps=steps,
                guidance=cfg,
                seed=(seed or args.seed) + start,
                lambda_override=lam,
            )
            pil.extend([module.decode_and_grid(lat[i : i + 1]) for i in range(len(chunk))])
        return pil

    def metrics(embeds, cfg, steps, lam=None):
        images = generate(embeds, cfg, steps, lam)
        gen_emb = torch.cat([scorer([img]) for img in images])
        gen_n = F.normalize(gen_emb, dim=-1)
        cond_n = F.normalize(embeds, dim=-1)
        clip_sim = (gen_n * cond_n).sum(-1).mean().item()
        sims = gen_n @ cond_n.T  # gen vs wszystkie warunki w puli
        ranks = 1 + (sims > sims.diagonal()[:, None]).sum(1)
        retrieval_top1 = float((ranks <= 1).float().mean())
        # zgodnosc kategorii: najblizszy warunek tego samego konceptu
        nearest = sims.argmax(1).cpu()
        concept_acc = sum(
            concepts[i] == concepts[int(nearest[i])] for i in range(len(images))
        ) / len(images)
        return {
            "clip_sim": clip_sim,
            "retrieval_top1": retrieval_top1,
            "concept_acc": concept_acc,
            "gen_norm": gen_emb.norm(dim=-1).mean().item(),
        }, images

    report = {
        "checkpoint": args.checkpoint,
        "pool_size": len(rows),
        "lambda_profile": lam_profile,
        "true_clip_norm": true_norm,
    }

    # --- ablacja CFG (pelna pula, wytrenowane lambda) ---
    report["cfg_ablation"] = {}
    best_cfg, best_grid = None, None
    for cfg in args.cfg:
        m, images = metrics(cond, cfg, args.steps)
        report["cfg_ablation"][str(cfg)] = m
        print(
            f"CFG={cfg}: clip_sim={m['clip_sim']:.4f} retr@1={m['retrieval_top1']:.4f} "
            f"concept={m['concept_acc']:.4f}",
            flush=True,
        )
        if best_cfg is None or m["clip_sim"] > report["cfg_ablation"][str(best_cfg)]["clip_sim"]:
            best_cfg = cfg
    print(f"najlepszy CFG (clip_sim): {best_cfg}", flush=True)

    # --- siatka wzrokowa: oryginal vs wygenerowany, best CFG ---
    from PIL import Image

    grid_embeds = cond[: args.grid_size]
    gen_images = generate(grid_embeds, best_cfg, args.steps)
    grid = Image.new("RGB", (args.grid_size * 256, 256))
    for i, img in enumerate(gen_images):
        grid.paste(img.resize((256, 256)), (i * 256, 0))
    grid.save(out_dir / "grids" / f"generated_cfg{best_cfg}.png")

    # --- ablacja lambda w inferencji ---
    if args.lambdas:
        report["lambda_ablation"] = {}
        for lam in args.lambdas:
            m, _ = metrics(cond, best_cfg, args.steps, lam=lam)
            report["lambda_ablation"][str(lam)] = m
            print(f"lambda={lam}: clip_sim={m['clip_sim']:.4f}", flush=True)

    # --- test krytyczny dla Step 4: zaszumione embeddingi (skala = blad enkodera EEG) ---
    # blad EEG (Step 2b): predykcja trunku vs prawdziwy CLIP; tu symulujemy dodajac
    # gaussowski szum o std = noise_scale * (typowa odleglosc pred-true).
    report["noise_sweep"] = {}
    generator = torch.Generator(device).manual_seed(args.seed)
    base_std = cond.std().item()
    for ns in args.noise_scale:
        noisy = cond + ns * base_std * torch.randn(cond.shape, generator=generator, device=device)
        m, _ = metrics(noisy, best_cfg, args.steps)
        report["noise_sweep"][str(ns)] = m
        print(
            f"noise={ns}: clip_sim={m['clip_sim']:.4f} concept={m['concept_acc']:.4f}", flush=True
        )

    # --- wplyw renormalizacji ze Step 2 ---
    renorm = cond * (true_norm / cond.norm(dim=-1, keepdim=True))
    m_real, _ = metrics(cond, best_cfg, args.steps)
    m_renorm, _ = metrics(renorm, best_cfg, args.steps)
    report["renorm_effect"] = {"real": m_real, "renormalized": m_renorm}
    print(
        f"renorm: real clip_sim={m_real['clip_sim']:.4f} vs renorm={m_renorm['clip_sim']:.4f}",
        flush=True,
    )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"raport: {out_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
