"""Train a task-specific EEG color residual without moving the frozen baseline."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .color import (
    ColorOutput,
    color_descriptor,
    color_metrics,
    color_sample_errors,
    load_color_decoder,
    rendered_descriptor_contrastive,
)
from .official_200way_benchmark import cached_array
from .semantic_edge import EEGDINOGridPredictor
from .semantic_encoder import load_semantic_encoder
from .task_adapter import TaskResidualAdapter, cvar, improvement_loss, shuffled_ranking_loss
from .train_eeg_color import EEGColorDataset, split_records


def arguments():
    p = argparse.ArgumentParser(description="EEG color residual adapter v2")
    p.add_argument("--semantic-encoder", default="runs/eeg_semantic_encoder_v2/best.pt")
    p.add_argument("--baseline", default="runs/eeg_color_encoder_v1/best.pt")
    p.add_argument("--color-decoder", default="runs/dino_color_oracle_v1/best.pt")
    p.add_argument("--visual-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    p.add_argument("--rgb-bank", default="data/derived/rich_visual_targets_32.pt")
    p.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    p.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    p.add_argument("--cache", default="data/derived/eeg_float32_cache")
    p.add_argument("--output", default="runs/eeg_color_adapter_v2")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--margin", type=float, default=0.012)
    p.add_argument("--rank-margin", type=float, default=0.10)
    p.add_argument("--panel-images", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def semantic(predictor, eeg):
    batch, subjects = eeg.shape[:2]
    sid = torch.arange(subjects, device=eeg.device).repeat(batch)
    values = predictor.encoder(eeg.flatten(0, 1), sid)
    return F.normalize(values.reshape(batch, subjects, -1).mean(1), dim=-1)


def features(predictor, adapter, eeg):
    grid, contexts = predictor.forward_ensemble(eeg)
    return adapter(grid, contexts, semantic(predictor, eeg))


def rendered(output):
    return ColorOutput(output.rgb32, color_descriptor(output.rgb32))


def diversity_penalty(prediction, target, minimum=0.65):
    pred_std = prediction.flatten(1).std(0, unbiased=False).mean()
    target_std = target.flatten(1).std(0, unbiased=False).mean()
    return F.relu(minimum * target_std - pred_std)


def train_epoch(predictor, reference, adapter, decoder, loader, optimizer, scaler, device, args):
    predictor.eval()
    reference.eval()
    decoder.eval()
    adapter.train()
    totals = {}
    count = 0
    for batch in loader:
        eeg, rgb, desc = (
            batch["eeg"].to(device),
            batch["rgb"].to(device),
            batch["descriptor"].to(device),
        )
        target_grid = F.normalize(batch["grid"].to(device), dim=-1)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            ref_grid, _ = reference.forward_ensemble(eeg)
            ref_global = semantic(reference, eeg)
            base_output = rendered(decoder(ref_grid, ref_global))
            base_error, _ = color_sample_errors(base_output, rgb, desc)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            adapted = features(predictor, adapter, eeg)
            output = rendered(decoder(adapted.grid, adapted.global_feature))
            error, parts = color_sample_errors(output, rgb, desc)
            order = torch.roll(torch.arange(len(rgb), device=device), 1)
            wrong_error, _ = color_sample_errors(output, rgb[order], desc[order])
            task = error.mean()
            hard = cvar(error)
            improve = improvement_loss(error, base_error, args.margin)
            rank = (
                shuffled_ranking_loss(error, wrong_error, args.rank_margin)
                if len(rgb) > 1
                else task * 0
            )
            contrast = (
                rendered_descriptor_contrastive(output.rgb32, desc) if len(rgb) > 1 else task * 0
            )
            direct = F.smooth_l1_loss(adapted.descriptor, desc.float(), beta=0.04)
            diversity = diversity_penalty(output.rgb32, rgb)
            anchor = (1 - F.cosine_similarity(adapted.grid, target_grid, dim=-1)).mean()
            delta = adapted.grid_delta.square().mean() + adapted.global_delta.square().mean()
            loss = (
                task
                + 0.70 * hard
                + 1.75 * improve
                + 0.55 * rank
                + 0.20 * contrast
                + 0.20 * direct
                + 0.80 * diversity
                + 0.08 * anchor
                + 0.002 * delta
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(adapter.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        values = {
            "loss": loss,
            "task": task,
            "hard": hard,
            "improvement": improve,
            "ranking": rank,
            "contrastive": contrast,
            "diversity": diversity,
            "grid_anchor": anchor,
            **parts,
        }
        for k, v in values.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().mean())
        count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}


@torch.no_grad()
def predict(predictor, adapter, decoder, loader, device):
    predictor.eval()
    adapter.eval()
    decoder.eval()
    outputs = []
    rgbs = []
    descs = []
    grids = []
    targets = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            value = features(predictor, adapter, eeg)
            out = decoder(value.grid, value.global_feature)
        outputs.append(out.rgb32.float().cpu())
        rgbs.append(batch["rgb"].float())
        descs.append(batch["descriptor"].float())
        grids.append(value.grid.float().cpu())
        targets.append(batch["grid"].float())
    rgb = torch.cat(outputs)
    truth = torch.cat(rgbs)
    descriptor = torch.cat(descs)
    result = rendered(ColorOutput(rgb, descriptor.new_empty((len(rgb), 0))))
    grid_cos = float(F.cosine_similarity(torch.cat(grids), torch.cat(targets), dim=-1).mean())
    return result, truth, descriptor, grid_cos


def evaluate(predictor, adapter, decoder, loader, device, mean, seed=None):
    output, rgb, desc, grid = predict(predictor, adapter, decoder, loader, device)
    if seed is not None:
        order = torch.randperm(len(rgb), generator=torch.Generator().manual_seed(seed))
        output = ColorOutput(output.rgb32[order], output.descriptor[order])
    value = color_metrics(output, rgb, desc, mean)
    value["grid_cosine"] = grid
    return value


def score(value, shuffled):
    specificity = (
        value["palette_pearson"]
        - shuffled["palette_pearson"]
        + value["descriptor_cosine"]
        - shuffled["descriptor_cosine"]
    )
    return (
        1.25 * value["palette_pearson"]
        + value["descriptor_cosine"]
        + value.get("rgb_gain_vs_train_mean", 0)
        + 1.25 * specificity
        - value["rgb_mae"]
        - 1.5 * value["chroma_mae"]
        + 0.25 * min(value["between_image_std_ratio"], 1.0)
        + 0.10 * value["grid_cosine"]
    )


def to_image(value, tile=128):
    a = (value.permute(1, 2, 0).clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(a).resize((tile, tile), Image.Resampling.NEAREST)


@torch.no_grad()
def save_panel(reference, predictor, adapter, decoder, dataset, device, path, n):
    samples = [dataset[i] for i in range(min(n, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([s["eeg"] for s in samples]).to(device)
    rg, _ = reference.forward_ensemble(eeg)
    base = decoder(rg, semantic(reference, eeg)).rgb32.cpu()
    value = features(predictor, adapter, eeg)
    pred = decoder(value.grid, value.global_feature).rgb32.cpu()
    tile, label = 128, 22
    headers = ("target", "frozen baseline", "adapter v2", "difference x4")
    canvas = Image.new("RGB", (4 * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, h in enumerate(headers):
        draw.text((c * tile + 3, 3), h, fill="black")
    for i, s in enumerate(samples):
        y = (i + 1) * (tile + label)
        draw.text((3, y - label + 3), s["filename"], fill="black")
        diff = (pred[i] - base[i]).abs().mul(4).clamp(0, 1)
        for c, v in enumerate((s["rgb"], base[i], pred[i], diff)):
            canvas.paste(to_image(v), (c * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    args = arguments()
    if args.smoke:
        args.epochs = 1
        args.batch_size = 2
        args.eval_batch_size = 4
        args.panel_images = 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    visual = torch.load(args.visual_bank, map_location="cpu", weights_only=False)
    rich = torch.load(args.rgb_bank, map_location="cpu", weights_only=False)
    state = torch.load(args.baseline, map_location="cpu", weights_only=False)
    encoder = load_semantic_encoder(args.semantic_encoder, device)
    predictor = EEGDINOGridPredictor(encoder, **state["model_config"]).to(device)
    predictor.load_state_dict(state["model"], strict=True)
    reference = copy.deepcopy(predictor).eval()
    for model in (predictor, reference):
        for p in model.parameters():
            p.requires_grad_(False)
    decoder, decoder_state = load_color_decoder(args.color_decoder, device)
    for p in decoder.parameters():
        p.requires_grad_(False)
    adapter = TaskResidualAdapter(
        predictor.grid_dim,
        predictor.dim,
        predictor.subjects,
        predictor.heads,
        2,
        0.10,
        predictor.encoder.output_dim,
        visual["color"].shape[-1],
    ).to(device)
    arrays = [cached_array(Path(args.archives), Path(args.cache), s, "training") for s in range(10)]
    records = {
        s: split_records(Path(args.index), s, visual, rich) for s in ("train", "val", "test")
    }
    if args.smoke:
        records = {k: v[:24] for k, v in records.items()}
    datasets = {k: EEGColorDataset(arrays, v, visual, rich) for k, v in records.items()}
    loaders = {
        k: DataLoader(
            v,
            batch_size=args.batch_size if k == "train" else args.eval_batch_size,
            shuffle=k == "train",
            num_workers=0,
        )
        for k, v in datasets.items()
    }
    mean = rich["rgb"][[r[2] for r in records["train"]]].float().mean(0, keepdim=True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    base_adapter = copy.deepcopy(adapter)
    baseline = evaluate(reference, base_adapter, decoder, loaders["val"], device, mean)
    baseline_shuffled = evaluate(
        reference, base_adapter, decoder, loaders["val"], device, mean, args.seed
    )
    baseline_score = score(baseline, baseline_shuffled)
    (out / "baseline_validation.json").write_text(
        json.dumps(
            {"score": baseline_score, "validation": baseline, "shuffled": baseline_shuffled},
            indent=2,
        ),
        encoding="utf-8",
    )
    save_panel(
        reference,
        predictor,
        adapter,
        decoder,
        datasets["val"],
        device,
        out / "panels" / "baseline.png",
        args.panel_images,
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    identity_payload = {
        "epoch": -1,
        "adapter": adapter.state_dict(),
        "adapter_config": adapter.config(),
        "predictor": state["model"],
        "predictor_config": state["model_config"],
        "baseline": args.baseline,
        "decoder": args.color_decoder,
        "decoder_epoch": decoder_state.get("epoch"),
        "score": baseline_score,
        "baseline_score": baseline_score,
        "baseline_retained": True,
    }
    torch.save(identity_payload, out / "best.pt")
    best = baseline_score
    stale = 0
    history = out / "history.jsonl"
    history.unlink(missing_ok=True)
    for epoch in range(args.epochs):
        train = train_epoch(
            predictor,
            reference,
            adapter,
            decoder,
            loaders["train"],
            optimizer,
            scaler,
            device,
            args,
        )
        val = evaluate(predictor, adapter, decoder, loaders["val"], device, mean)
        shuffled = evaluate(
            predictor, adapter, decoder, loaders["val"], device, mean, args.seed + epoch + 1
        )
        current = score(val, shuffled)
        scheduler.step(current)
        improved = current > best + 1e-5
        stale = 0 if improved else stale + 1
        record = {
            "epoch": epoch,
            "score": current,
            "beats_frozen_baseline": current > baseline_score,
            "train": train,
            "validation": val,
            "shuffled": shuffled,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with history.open("a", encoding="utf-8") as h:
            h.write(json.dumps(record) + "\n")
        payload = {
            "epoch": epoch,
            "adapter": adapter.state_dict(),
            "adapter_config": adapter.config(),
            "predictor": state["model"],
            "predictor_config": state["model_config"],
            "baseline": args.baseline,
            "decoder": args.color_decoder,
            "decoder_epoch": decoder_state.get("epoch"),
            "score": current,
            "baseline_score": baseline_score,
            "record": record,
        }
        torch.save(payload, out / "last.pt")
        if improved:
            best = current
            torch.save(payload, out / "best.pt")
        save_panel(
            reference,
            predictor,
            adapter,
            decoder,
            datasets["val"],
            device,
            out / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} baseline={baseline_score:.3f} palette={val['palette_pearson']:.3f} specificity={val['descriptor_cosine'] - shuffled['descriptor_cosine']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    adapter.load_state_dict(selected["adapter"])
    test = evaluate(predictor, adapter, decoder, loaders["test"], device, mean)
    shuffled = evaluate(predictor, adapter, decoder, loaders["test"], device, mean, args.seed + 999)
    report = {
        "checkpoint_epoch": selected["epoch"],
        "validation_baseline_score": baseline_score,
        "validation_selected_score": selected["score"],
        "eeg": test,
        "shuffled": shuffled,
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
    }
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
