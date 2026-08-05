"""Train a task-specific EEG edge residual against a fixed strong baseline."""

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

from .build_pyramid_edge_targets import PYRAMID_KEYS
from .official_200way_benchmark import cached_array
from .pyramid_edge import PyramidEdgeOutput, load_pyramid_edge_decoder, pyramid_edge_sample_errors
from .semantic_edge import EEGDINOGridPredictor
from .semantic_encoder import load_semantic_encoder
from .task_adapter import TaskResidualAdapter, cvar, improvement_loss, shuffled_ranking_loss
from .train_eeg_pyramid_edges import EEGPyramidDataset, all_records, concatenate, split_records
from .train_pyramid_edge_oracle import metrics as pyramid_metrics


def arguments():
    p = argparse.ArgumentParser(description="EEG pyramid-edge residual adapter v2")
    p.add_argument("--semantic-encoder", default="runs/eeg_semantic_encoder_v2/best.pt")
    p.add_argument("--baseline", default="runs/eeg_pyramid_edges_v1/best.pt")
    p.add_argument("--pyramid-decoder", default="runs/dino_pyramid_edge_oracle_v1/best.pt")
    p.add_argument("--visual-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    p.add_argument("--target-bank", default="data/derived/pyramid_edge_targets_v1.pt")
    p.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    p.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    p.add_argument("--cache", default="data/derived/eeg_float32_cache")
    p.add_argument("--output", default="runs/eeg_edge_adapter_v2")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--margin", type=float, default=0.025)
    p.add_argument("--rank-margin", type=float, default=0.18)
    p.add_argument("--panel-images", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def features(predictor, adapter, eeg):
    grid, contexts = predictor.forward_ensemble(eeg)
    return adapter(grid, contexts)


def diversity_penalty(output, target):
    pairs = (
        (output.semantic_logits32.sigmoid(), target["semantic32"]),
        (output.crisp_logits64.sigmoid(), target["crisp64"]),
        (output.perceptual_logits64.sigmoid(), target["perceptual64"]),
    )
    return (
        sum(
            F.relu(
                0.60 * t.flatten(1).std(0, unbiased=False).mean()
                - p.flatten(1).std(0, unbiased=False).mean()
            )
            for p, t in pairs
        )
        / 3
    )


def train_epoch(predictor, reference, adapter, decoder, loader, optimizer, scaler, device, args):
    predictor.eval()
    reference.eval()
    adapter.train()
    decoder.eval()
    totals = {}
    batches = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        target = {k: batch[k].to(device) for k in PYRAMID_KEYS}
        target_grid = batch["grid"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            ref_grid, _ = reference.forward_ensemble(eeg)
            base = decoder(ref_grid)
            base_error, _ = pyramid_edge_sample_errors(base, target)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            adapted = features(predictor, adapter, eeg)
            output = decoder(adapted.grid)
            error, parts = pyramid_edge_sample_errors(output, target)
            order = torch.roll(torch.arange(len(eeg), device=device), 1)
            wrong = {k: v[order] for k, v in target.items()}
            wrong_error, _ = pyramid_edge_sample_errors(output, wrong)
            task = error.mean()
            hard = cvar(error)
            improve = improvement_loss(error, base_error, args.margin)
            rank = (
                shuffled_ranking_loss(error, wrong_error, args.rank_margin)
                if len(eeg) > 1
                else task * 0
            )
            diversity = diversity_penalty(output, target)
            anchor = (1 - F.cosine_similarity(adapted.grid, target_grid, dim=-1)).mean()
            delta = adapted.grid_delta.square().mean()
            loss = (
                task
                + 0.70 * hard
                + 1.75 * improve
                + 0.60 * rank
                + 0.75 * diversity
                + 0.08 * anchor
                + 0.002 * delta
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(adapter.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        batches += 1
        values = {
            "loss": loss,
            "task": task,
            "hard": hard,
            "improvement": improve,
            "ranking": rank,
            "diversity": diversity,
            "grid_anchor": anchor,
            **parts,
        }
        for k, v in values.items():
            totals[k] = totals.get(k, 0.0) + float(v.detach().mean())
    return {k: v / max(batches, 1) for k, v in totals.items()}


@torch.no_grad()
def predict(predictor, adapter, decoder, loader, device):
    predictor.eval()
    adapter.eval()
    decoder.eval()
    outputs = []
    targets = {k: [] for k in PYRAMID_KEYS}
    grids = []
    true_grids = []
    for batch in loader:
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            value = features(predictor, adapter, batch["eeg"].to(device))
            output = decoder(value.grid)
        outputs.append(
            PyramidEdgeOutput(
                **{
                    f: getattr(output, f).float().cpu()
                    for f in PyramidEdgeOutput.__dataclass_fields__
                }
            )
        )
        for k in PYRAMID_KEYS:
            targets[k].append(batch[k].float())
        grids.append(value.grid.float().cpu())
        true_grids.append(batch["grid"].float())
    output = concatenate(outputs)
    target = {k: torch.cat(v) for k, v in targets.items()}
    grid = torch.cat(grids)
    truth = torch.cat(true_grids)
    return output, target, float(F.cosine_similarity(grid, truth, dim=-1).mean())


def evaluate(predictor, adapter, decoder, loader, device, seed):
    output, target, grid = predict(predictor, adapter, decoder, loader, device)
    value = pyramid_metrics(output, target)
    value["grid_cosine"] = grid
    order = torch.randperm(
        len(output.shape_logits16), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = PyramidEdgeOutput(
        **{f: getattr(output, f)[order] for f in PyramidEdgeOutput.__dataclass_fields__}
    )
    wrong = pyramid_metrics(shuffled, target)
    return value, wrong


def visual_score(value, shuffled):
    gap = sum(
        value[k] - shuffled[k]
        for k in (
            "semantic_tolerant_f1",
            "crisp_tolerant_f1",
            "perceptual_pearson",
            "distance_correlation",
        )
    )
    return (
        value["shape_iou"]
        + value["semantic_tolerant_f1"]
        + value["crisp_tolerant_f1"]
        + value["perceptual_pearson"]
        + value["distance_correlation"]
        + 0.9 * gap
        + 0.10 * value["grid_cosine"]
    )


def gray(value, tile=112):
    a = (value.squeeze().clamp(0, 1) * 255).byte().numpy()
    return Image.fromarray(a, "L").resize((tile, tile), Image.Resampling.NEAREST).convert("RGB")


@torch.no_grad()
def save_panel(reference, predictor, adapter, decoder, dataset, device, path, n):
    samples = [dataset[i] for i in range(min(n, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([s["eeg"] for s in samples]).to(device)
    rg, _ = reference.forward_ensemble(eeg)
    base = decoder(rg)
    value = features(predictor, adapter, eeg)
    pred = decoder(value.grid)
    tile, label = 112, 22
    headers = (
        "semantic GT",
        "semantic base",
        "semantic v2",
        "change x4",
        "crisp GT",
        "crisp base",
        "crisp v2",
        "change x4",
    )
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, h in enumerate(headers):
        draw.text((c * tile + 3, 3), h, fill="black")
    for i, s in enumerate(samples):
        y = (i + 1) * (tile + label)
        draw.text((3, y - label + 3), s["filename"], fill="black")
        sb = base.semantic_logits32.sigmoid().cpu()[i]
        sp = pred.semantic_logits32.sigmoid().cpu()[i]
        cb = base.crisp_logits64.sigmoid().cpu()[i]
        cp = pred.crisp_logits64.sigmoid().cpu()[i]
        values = (
            s["semantic32"],
            sb,
            sp,
            (sp - sb).abs() * 4,
            s["crisp64"],
            cb,
            cp,
            (cp - cb).abs() * 4,
        )
        for c, v in enumerate(values):
            canvas.paste(gray(v, tile), (c * tile, y))
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
    target_bank = torch.load(args.target_bank, map_location="cpu", weights_only=False)
    visual_grid = F.normalize(visual["dino_grid"].float(), dim=-1)
    state = torch.load(args.baseline, map_location="cpu", weights_only=False)
    encoder = load_semantic_encoder(args.semantic_encoder, device)
    predictor = EEGDINOGridPredictor(encoder, **state["model_config"]).to(device)
    predictor.load_state_dict(state["model"], strict=True)
    reference = copy.deepcopy(predictor).eval()
    for model in (predictor, reference):
        for p in model.parameters():
            p.requires_grad_(False)
    decoder, decoder_state = load_pyramid_edge_decoder(args.pyramid_decoder, device)
    for p in decoder.parameters():
        p.requires_grad_(False)
    adapter = TaskResidualAdapter(
        predictor.grid_dim, predictor.dim, predictor.subjects, predictor.heads, 2, 0.10
    ).to(device)
    arrays = [cached_array(Path(args.archives), Path(args.cache), s, "training") for s in range(10)]
    records = {
        s: split_records(args.index, s, visual, target_bank) for s in ("train", "val", "test")
    }
    if args.smoke:
        if any(not v for v in records.values()):
            available = all_records(args.index, visual, target_bank)
            a = max(1, int(0.7 * len(available)))
            b = max(a + 1, int(0.85 * len(available)))
            records = {"train": available[:a], "val": available[a:b], "test": available[b:]}
        records = {k: v[:24] for k, v in records.items()}
    datasets = {
        k: EEGPyramidDataset(arrays, v, visual_grid, target_bank) for k, v in records.items()
    }
    loaders = {
        k: DataLoader(
            v,
            batch_size=args.batch_size if k == "train" else args.eval_batch_size,
            shuffle=k == "train",
            num_workers=0,
        )
        for k, v in datasets.items()
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    identity = copy.deepcopy(adapter)
    baseline, baseline_wrong = evaluate(
        reference, identity, decoder, loaders["val"], device, args.seed
    )
    baseline_score = visual_score(baseline, baseline_wrong)
    (out / "baseline_validation.json").write_text(
        json.dumps(
            {"score": baseline_score, "validation": baseline, "shuffled": baseline_wrong}, indent=2
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    identity_payload = {
        "epoch": -1,
        "adapter": adapter.state_dict(),
        "adapter_config": adapter.config(),
        "predictor": state["model"],
        "predictor_config": state["model_config"],
        "baseline": args.baseline,
        "decoder": args.pyramid_decoder,
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
        val, wrong = evaluate(
            predictor, adapter, decoder, loaders["val"], device, args.seed + epoch + 1
        )
        current = visual_score(val, wrong)
        scheduler.step(current)
        improved = current > best + 1e-5
        stale = 0 if improved else stale + 1
        record = {
            "epoch": epoch,
            "score": current,
            "beats_frozen_baseline": current > baseline_score,
            "train": train,
            "validation": val,
            "shuffled": wrong,
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
            "decoder": args.pyramid_decoder,
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
        gap = val["semantic_tolerant_f1"] - wrong["semantic_tolerant_f1"]
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} baseline={baseline_score:.3f} semantic={val['semantic_tolerant_f1']:.3f} crisp={val['crisp_tolerant_f1']:.3f} gap={gap:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    adapter.load_state_dict(selected["adapter"])
    test, wrong = evaluate(predictor, adapter, decoder, loaders["test"], device, args.seed + 999)
    report = {
        "checkpoint_epoch": selected["epoch"],
        "validation_baseline_score": baseline_score,
        "validation_selected_score": selected["score"],
        "eeg": test,
        "shuffled": wrong,
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
    }
    (out / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
