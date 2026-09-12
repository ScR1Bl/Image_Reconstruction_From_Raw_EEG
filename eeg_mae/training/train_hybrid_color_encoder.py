"""Train feature-level fusion of the proven and direct EEG color experts."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..evaluation.official_200way_benchmark import cached_array
from ..models.advanced_color import FullColorLatent, advanced_color_loss, color_artifact_metrics
from ..models.color import ColorOutput, color_descriptor, color_metrics, rgb_to_oklab
from ..models.hybrid_color import EEGHybridColorEncoder, initialize_hybrid_color_encoder
from ..models.model_utils import variance_covariance_regularization
from ..models.task_adapter import cvar, improvement_loss, shuffled_ranking_loss
from .color_training import EEGColorDataset, load_color_oracle, split_records, training_rgb_mean
from .train_color_encoder import color_sample_error, rare_chroma_loss, to_image
from .train_color_oracle import selection_score, update_ema


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independent hybrid EEG color encoder")
    parser.add_argument("--legacy-root", default="runs/eeg_color_encoder")
    parser.add_argument("--direct-root", default="runs/eeg_color_encoder_direct")
    parser.add_argument("--latent-bank", default="data/derived/full_color_latents_antiring")
    parser.add_argument("--rich-bank", default="data/derived/rich_dino_color_v2")
    parser.add_argument(
        "--oracle-renderer", default="runs/eeg_color_encoder/components/color_oracle.pt"
    )
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--output", default="runs/eeg_color_encoder_hybrid")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.012)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--ema-decay", type=float, default=0.998)
    parser.add_argument("--rank-margin", type=float, default=0.04)
    parser.add_argument("--guard-margin", type=float, default=0.002)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def spatial_color_terms(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    pred, truth = rgb_to_oklab(prediction), rgb_to_oklab(target)
    pred_chroma = pred[:, 1:].square().sum(1).sqrt()
    true_chroma = truth[:, 1:].square().sum(1).sqrt()
    threshold = torch.quantile(true_chroma.flatten(1), 0.90, dim=1)[:, None, None]
    salient = (true_chroma >= threshold).float()
    salient_error = ((pred[:, 1:] - truth[:, 1:]).abs().mean(1) * salient).sum(
        (1, 2)
    ) / salient.sum((1, 2)).clamp_min(1)
    height, width = true_chroma.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=target.device),
        torch.linspace(-1, 1, width, device=target.device),
        indexing="ij",
    )

    def centroid(magnitude: torch.Tensor) -> torch.Tensor:
        mass = magnitude.sum((1, 2)).clamp_min(1e-5)
        return torch.stack(
            ((magnitude * xx).sum((1, 2)) / mass, (magnitude * yy).sum((1, 2)) / mass), -1
        )

    centroid_error = (centroid(pred_chroma) - centroid(true_chroma)).square().sum(1).sqrt()
    layout4 = (
        (F.adaptive_avg_pool2d(pred, 4) - F.adaptive_avg_pool2d(truth, 4)).abs().mean((1, 2, 3))
    )
    layout8 = (
        (F.adaptive_avg_pool2d(pred, 8) - F.adaptive_avg_pool2d(truth, 8)).abs().mean((1, 2, 3))
    )
    strength_recall = torch.minimum(pred_chroma, true_chroma).mul(salient).sum((1, 2)) / (
        true_chroma.mul(salient).sum((1, 2)).clamp_min(1e-5)
    )
    return {
        "salient_chroma_mae": salient_error,
        "chroma_centroid_error": centroid_error,
        "layout4_oklab_mae": layout4,
        "layout8_oklab_mae": layout8,
        "salient_chroma_recall": strength_recall.clamp(0, 1),
    }


def safe_routing_terms(output, target: torch.Tensor) -> dict[str, torch.Tensor]:
    """Supervise the router and punish local errors worse than both parents."""
    truth = rgb_to_oklab(target).float()
    legacy = output.legacy_image.oklab.float()
    direct = output.direct_image.oklab.float()
    proposal = output.proposal_oklab.float()
    experts = torch.stack((legacy, direct, proposal), 2)
    # Each OKLab channel chooses its own best expert at every pixel.
    channel_error = (experts - truth[:, :, None]).abs()
    labels = channel_error.detach().argmin(2, keepdim=True)
    routing_nll = -output.expert_weights.float().clamp_min(1e-6).log().gather(2, labels).mean()

    perceptual_scale = truth.new_tensor([0.65, 1.0, 1.0])[None, :, None, None]

    def delta(value: torch.Tensor) -> torch.Tensor:
        return ((value - truth) * perceptual_scale).square().sum(1).add(1e-8).sqrt()

    legacy_delta, direct_delta = delta(legacy), delta(direct)
    parent_delta = torch.minimum(legacy_delta, direct_delta).detach()
    final_excess = F.relu(delta(output.image.oklab.float()) - parent_delta + 0.002).flatten(1)
    proposal_excess = F.relu(delta(proposal) - parent_delta + 0.003).flatten(1)
    count = max(1, round(final_excess.shape[1] * 0.10))
    pixel_guard = final_excess.topk(count, 1).values.mean()
    proposal_guard = proposal_excess.topk(count, 1).values.mean()
    proposal_reconstruction = F.smooth_l1_loss(proposal, truth, beta=0.02)
    return {
        "routing_nll": routing_nll,
        "pixel_guard_cvar10": pixel_guard,
        "proposal_guard_cvar10": proposal_guard,
        "proposal_reconstruction": proposal_reconstruction,
    }


def image_metrics(prediction, target, train_mean, oklab=None, linear=None):
    values = color_metrics(
        ColorOutput(prediction, color_descriptor(prediction)),
        target,
        color_descriptor(target),
        train_mean,
    )
    values.update(
        color_artifact_metrics(rgb_to_oklab(prediction) if oklab is None else oklab, target, linear)
    )
    values.update(
        {key: float(value.mean()) for key, value in spatial_color_terms(prediction, target).items()}
    )
    return values


def set_training_mode(model: EEGHybridColorEncoder) -> None:
    model.eval()
    model.fusion.train()


def train_epoch(model, ema, oracle, loader, optimizer, scaler, device, args):
    set_training_mode(model)
    totals, batches = {}, 0
    for batch in loader:
        eeg = batch["eeg"].to(device, non_blocking=True)
        target = batch["rgb"].to(device, non_blocking=True)
        true_latent = FullColorLatent(
            batch["grid"].float().to(device), batch["global"].float().to(device)
        )
        with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            teacher = oracle.decode_full_latent(true_latent)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(eeg)
            reconstruction, parts = advanced_color_loss(
                output.image,
                target,
                chroma_gradient_weight=0.80,
                hard_pixel_weight=1.20,
                spurious_edge_weight=2.0,
                gamut_weight=0.40,
                hard_fraction=0.10,
            )
            error = color_sample_error(output.image.rgb, target)
            legacy_error = color_sample_error(output.legacy_image.rgb, target)
            direct_error = color_sample_error(output.direct_image.rgb, target)
            parent_error = torch.minimum(legacy_error, direct_error).detach()
            guard = improvement_loss(error, parent_error, args.guard_margin)
            hard = cvar(error)
            order = torch.roll(torch.arange(len(target), device=device), 1)
            wrong_error = color_sample_error(output.image.rgb, target[order])
            ranking = (
                shuffled_ranking_loss(error, wrong_error, args.rank_margin)
                if len(target) > 1
                else error.mean() * 0
            )
            spatial = spatial_color_terms(output.image.rgb, target)
            routing = safe_routing_terms(output, target)
            spatial_loss = (
                1.2 * spatial["salient_chroma_mae"].mean()
                + 0.20 * spatial["chroma_centroid_error"].mean()
                + 0.45 * spatial["layout4_oklab_mae"].mean()
                + 0.65 * spatial["layout8_oklab_mae"].mean()
                + 0.25 * (1 - spatial["salient_chroma_recall"]).mean()
            )
            oracle_chroma = F.smooth_l1_loss(
                output.image.oklab[:, 1:], teacher.oklab[:, 1:], beta=0.012
            )
            oracle_layout = F.smooth_l1_loss(
                F.adaptive_avg_pool2d(output.image.oklab, 8),
                F.adaptive_avg_pool2d(teacher.oklab, 8),
                beta=0.02,
            )
            rare = rare_chroma_loss(output.image.chroma_logits16, target)
            usage = output.expert_weights.mean((0, 1, 3, 4))
            router_floor = F.relu(0.06 - usage).sum()
            router_tv = (
                output.expert_weights[..., 1:, :] - output.expert_weights[..., :-1, :]
            ).abs().mean() + (
                output.expert_weights[..., :, 1:] - output.expert_weights[..., :, :-1]
            ).abs().mean()
            target_std = target.flatten(1).std(0, unbiased=False).mean()
            output_std = output.image.rgb.flatten(1).std(0, unbiased=False).mean()
            diversity = F.relu(0.80 * target_std - output_std)
            representation = output.condition.grid16.mean((1, 2))
            variance, covariance = variance_covariance_regularization(representation)
            loss = (
                reconstruction
                + 0.75 * error.mean()
                + 0.80 * hard
                + 1.80 * guard
                + 0.45 * ranking
                + 0.80 * spatial_loss
                + 1.20 * routing["routing_nll"]
                + 1.50 * routing["pixel_guard_cvar10"]
                + 0.60 * routing["proposal_guard_cvar10"]
                + 0.35 * routing["proposal_reconstruction"]
                + 0.45 * oracle_chroma
                + 0.30 * oracle_layout
                + 0.12 * rare
                + 0.65 * diversity
                + 0.08 * router_floor
                + 0.08 * router_tv
                + 0.10 * variance
                + 0.02 * covariance
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.fusion.parameters(), 3.0)
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema, model, args.ema_decay)
        values = {
            "loss": loss,
            "task": error.mean(),
            "hard": hard,
            "guard": guard,
            "legacy_task": legacy_error.mean(),
            "direct_task": direct_error.mean(),
            "ranking": ranking,
            "spatial": spatial_loss,
            "oracle_chroma": oracle_chroma,
            "oracle_layout": oracle_layout,
            "rare_chroma": rare,
            "diversity": diversity,
            "router_floor": router_floor,
            "router_tv": router_tv,
            "representation_variance": variance,
            "representation_covariance": covariance,
            "route_legacy": usage[0],
            "route_direct": usage[1],
            "route_proposal": usage[2],
            **{key: value.mean() for key, value in spatial.items()},
            **routing,
            **parts,
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach())
        batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


@torch.no_grad()
def predict(model, loader, device, hard=False):
    model.eval()
    values = {
        key: [] for key in ("hybrid", "legacy", "direct", "oklab", "linear", "weights", "target")
    }
    names = []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            output = model(eeg, hard_chroma=hard)
        values["hybrid"].append(output.image.rgb.float().cpu())
        values["legacy"].append(output.legacy_image.rgb.float().cpu())
        values["direct"].append(output.direct_image.rgb.float().cpu())
        values["oklab"].append(output.image.oklab.float().cpu())
        values["linear"].append(output.image.linear_rgb.float().cpu())
        values["weights"].append(output.expert_weights.float().cpu())
        values["target"].append(batch["rgb"].float())
        names.extend(batch["filename"])
    return {key: torch.cat(value) for key, value in values.items()}, names


def evaluate(model, loader, device, train_mean, seed=None, hard=False):
    output, _ = predict(model, loader, device, hard)
    metrics = image_metrics(
        output["hybrid"], output["target"], train_mean, output["oklab"], output["linear"]
    )
    parents = {
        "legacy": image_metrics(output["legacy"], output["target"], train_mean),
        "direct": image_metrics(output["direct"], output["target"], train_mean),
    }
    usage = output["weights"].mean((0, 1, 3, 4))
    metrics.update(
        {
            f"route_{name}": float(usage[i])
            for i, name in enumerate(("legacy", "direct", "proposal"))
        }
    )
    if seed is None:
        return metrics, parents
    order = torch.randperm(len(output["hybrid"]), generator=torch.Generator().manual_seed(seed))
    shuffled = image_metrics(output["hybrid"][order], output["target"], train_mean)
    return metrics, parents, shuffled


def hybrid_score(value, shuffled, parents):
    best_rgb = min(parent["rgb_mae"] for parent in parents.values())
    best_chroma = min(parent["chroma_mae"] for parent in parents.values())
    best_palette = max(parent["palette_pearson"] for parent in parents.values())
    best_salient = min(parent["salient_chroma_mae"] for parent in parents.values())
    parent_artifact = max(parent["chroma_edge_artifact"] for parent in parents.values())
    return (
        selection_score(value, shuffled)
        + 2.0 * (best_rgb - value["rgb_mae"])
        + 3.0 * (best_chroma - value["chroma_mae"])
        + 0.8 * (value["palette_pearson"] - best_palette)
        + 1.2 * (best_salient - value["salient_chroma_mae"])
        + 0.3 * value["salient_chroma_recall"]
        - 0.2 * value["chroma_centroid_error"]
        - 5000.0 * max(0.0, value["chroma_edge_artifact"] - parent_artifact)
    )


def qualifies(value, parents):
    return (
        value["rgb_mae"] <= min(parent["rgb_mae"] for parent in parents.values()) + 0.003
        and value["chroma_mae"] <= min(parent["chroma_mae"] for parent in parents.values()) + 0.0015
        and value["palette_pearson"]
        >= max(parent["palette_pearson"] for parent in parents.values()) - 0.02
        and value["between_image_std_ratio"]
        >= max(parent["between_image_std_ratio"] for parent in parents.values()) - 0.08
        and value["salient_chroma_mae"]
        <= min(parent["salient_chroma_mae"] for parent in parents.values()) + 0.004
        and value["chroma_edge_artifact"]
        <= max(parent["chroma_edge_artifact"] for parent in parents.values()) + 2e-6
    )


def routing_image(weights: torch.Tensor, tile: int) -> Image.Image:
    # Red=legacy, green=direct, blue=fused proposal.
    rgb = weights.mean(0).clamp(0, 1)
    return to_image(rgb, tile)


@torch.no_grad()
def save_panel(model, oracle, dataset, device, path, count):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([sample["eeg"] for sample in samples]).to(device)
    grid = torch.stack([sample["grid"] for sample in samples]).float().to(device)
    glob = torch.stack([sample["global"] for sample in samples]).float().to(device)
    model.eval()
    soft = model(eeg)
    hard = model(eeg, hard_chroma=True)
    teacher = oracle.decode_full_latent(FullColorLatent(grid, glob))
    headers = ("target", "legacy", "direct", "hybrid", "hybrid argmax", "oracle", "routing RGB")
    tile, label = 120, 22
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples):
        y = (row + 1) * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        images = (
            to_image(sample["rgb"], tile),
            to_image(soft.legacy_image.rgb[row].cpu(), tile),
            to_image(soft.direct_image.rgb[row].cpu(), tile),
            to_image(soft.image.rgb[row].cpu(), tile),
            to_image(hard.image.rgb[row].cpu(), tile),
            to_image(teacher.rgb[row].cpu(), tile),
            routing_image(soft.expert_weights[row].cpu(), tile),
        )
        for column, image in enumerate(images):
            canvas.paste(image, (column * tile, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main() -> None:
    args = arguments()
    if args.smoke:
        args.epochs, args.batch_size, args.eval_batch_size, args.panel_images = 1, 2, 4, 2
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    latent_root = Path(args.latent_bank)
    manifest = json.loads((latent_root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError("full color latent bank is incomplete")
    arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "training")
        for subject in range(10)
    ]
    records = {
        split: split_records(Path(args.index), split, manifest["filenames"])
        for split in ("train", "val", "test")
    }
    if args.smoke:
        records = {key: value[:16] for key, value in records.items()}
    datasets = {
        key: EEGColorDataset(arrays, value, latent_root, Path(args.rich_bank))
        for key, value in records.items()
    }
    loaders = {
        key: DataLoader(
            value,
            batch_size=args.batch_size if key == "train" else args.eval_batch_size,
            shuffle=key == "train",
            num_workers=0,
            drop_last=key == "train",
        )
        for key, value in datasets.items()
    }
    train_mean = training_rgb_mean(datasets["train"])
    oracle, oracle_payload = load_color_oracle(Path(args.oracle_renderer), device)
    oracle.eval()
    for parameter in oracle.parameters():
        parameter.requires_grad_(False)
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        model = EEGHybridColorEncoder(**resumed["model_config"]).to(device)
        model.load_state_dict(resumed["model"], strict=True)
    else:
        resumed = None
        model = initialize_hybrid_color_encoder(args.direct_root, args.legacy_root, device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("fusion."))
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.fusion.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=2e-6
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    start, best, best_qualified, stale = 0, -math.inf, -math.inf, 0
    if resumed:
        ema.load_state_dict(resumed["ema"])
        optimizer.load_state_dict(resumed["optimizer"])
        scheduler.load_state_dict(resumed["scheduler"])
        if "scaler" in resumed:
            scaler.load_state_dict(resumed["scaler"])
        start = resumed["epoch"] + 1
        best = resumed["best_score"]
        best_qualified = resumed.get("best_qualified_score", -math.inf)
        stale = resumed.get("stale", 0)
        if resumed.get("objective_version") != 2:
            best, best_qualified, stale = -math.inf, -math.inf, 0
            scheduler.best = scheduler.mode_worse
            scheduler.num_bad_epochs = 0
            scheduler.cooldown_counter = 0
            print("routing objective changed: checkpoint selection reset for safe v2", flush=True)
    history = output / "history.jsonl"
    if not resumed:
        history.unlink(missing_ok=True)
    for epoch in range(start, args.epochs):
        train = train_epoch(model, ema, oracle, loaders["train"], optimizer, scaler, device, args)
        validation, parents, shuffled = evaluate(ema, loaders["val"], device, train_mean, args.seed)
        hard, _ = evaluate(ema, loaders["val"], device, train_mean, hard=True)
        current = hybrid_score(validation, shuffled, parents)
        qualified = qualifies(validation, parents)
        improved = current > best + 1e-5
        qualified_improved = qualified and current > best_qualified + 1e-5
        stale = 0 if improved else stale + 1
        scheduler.step(current)
        record = {
            "epoch": epoch,
            "score": current,
            "qualified": qualified,
            "train": train,
            "validation": validation,
            "validation_hard": hard,
            "parents": parents,
            "shuffled": shuffled,
            "lr": optimizer.param_groups[0]["lr"],
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "model_config": model.config(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_score": max(best, current),
            "best_qualified_score": max(best_qualified, current) if qualified else best_qualified,
            "stale": stale,
            "record": record,
            "architecture": "independent_hybrid_color_v1",
            "objective_version": 2,
            "oracle_renderer": args.oracle_renderer,
            "oracle_epoch": oracle_payload.get("epoch"),
            "latent_bank_manifest": manifest,
        }
        torch.save(payload, output / "last.pt")
        if improved:
            best = current
            torch.save(payload, output / "best.pt")
        if qualified_improved:
            best_qualified = current
            torch.save(payload, output / "best_qualified.pt")
        save_panel(
            ema,
            oracle,
            datasets["val"],
            device,
            output / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} "
            f"qualified={qualified} rgb={validation['rgb_mae']:.3f} "
            f"palette={validation['palette_pearson']:.3f} salient={validation['salient_chroma_mae']:.3f} "
            f"routes={validation['route_legacy']:.2f}/{validation['route_direct']:.2f}/"
            f"{validation['route_proposal']:.2f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    qualified_path = output / "best_qualified.pt"
    qualified_state = (
        torch.load(qualified_path, map_location="cpu", weights_only=False)
        if qualified_path.exists()
        else None
    )
    safely_qualified = bool(
        qualified_state
        and qualified_state.get("objective_version") == 2
        and qualified_state.get("record", {}).get("qualified")
    )
    selected_path = qualified_path if safely_qualified else output / "best.pt"
    selected = (
        qualified_state
        if safely_qualified
        else torch.load(selected_path, map_location="cpu", weights_only=False)
    )
    ema.load_state_dict(selected["ema"])
    ema.to(device)
    test, parents, shuffled = evaluate(ema, loaders["test"], device, train_mean, args.seed + 999)
    report = {
        "checkpoint": selected_path.name,
        "checkpoint_epoch": selected["epoch"],
        "qualified": selected["record"].get("qualified", False),
        "hybrid": test,
        "parents": parents,
        "shuffled": shuffled,
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
        "inference_inputs": ["preprocessed_eeg"],
        "architecture": "independent_hybrid_color_v1",
    }
    (output / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
