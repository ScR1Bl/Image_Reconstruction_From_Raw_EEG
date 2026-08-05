"""Specialize the spatial EEG encoder for low-frequency image color."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .color import ColorOutput, color_loss, color_metrics, load_color_decoder
from .model_utils import grid_contrastive, grid_structure_losses
from .official_200way_benchmark import cached_array
from .semantic_edge import EEGDINOGridPredictor
from .semantic_encoder import load_semantic_encoder


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EEG encoder specialized for color")
    parser.add_argument("--semantic-encoder", default="runs/eeg_semantic_encoder_v2/best.pt")
    parser.add_argument(
        "--resume",
        default="runs/eeg_pyramid_edges_v1/best.pt",
        help="spatial EEG checkpoint used to initialize the independent color encoder",
    )
    parser.add_argument("--color-decoder", default="runs/dino_color_oracle_v1/best.pt")
    parser.add_argument("--visual-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--rgb-bank", default="data/derived/rich_visual_targets_32.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--output", default="runs/eeg_color_encoder_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--spatial-lr", type=float, default=3e-6)
    parser.add_argument("--color-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def split_records(index_path: Path, split: str, visual: dict, rich: dict):
    visual_lookup = {name: index for index, name in enumerate(visual["filenames"])}
    rich_lookup = {name: index for index, name in enumerate(rich["filenames"])}
    records = {}
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) != 0 or row["split"] != split:
                continue
            name, condition = row["image_file"], int(row["image_index"])
            if name in visual_lookup and name in rich_lookup:
                records[condition] = (condition, visual_lookup[name], rich_lookup[name], name)
    return [records[key] for key in sorted(records)]


class EEGColorDataset(Dataset):
    def __init__(self, arrays, records, visual, rich):
        self.arrays, self.records = arrays, records
        self.grid, self.global_feature = visual["dino_grid"], visual["dino_global"]
        self.descriptor, self.rgb = visual["color"], rich["rgb"]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        condition, visual_index, rich_index, name = self.records[index]
        eeg = np.stack(
            [np.asarray(array[condition], dtype=np.float32).mean(0) for array in self.arrays]
        )
        return {
            "eeg": torch.from_numpy(eeg),
            "grid": self.grid[visual_index].float(),
            "global": self.global_feature[visual_index].float(),
            "descriptor": self.descriptor[visual_index].float(),
            "rgb": self.rgb[rich_index].float(),
            "filename": name,
        }


def semantic_ensemble(predictor: EEGDINOGridPredictor, eeg: torch.Tensor) -> torch.Tensor:
    batch, subjects = eeg.shape[:2]
    subject = torch.arange(subjects, device=eeg.device).repeat(batch)
    with torch.no_grad():
        values = predictor.encoder(eeg.flatten(0, 1), subject)
    return F.normalize(values.reshape(batch, subjects, -1).mean(1), dim=-1)


def train_epoch(predictor, decoder, data, optimizer, device, color_weight):
    predictor.train()
    decoder.eval()
    totals = {}
    batches = 0
    for batch in data:
        eeg = batch["eeg"].to(device)
        target_grid = F.normalize(batch["grid"].to(device), dim=-1)
        rgb = batch["rgb"].to(device)
        descriptor = batch["descriptor"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            grid, contexts = predictor.forward_ensemble(eeg)
            semantic = semantic_ensemble(predictor, eeg)
            output = decoder(grid, semantic)
            reconstruction, color_parts = color_loss(output, rgb, descriptor)
            local = (1 - F.cosine_similarity(grid, target_grid, dim=-1)).mean()
            residual, gradients = grid_structure_losses(grid, target_grid)
            contrastive = grid_contrastive(grid, target_grid)
            chosen = torch.randint(contexts.shape[1], (len(eeg),), device=device)
            context = contexts[torch.arange(len(eeg), device=device), chosen]
            individual = (
                1 - F.cosine_similarity(predictor.decode_context(context), target_grid, dim=-1)
            ).mean()
            loss = (
                1.2 * local
                + residual
                + 0.4 * gradients
                + 0.5 * contrastive
                + 0.2 * individual
                + color_weight * reconstruction
            )
        loss.backward()
        nn.utils.clip_grad_norm_(
            (parameter for parameter in predictor.parameters() if parameter.requires_grad), 5.0
        )
        optimizer.step()
        batches += 1
        values = {
            "loss": float(loss.detach()),
            "grid": float(local.detach()),
            "residual": float(residual.detach()),
            "gradient": float(gradients.detach()),
            "contrastive": float(contrastive.detach()),
            "individual": float(individual.detach()),
            **{f"color_{key}": float(value.detach()) for key, value in color_parts.items()},
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / max(batches, 1) for key, value in totals.items()}


@torch.no_grad()
def predict(predictor, decoder, data, device):
    predictor.eval()
    decoder.eval()
    rgb_predictions, descriptors, rgbs = [], [], []
    grid_cosines = []
    for batch in data:
        eeg = batch["eeg"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            grid, _ = predictor.forward_ensemble(eeg)
            output = decoder(grid, semantic_ensemble(predictor, eeg))
        rgb_predictions.append(output.rgb32.float().cpu())
        descriptors.append(output.descriptor.float().cpu())
        rgbs.append(batch["rgb"].float())
        grid_cosines.append(
            F.cosine_similarity(
                grid.float().cpu(), F.normalize(batch["grid"].float(), dim=-1), dim=-1
            ).mean()
        )
    return (
        ColorOutput(torch.cat(rgb_predictions), torch.cat(descriptors)),
        torch.cat(rgbs),
        torch.cat([batch["descriptor"].float() for batch in data]),
        float(torch.stack(grid_cosines).mean()),
    )


def evaluate(predictor, decoder, data, device, train_mean, shuffle_seed=None):
    output, rgb, descriptor, grid_cosine = predict(predictor, decoder, data, device)
    if shuffle_seed is not None:
        order = torch.randperm(len(rgb), generator=torch.Generator().manual_seed(shuffle_seed))
        output = ColorOutput(output.rgb32[order], output.descriptor[order])
    metrics = color_metrics(output, rgb, descriptor, train_mean)
    metrics["grid_cosine"] = grid_cosine
    return metrics


def score(metrics, shuffled):
    specificity = (
        metrics["palette_pearson"]
        - shuffled["palette_pearson"]
        + metrics["descriptor_cosine"]
        - shuffled["descriptor_cosine"]
    )
    return (
        metrics["grid_cosine"]
        + metrics["palette_pearson"]
        + metrics["descriptor_cosine"]
        + metrics.get("rgb_gain_vs_train_mean", 0)
        + specificity
        - metrics["rgb_mae"]
        - 1.5 * metrics["chroma_mae"]
        + 0.25 * min(metrics["between_image_std_ratio"], 1.0)
    )


@torch.no_grad()
def save_panel(predictor, decoder, dataset, device, path, count):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([sample["eeg"] for sample in samples]).to(device)
    predictor.eval()
    decoder.eval()
    grid, _ = predictor.forward_ensemble(eeg)
    output = decoder(grid, semantic_ensemble(predictor, eeg))
    tile, label = 160, 22
    canvas = Image.new("RGB", (2 * tile, len(samples) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, sample in enumerate(samples):
        y = row * (tile + label)
        draw.text((3, y + 3), sample["filename"], fill="black")
        for column, tensor in enumerate((sample["rgb"], output.rgb32[row].cpu())):
            array = (tensor.permute(1, 2, 0).float().clamp(0, 1) * 255).byte().numpy()
            canvas.paste(
                Image.fromarray(array).resize((tile, tile), Image.Resampling.NEAREST),
                (column * tile, y + label),
            )
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
    encoder = load_semantic_encoder(args.semantic_encoder, device)
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        predictor = EEGDINOGridPredictor(encoder, **resumed["model_config"]).to(device)
        predictor.load_state_dict(resumed["model"], strict=True)
    else:
        predictor = EEGDINOGridPredictor(encoder, grid_dim=visual["dino_grid"].shape[-1]).to(device)
        resumed = {"model": predictor.state_dict(), "model_config": predictor.config()}
    decoder, decoder_state = load_color_decoder(args.color_decoder, device)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    arrays = [
        cached_array(Path(args.archives), Path(args.cache), subject, "training")
        for subject in range(10)
    ]
    records = {
        split: split_records(Path(args.index), split, visual, rich)
        for split in ("train", "val", "test")
    }
    if args.smoke:
        records = {key: value[:32] for key, value in records.items()}
    datasets = {key: EEGColorDataset(arrays, value, visual, rich) for key, value in records.items()}
    loaders = {
        key: DataLoader(
            value,
            batch_size=args.batch_size if key == "train" else args.eval_batch_size,
            shuffle=key == "train",
            num_workers=0,
            drop_last=False,
        )
        for key, value in datasets.items()
    }
    train_rgb_indices = [row[2] for row in records["train"]]
    train_mean = rich["rgb"][train_rgb_indices].float().mean(0, keepdim=True)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    baseline = evaluate(predictor, decoder, loaders["val"], device, train_mean)
    baseline_shuffled = evaluate(predictor, decoder, loaders["val"], device, train_mean, args.seed)
    baseline_score = score(baseline, baseline_shuffled)
    (output_dir / "baseline_validation.json").write_text(
        json.dumps(
            {
                "score": baseline_score,
                "validation": baseline,
                "validation_shuffled": baseline_shuffled,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    best_payload = {
        "epoch": -1,
        "model": resumed["model"],
        "model_config": resumed["model_config"],
        "best_score": baseline_score,
        "source_checkpoint": args.resume,
        "baseline_retained": True,
    }
    torch.save(best_payload, output_dir / "best.pt")
    spatial = [
        predictor.spatial_norm_weight,
        predictor.spatial_norm_bias,
        predictor.spatial_weight,
        predictor.spatial_bias,
    ]
    spatial_ids = {id(value) for value in spatial}
    other = [
        parameter
        for parameter in predictor.parameters()
        if parameter.requires_grad and id(parameter) not in spatial_ids
    ]
    optimizer = torch.optim.AdamW(
        [{"params": spatial, "lr": args.spatial_lr}, {"params": other, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=3e-7
    )
    history = output_dir / "history.jsonl"
    history.unlink(missing_ok=True)
    best, stale = baseline_score, 0
    print(
        f"baseline score={best:.4f} palette={baseline['palette_pearson']:.3f} "
        f"gain={baseline['rgb_gain_vs_train_mean']:.3f}",
        flush=True,
    )
    for epoch in range(args.epochs):
        train = train_epoch(
            predictor, decoder, loaders["train"], optimizer, device, args.color_loss_weight
        )
        validation = evaluate(predictor, decoder, loaders["val"], device, train_mean)
        shuffled = evaluate(
            predictor, decoder, loaders["val"], device, train_mean, args.seed + epoch + 1
        )
        current = score(validation, shuffled)
        improved = current > best + 1e-5
        stale = 0 if improved else stale + 1
        scheduler.step(current)
        record = {
            "epoch": epoch,
            "score": current,
            "train": train,
            "validation": validation,
            "validation_shuffled": shuffled,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        payload = {
            "epoch": epoch,
            "model": predictor.state_dict(),
            "model_config": predictor.config(),
            "best_score": max(best, current),
            "source_checkpoint": args.resume,
            "baseline_score": baseline_score,
            "baseline_retained": False,
            "record": record,
            "color_decoder": args.color_decoder,
            "color_decoder_epoch": decoder_state.get("epoch"),
        }
        torch.save(payload, output_dir / "last.pt")
        if improved:
            best = current
            torch.save(payload, output_dir / "best.pt")
        save_panel(
            predictor,
            decoder,
            datasets["val"],
            device,
            output_dir / "panels" / f"epoch_{epoch:03d}.png",
            args.panel_images,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} "
            f"palette={validation['palette_pearson']:.3f} "
            f"gain={validation['rgb_gain_vs_train_mean']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    predictor.load_state_dict(selected["model"], strict=True)
    predictor.to(device)
    result = evaluate(predictor, decoder, loaders["test"], device, train_mean)
    shuffled = evaluate(predictor, decoder, loaders["test"], device, train_mean, args.seed + 999)
    report = {
        "checkpoint_epoch": selected["epoch"],
        "baseline_retained": selected.get("baseline_retained", False),
        "validation_baseline_score": baseline_score,
        "validation_selected_score": selected["best_score"],
        "eeg": result,
        "shuffled": shuffled,
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
    }
    (output_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
