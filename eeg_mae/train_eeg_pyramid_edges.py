from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from .build_pyramid_edge_targets import PYRAMID_KEYS
from .model_utils import grid_contrastive, grid_structure_losses
from .official_200way_benchmark import cached_array
from .pyramid_edge import PyramidEdgeOutput, load_pyramid_edge_decoder, pyramid_edge_loss
from .semantic_edge import EEGDINOGridPredictor
from .semantic_encoder import load_semantic_encoder
from .train_pyramid_edge_oracle import metrics as pyramid_metrics


def arguments():
    parser = argparse.ArgumentParser(
        description="Safely refine the strong EEG grid predictor against multi-scale edge targets"
    )
    parser.add_argument("--semantic-encoder", default="runs/eeg_semantic_encoder_v2/best.pt")
    parser.add_argument(
        "--resume",
        default=None,
        help="optional spatial-encoder checkpoint; omit to train from the semantic encoder",
    )
    parser.add_argument("--pyramid-decoder", default="runs/dino_pyramid_edge_oracle_v1/best.pt")
    parser.add_argument("--visual-bank", default="data/derived/visual_targets_dinov2s_192.pt")
    parser.add_argument("--target-bank", default="data/derived/pyramid_edge_targets_v1.pt")
    parser.add_argument(
        "--index", default="data/things_eeg2_osf/preprocessed_train_all_subjects_holdout_index.csv"
    )
    parser.add_argument(
        "--images-zip", default="data/things_eeg2_osf/image_set/training_images.zip"
    )
    parser.add_argument("--archives", default="data/things_eeg2_osf/preprocessed")
    parser.add_argument("--cache", default="data/derived/eeg_float32_cache")
    parser.add_argument("--output", default="runs/eeg_pyramid_edges_v1")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--spatial-lr", type=float, default=3e-6)
    parser.add_argument("--pyramid-loss-weight", type=float, default=0.18)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--panel-images", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def split_records(index_path, split, visual, target):
    visual_lookup = {name: i for i, name in enumerate(visual["filenames"])}
    target_lookup = {name: i for i, name in enumerate(target["filenames"])}
    records = {}
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) != 0 or row["split"] != split:
                continue
            name, condition = row["image_file"], int(row["image_index"])
            vi, ti = visual_lookup.get(name), target_lookup.get(name)
            if vi is not None and ti is not None and bool(target["valid"][ti]):
                records[condition] = (condition, vi, ti, name)
    return [records[key] for key in sorted(records)]


def all_records(index_path, visual, target):
    """Deterministic fallback used only by a partial-bank smoke test."""
    visual_lookup = {name: i for i, name in enumerate(visual["filenames"])}
    target_lookup = {name: i for i, name in enumerate(target["filenames"])}
    records = {}
    with Path(index_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["subject_id"]) != 0:
                continue
            name, condition = row["image_file"], int(row["image_index"])
            vi, ti = visual_lookup.get(name), target_lookup.get(name)
            if vi is not None and ti is not None and bool(target["valid"][ti]):
                records[(row["split"], condition)] = (condition, vi, ti, name)
    return [records[key] for key in sorted(records)]


class EEGPyramidDataset(Dataset):
    def __init__(self, arrays, records, visual_grid, target):
        self.arrays, self.records, self.grid, self.target = arrays, records, visual_grid, target

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        condition, vi, ti, name = self.records[index]
        eeg = np.stack(
            [np.asarray(array[condition], dtype=np.float32).mean(0) for array in self.arrays]
        )
        return {
            "eeg": torch.from_numpy(eeg),
            "grid": self.grid[vi],
            "filename": name,
            **{key: self.target[key][ti].float() for key in PYRAMID_KEYS},
        }


def concatenate(outputs):
    return PyramidEdgeOutput(
        **{
            field: torch.cat([getattr(output, field) for output in outputs])
            for field in PyramidEdgeOutput.__dataclass_fields__
        }
    )


def shuffled_metrics(output, target, seed):
    order = torch.randperm(
        len(output.shape_logits16), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = PyramidEdgeOutput(
        **{field: getattr(output, field)[order] for field in PyramidEdgeOutput.__dataclass_fields__}
    )
    return pyramid_metrics(shuffled, target)


@torch.no_grad()
def evaluate(predictor, decoder, loader, device, seed=0):
    predictor.eval()
    decoder.eval()
    outputs = []
    targets = {key: [] for key in PYRAMID_KEYS}
    grids, target_grids = [], []
    for batch in loader:
        eeg = batch["eeg"].to(device)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            grid, _ = predictor.forward_ensemble(eeg)
            output = decoder(grid)
        outputs.append(
            PyramidEdgeOutput(
                **{
                    field: getattr(output, field).float().cpu()
                    for field in PyramidEdgeOutput.__dataclass_fields__
                }
            )
        )
        for key in PYRAMID_KEYS:
            targets[key].append(batch[key].float())
        grids.append(grid.float().cpu())
        target_grids.append(batch["grid"].float())
    if not outputs:
        return {"empty": True}, {"empty": True}, None, None
    output = concatenate(outputs)
    target = {key: torch.cat(values) for key, values in targets.items()}
    result = pyramid_metrics(output, target)
    prediction_grid, target_grid = torch.cat(grids), torch.cat(target_grids)
    result["grid_cosine"] = float(F.cosine_similarity(prediction_grid, target_grid, dim=-1).mean())
    pred_residual = F.normalize(
        prediction_grid - prediction_grid.mean((1, 2), keepdim=True), dim=-1
    )
    true_residual = F.normalize(target_grid - target_grid.mean((1, 2), keepdim=True), dim=-1)
    result["grid_residual_cosine"] = float(
        F.cosine_similarity(pred_residual, true_residual, dim=-1).mean()
    )
    shuffled = shuffled_metrics(output, target, seed)
    return result, shuffled, output, target


def specificity(result, shuffled):
    return {
        key: result[key] - shuffled[key]
        for key in (
            "semantic_tolerant_f1",
            "crisp_tolerant_f1",
            "perceptual_pearson",
            "distance_correlation",
        )
    }


def score(result, shuffled):
    gap = specificity(result, shuffled)
    return (
        result["grid_cosine"]
        + result["grid_residual_cosine"]
        + result["shape_iou"]
        + result["semantic_tolerant_f1"]
        + result["crisp_tolerant_f1"]
        + result["perceptual_pearson"]
        + result["distance_correlation"]
        + 0.75 * sum(gap.values())
    )


def train_epoch(predictor, decoder, loader, optimizer, device, pyramid_weight):
    predictor.train()
    decoder.eval()
    totals = {}
    batches = 0
    for batch in loader:
        eeg = batch["eeg"].to(device)
        target_grid = batch["grid"].to(device)
        target = {key: batch[key].to(device) for key in PYRAMID_KEYS}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.float16, enabled=device.type == "cuda"):
            grid, contexts = predictor.forward_ensemble(eeg)
            local = (1 - F.cosine_similarity(grid, target_grid, dim=-1)).mean()
            residual, gradients = grid_structure_losses(grid, target_grid)
            contrastive = grid_contrastive(grid, target_grid)
            chosen = torch.randint(contexts.shape[1], (len(eeg),), device=device)
            context = contexts[torch.arange(len(eeg), device=device), chosen]
            individual = predictor.decode_context(context)
            individual_loss = (1 - F.cosine_similarity(individual, target_grid, dim=-1)).mean()
            edge_loss, edge_parts = pyramid_edge_loss(decoder(grid), target)
            loss = (
                1.35 * local
                + 1.0 * residual
                + 0.4 * gradients
                + 0.5 * contrastive
                + 0.2 * individual_loss
                + pyramid_weight * edge_loss
            )
        loss.backward()
        nn.utils.clip_grad_norm_((p for p in predictor.parameters() if p.requires_grad), 5.0)
        optimizer.step()
        batches += 1
        values = {
            "loss": float(loss.detach()),
            "grid": float(local.detach()),
            "residual": float(residual.detach()),
            "gradient": float(gradients.detach()),
            "contrastive": float(contrastive.detach()),
            "pyramid": float(edge_loss.detach()),
            **{f"pyramid_{key}": float(value.detach()) for key, value in edge_parts.items()},
        }
        for key, value in values.items():
            totals[key] = totals.get(key, 0.0) + value
    return {key: value / max(batches, 1) for key, value in totals.items()}


def gray(value, size=128):
    array = (value.squeeze().float().clamp(0, 1) * 255).to(torch.uint8).numpy()
    return Image.fromarray(array, "L").resize((size, size), Image.Resampling.NEAREST).convert("RGB")


def source_images(samples, images_zip, size):
    with zipfile.ZipFile(images_zip) as archive:
        members = {Path(member).name: member for member in archive.namelist()}
        result = []
        for sample in samples:
            with Image.open(io.BytesIO(archive.read(members[sample["filename"]]))) as image:
                result.append(image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS))
    return result


@torch.no_grad()
def save_panel(predictor, decoder, dataset, device, path, count, images_zip):
    samples = [dataset[index] for index in range(min(count, len(dataset)))]
    if not samples:
        return
    eeg = torch.stack([sample["eeg"] for sample in samples]).to(device)
    predictor.eval()
    decoder.eval()
    grid, _ = predictor.forward_ensemble(eeg)
    output = decoder(grid)
    originals = source_images(samples, images_zip, 128)
    headers = (
        "original GT",
        "shape GT",
        "shape EEG",
        "semantic GT",
        "semantic EEG",
        "crisp GT",
        "crisp EEG",
        "confidence GT",
        "confidence EEG",
    )
    tile, label = 128, 22
    canvas = Image.new("RGB", (len(headers) * tile, (len(samples) + 1) * (tile + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((column * tile + 3, 3), header, fill="black")
    for row, sample in enumerate(samples, 1):
        y = row * (tile + label)
        draw.text((3, y - label + 3), sample["filename"], fill="black")
        images = (
            originals[row - 1],
            gray(sample["shape16"]),
            gray(output.shape_logits16.sigmoid().cpu()[row - 1]),
            gray(sample["semantic32"]),
            gray(output.semantic_logits32.sigmoid().cpu()[row - 1]),
            gray(sample["crisp64"]),
            gray(output.crisp_logits64.sigmoid().cpu()[row - 1]),
            gray(sample["confidence64"]),
            gray(output.confidence64.cpu()[row - 1]),
        )
        for column, image in enumerate(images):
            canvas.paste(image, (column * tile, y))
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
    target = torch.load(args.target_bank, map_location="cpu", weights_only=False)
    visual_grid = F.normalize(visual["dino_grid"].float(), dim=-1)
    encoder = load_semantic_encoder(args.semantic_encoder, device)
    if args.resume:
        resumed = torch.load(args.resume, map_location="cpu", weights_only=False)
        predictor = EEGDINOGridPredictor(encoder, **resumed["model_config"]).to(device)
        predictor.load_state_dict(resumed["model"], strict=True)
    else:
        predictor = EEGDINOGridPredictor(encoder, grid_dim=visual_grid.shape[-1]).to(device)
        resumed = {"model": predictor.state_dict(), "model_config": predictor.config()}
    decoder, decoder_state = load_pyramid_edge_decoder(args.pyramid_decoder, device)
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    arrays = [
        cached_array(Path(args.archives), Path(args.cache), sid, "training") for sid in range(10)
    ]
    records = {
        split: split_records(args.index, split, visual, target)
        for split in ("train", "val", "test")
    }
    if args.smoke:
        if any(not value for value in records.values()):
            available = all_records(args.index, visual, target)
            first = max(1, int(0.70 * len(available)))
            second = max(first + 1, int(0.85 * len(available)))
            records = {
                "train": available[:first],
                "val": available[first:second],
                "test": available[second:],
            }
        records = {key: value[:32] for key, value in records.items()}
    datasets = {
        key: EEGPyramidDataset(arrays, value, visual_grid, target) for key, value in records.items()
    }
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
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    baseline, baseline_shuffled, _, _ = evaluate(
        predictor, decoder, loaders["val"], device, args.seed
    )
    if baseline.get("empty"):
        raise RuntimeError("validation split has no valid pyramid targets")
    baseline_score = score(baseline, baseline_shuffled)
    baseline_report = {
        "score": baseline_score,
        "validation": baseline,
        "validation_shuffled": baseline_shuffled,
        "specificity_gap": specificity(baseline, baseline_shuffled),
    }
    (output_dir / "baseline_validation.json").write_text(
        json.dumps(baseline_report, indent=2), encoding="utf-8"
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
    other = [p for p in predictor.parameters() if p.requires_grad and id(p) not in spatial_ids]
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
        f"baseline score={best:.4f} semantic={baseline['semantic_tolerant_f1']:.3f} "
        f"crisp={baseline['crisp_tolerant_f1']:.3f}",
        flush=True,
    )
    for epoch in range(args.epochs):
        begun = time.perf_counter()
        train = train_epoch(
            predictor, decoder, loaders["train"], optimizer, device, args.pyramid_loss_weight
        )
        val, shuffled, _, _ = evaluate(
            predictor, decoder, loaders["val"], device, args.seed + epoch + 1
        )
        current = score(val, shuffled)
        improved = current > best + 1e-5
        stale = 0 if improved else stale + 1
        scheduler.step(current)
        record = {
            "epoch": epoch,
            "seconds": time.perf_counter() - begun,
            "score": current,
            "improved_over_original": current > baseline_score,
            "train": train,
            "validation": val,
            "validation_shuffled": shuffled,
            "specificity_gap": specificity(val, shuffled),
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
            "pyramid_decoder": args.pyramid_decoder,
            "pyramid_decoder_epoch": decoder_state.get("epoch"),
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
            args.images_zip,
        )
        print(
            f"epoch={epoch:03d} loss={train['loss']:.3f} score={current:.3f} "
            f"semantic={val['semantic_tolerant_f1']:.3f} crisp={val['crisp_tolerant_f1']:.3f} "
            f"gap={specificity(val, shuffled)['semantic_tolerant_f1']:.3f}",
            flush=True,
        )
        if stale >= args.patience:
            break
    selected = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)
    predictor.load_state_dict(selected["model"], strict=True)
    predictor.to(device)
    result, shuffled, _, _ = evaluate(predictor, decoder, loaders["test"], device, args.seed + 999)
    report = {
        "checkpoint_epoch": selected["epoch"],
        "baseline_retained": selected.get("baseline_retained", False),
        "validation_baseline_score": baseline_score,
        "validation_selected_score": selected["best_score"],
        "eeg": result,
        "shuffled": shuffled,
        "specificity_gap": specificity(result, shuffled),
        "test_images": len(datasets["test"]),
        "untouched_full_test": not args.smoke,
    }
    (output_dir / "test_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
