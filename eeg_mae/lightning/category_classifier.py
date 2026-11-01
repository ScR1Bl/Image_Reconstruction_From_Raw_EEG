"""Supervised category head on frozen EEG features; training-image priors only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..data.build_category_prior import accuracy, category, training_labels
from ..data.dit_ten_subjects import OUTPUT, legacy_rows, load_archive_features
from ..lightning.common import sha256_of_file
from ..models.hierarchical_semantic import load_hierarchical_semantic_encoder


class CategoryClassifier(L.LightningModule):
    def __init__(self, dim=1792, classes=1654, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(dim, classes)
        self.validation = []

    def forward(self, features):
        return self.head(self.dropout((features - self.mean) / self.std))

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y, label_smoothing=0.1)
        self.log("train/loss", loss, on_step=False, on_epoch=True, batch_size=len(y))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.validation.append((self(x).detach().cpu(), y.cpu()))

    def on_validation_epoch_end(self):
        scores, labels = [torch.cat([item[i] for item in self.validation]) for i in range(2)]
        metrics = accuracy(scores, labels)
        self.log("val/top1", metrics["top1"])
        print(json.dumps({"epoch": self.current_epoch, **metrics}), flush=True)
        self.validation.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.1)


@torch.no_grad()
def encode_training(rows, checkpoint):
    encoder, _, _ = load_hierarchical_semantic_encoder(checkpoint, "cuda")
    encoder.eval().requires_grad_(False)
    arrays = [
        np.load(
            f"data/derived/eeg_float32_cache/sub-{sid + 1:02d}__preprocessed_eeg_training.npy",
            mmap_mode="r",
        )
        for sid in range(10)
    ]
    result = []
    for start in range(0, len(rows), 16):
        ids = [int(r["image_index"]) for r in rows[start : start + 16]]
        eeg = torch.from_numpy(
            np.stack([np.asarray(a[ids], dtype=np.float32).mean(1) for a in arrays], 1)
        ).cuda()
        with torch.autocast("cuda", torch.float16):
            value = encoder(
                eeg.flatten(0, 1), torch.arange(10, device="cuda").repeat(len(eeg))
            ).fused
        fused = F.normalize(value.float().reshape(len(eeg), 10, -1), dim=-1).mean(1)
        result.append(F.normalize(fused, dim=-1).cpu())
        if start % 1600 == 0:
            print(f"training EEG {start + len(eeg)}/{len(rows)}", flush=True)
    return torch.cat(result)


def select_reference(scores, query, bank):
    labels = bank["labels"]
    if any(r["split"] != "train" for r in bank["rows"]):
        raise ValueError("non-training image in reference bank")
    predicted = scores.argmax(1)
    similarity = (
        F.normalize(query.float(), dim=-1) @ F.normalize(bank["references"].float(), dim=-1).T
    )
    selected = similarity.masked_fill(labels[None] != predicted[:, None], -torch.inf).argmax(1)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior", type=Path, default=OUTPUT / "category_prior")
    parser.add_argument("--output", type=Path, default=OUTPUT / "category_classifier")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise ValueError("completed run exists; choose a fresh output")
    if (output / "checkpoints").exists() and not args.resume:
        raise ValueError("unfinished run exists; pass --resume")
    torch.set_num_threads(2)
    L.seed_everything(20260913)
    rows = legacy_rows()
    names, labels = training_labels(rows["train"])
    bank = torch.load(args.prior / "training_bank.pt", weights_only=False)
    val = torch.load(args.prior / "predictions.pt", weights_only=False)
    if bank["rows"] != rows["train"] or val["rows"] != rows["val"]:
        raise ValueError("feature index mismatch")
    checkpoint = Path(bank["metadata"]["checkpoint"])
    signature = {
        "checkpoint_sha256": sha256_of_file(checkpoint),
        "train_rows": rows["train"],
        "seed": 20260913,
    }
    cache = args.prior / "train_eeg.pt"
    if cache.exists():
        cached = torch.load(cache, weights_only=False)
        if cached["signature"] != signature:
            raise ValueError("training cache is stale")
        train = cached["features"]
    else:
        train = encode_training(rows["train"], checkpoint)
        torch.save({"features": train, "signature": signature}, cache)
    archive = load_archive_features()
    train_x = torch.cat([train, F.normalize(archive["train"].float().mean(1), dim=-1)], 1)
    val_x = torch.cat([val["predictions"], F.normalize(archive["val"].float().mean(1), dim=-1)], 1)
    lookup = {name: i for i, name in enumerate(names)}
    truth = torch.tensor([lookup[category(row)] for row in rows["val"]])
    model = CategoryClassifier(dim=train_x.shape[1], classes=len(names))
    model.mean.copy_(train_x.mean(0))
    model.std.copy_(train_x.std(0).clamp_min(0.001))
    callback = ModelCheckpoint(
        dirpath=output / "checkpoints",
        filename="best",
        monitor="val/top1",
        mode="max",
        save_last=True,
        enable_version_counter=False,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        max_epochs=args.epochs,
        callbacks=[callback, EarlyStopping(monitor="val/top1", mode="max", patience=12)],
        logger=CSVLogger(str(output), name="logs"),
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        gradient_clip_val=1,
    )
    trainer.fit(
        model,
        DataLoader(TensorDataset(train_x, labels), batch_size=256, shuffle=True),
        DataLoader(TensorDataset(val_x, truth), batch_size=256),
        ckpt_path=str(output / "checkpoints/last.ckpt") if args.resume else None,
    )
    best = (
        CategoryClassifier.load_from_checkpoint(callback.best_model_path, weights_only=False)
        .cpu()
        .eval()
    )
    with torch.no_grad():
        scores = best(val_x)
    selected = select_reference(scores, val["predictions"], bank)
    permutation = torch.randperm(len(truth), generator=torch.Generator().manual_seed(20260913))
    report = {
        "complete": True,
        "metrics": accuracy(scores, truth),
        "shuffled_eeg": accuracy(scores[permutation], truth),
        "checkpoint": callback.best_model_path,
        "checkpoint_sha256": sha256_of_file(Path(callback.best_model_path)),
        "protocol": bank["metadata"],
        "loss": "supervised cross entropy, label smoothing .1; frozen hierarchical and CLIP EEG",
        "selection": "best legacy validation top1; no generated image selection",
        "confidence": "uncalibrated logits; report top5 alternatives, not certainty",
        "test_evaluated": False,
    }
    records = []
    top = scores.topk(5, dim=1)
    for i, row in enumerate(rows["val"]):
        records.append(
            {
                "filename": row["filename"],
                "true_category": category(row),
                "predicted_category": names[int(top.indices[i, 0])],
                "top5": [names[int(c)] for c in top.indices[i]],
                "logit_margin": float(top.values[i, 0] - top.values[i, 1]),
                "correct": bool(top.indices[i, 0] == truth[i]),
                "reference": rows["train"][int(selected[i])],
            }
        )
    torch.save(
        {
            "rows": rows["val"],
            "variants": {"classified_train_prior": bank["clip"][selected].float()},
            "metadata": report,
        },
        output / "conditions.pt",
    )
    torch.save(
        {"scores": scores, "selected_train_indices": selected, "rows": rows["val"]},
        output / "predictions.pt",
    )
    (output / "predictions.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
