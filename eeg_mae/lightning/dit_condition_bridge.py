"""Small EEG-condition bridge into the frozen CLIP-trained DiT adapter.

The bridge is global, not native spatial cross-attention. Every ablation has
the same parameter shapes; absent modalities are zeroed before its first layer.
"""

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
from torch.utils.data import DataLoader, Dataset

from ..data.dit_ten_subjects import OUTPUT, normalize_clip
from ..evaluation.geometry_metrics import retrieval_metrics


class BridgeDataset(Dataset):
    def __init__(self, root, split):
        self.data = {
            k: np.load(root / f"{split}_{k}.npy", mmap_mode="r") for k in ("S", "E", "C", "target")
        }

    def __len__(self):
        return len(self.data["S"])

    def __getitem__(self, index):
        return {
            k: torch.from_numpy(np.array(v[index], dtype=np.float32, copy=True)).flatten()
            for k, v in self.data.items()
        }


class DiTConditionBridge(L.LightningModule):
    def __init__(self, variant="S", structural_dim=12288, hidden=256, target_norm=20.42, lr=0.0003):
        super().__init__()
        self.save_hyperparameters()
        if variant not in {"S", "S_E", "S_C", "S_E_C"}:
            raise ValueError(variant)
        self.head = nn.Sequential(
            nn.Linear(768 + 2 * structural_dim, hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, 768),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.validation = []

    def forward(self, batch):
        parts = [F.layer_norm(batch["S"], (768,))]
        for label in ("E", "C"):
            value = batch[label]
            parts.append(
                F.layer_norm(value, (value.shape[-1],))
                if label in self.hparams.variant.split("_")
                else torch.zeros_like(value)
            )
        base = F.normalize(batch["S"], dim=-1)
        return normalize_clip(base + 0.1 * self.head(torch.cat(parts, 1)), self.hparams.target_norm)

    def training_step(self, batch, batch_idx):
        raw_prediction = self(batch).float()
        prediction = F.normalize(raw_prediction, dim=-1)
        target = F.normalize(batch["target"].float(), dim=-1)
        logits = prediction @ target.T / 0.07
        labels = torch.arange(len(prediction), device=self.device)
        contrastive = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        # Preserve native CLIP geometry for the frozen image adapter.
        mse = (
            F.mse_loss(raw_prediction, batch["target"].float()) * 768 / self.hparams.target_norm**2
        )
        loss = mse + 0.1 * contrastive
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite bridge loss")
        self.log("train/loss", loss, on_epoch=True, on_step=False, batch_size=len(prediction))
        return loss

    def validation_step(self, batch, batch_idx):
        self.validation.append((self(batch).float().cpu(), batch["target"].float().cpu()))

    def on_validation_epoch_end(self):
        if not self.validation:
            return
        pred, target = [torch.cat([b[i] for b in self.validation]) for i in range(2)]
        metrics = retrieval_metrics(pred, target)
        metrics["native_clip_mse"] = float(F.mse_loss(pred, target))
        self.log("val/mrr", metrics["mrr"])
        self.log("val/mse", metrics["native_clip_mse"])
        print(
            json.dumps({"variant": self.hparams.variant, "epoch": self.current_epoch, **metrics}),
            flush=True,
        )
        self.validation = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", type=Path, default=OUTPUT / "conditions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=("S", "S_E", "S_C", "S_E_C"), required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if (args.output / "checkpoints").exists():
        raise RuntimeError("use a fresh bridge run")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    L.seed_everything(20260913)
    manifest = json.loads((args.conditions / "manifest.json").read_text())
    if not manifest["complete"] or manifest["subjects"] != list(range(10)):
        raise ValueError("wrong conditions")
    train, val = BridgeDataset(args.conditions, "train"), BridgeDataset(args.conditions, "val")
    norm = float(
        torch.from_numpy(np.array(train.data["target"], dtype=np.float32)).norm(dim=-1).mean()
    )
    module = DiTConditionBridge(variant=args.variant, target_norm=norm)
    checkpoint = ModelCheckpoint(
        dirpath=args.output / "checkpoints",
        filename="best",
        monitor="val/mse",
        mode="min",
        save_last=True,
        enable_version_counter=False,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        max_epochs=args.epochs,
        callbacks=[checkpoint, EarlyStopping(monitor="val/mse", mode="min", patience=7)],
        logger=CSVLogger(str(args.output), name="logs"),
        gradient_clip_val=1,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=str(args.output),
    )
    trainer.fit(
        module,
        DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0),
        DataLoader(val, batch_size=128, num_workers=0),
    )
    best = (
        DiTConditionBridge.load_from_checkpoint(checkpoint.best_model_path, weights_only=False)
        .cuda()
        .eval()
    )
    predicted = []
    with torch.no_grad():
        for batch in DataLoader(val, batch_size=128):
            predicted.append(best({k: v.cuda() for k, v in batch.items()}).float().cpu())
    pred = torch.cat(predicted)
    target = torch.from_numpy(np.array(val.data["target"], dtype=np.float32))
    baseline = torch.from_numpy(np.array(val.data["S"], dtype=np.float32))
    report = {
        "variant": args.variant,
        "metrics": retrieval_metrics(pred, target),
        "baseline": retrieval_metrics(baseline, target),
        "checkpoint": checkpoint.best_model_path,
        "subjects": list(range(10)),
        "test_evaluated": False,
        "bridge_type": "global CLIP residual from EEG semantic and pooled spatial tokens; not native spatial DiT conditioning",
        "validation_limitation": "legacy val used for encoder and bridge selection; in-sample encoder predictions on bridge train",
    }
    torch.save(
        {"prediction": pred, "rows": manifest["rows"]["val"], "metadata": report},
        args.output / "predictions.pt",
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
