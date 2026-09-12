"""Geometry diagnostics with explicit tie and degenerate-shape handling."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F


def shape_moments(masks: torch.Tensor) -> dict[str, torch.Tensor]:
    value = masks.float().reshape(-1, masks.shape[-2], masks.shape[-1]).clamp(0, 1)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, value.shape[-2], device=value.device),
        torch.linspace(0, 1, value.shape[-1], device=value.device),
        indexing="ij",
    )
    mass = value.sum((1, 2)).clamp_min(1e-8)
    cx, cy = (value * xx).sum((1, 2)) / mass, (value * yy).sum((1, 2)) / mass
    dx, dy = xx - cx[:, None, None], yy - cy[:, None, None]
    vx = (value * dx.square()).sum((1, 2)) / mass
    vy = (value * dy.square()).sum((1, 2)) / mass
    xy = (value * dx * dy).sum((1, 2)) / mass
    doubled = torch.stack((vx - vy, 2 * xy), -1)
    norm = doubled.norm(dim=-1)
    return {
        "centroid": torch.stack((cx, cy), -1),
        "area": value.mean((1, 2)),
        "axis": F.normalize(doubled, dim=-1, eps=1e-8),
        "anisotropy": norm / (vx + vy).clamp_min(1e-8),
        "axis_norm": norm,
    }


def same_category_2afc(prediction, target, concepts):
    pred = prediction.float().flatten(1)
    truth = target.float().flatten(1)
    groups = defaultdict(list)
    for i, concept in enumerate(concepts):
        groups[concept].append(i)
    values, margins, rows = [], [], []
    for indices in groups.values():
        if len(indices) != 2:
            continue
        for i, j in (indices, indices[::-1]):
            correct = (pred[i] - truth[i]).square().mean()
            wrong = (pred[i] - truth[j]).square().mean()
            margin = float(wrong - correct)
            values.append(float(margin > 1e-8) + 0.5 * float(abs(margin) <= 1e-8))
            margins.append(margin)
            rows.append(i)
    return {
        "accuracy": float(np.mean(values)) if values else None,
        "images": len(values),
        "pairs": len(values) // 2,
        "scores": values,
        "margins": margins,
        "rows": rows,
    }


def geometry_metrics(prediction, target, concepts):
    pred, truth = prediction.float().clamp(0, 1), target.float()
    pm, tm = shape_moments(pred), shape_moments(truth)
    p, t = pred >= 0.5, truth >= 0.5
    intersection = (p & t).flatten(1).sum(1)
    union = (p | t).flatten(1).sum(1)
    iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union).float())
    eligible = (tm["anisotropy"] >= 0.2) & (tm["area"] > 0.01)
    cosine = (pm["axis"] * tm["axis"]).sum(-1).clamp(-1, 1)
    degrees = torch.rad2deg(torch.acos(cosine)) * 0.5
    degrees = torch.where(pm["axis_norm"] > 1e-8, degrees, torch.full_like(degrees, 90))
    pair = same_category_2afc(pred, truth, concepts)
    return {
        "shape_mse": float(F.mse_loss(pred, truth)),
        "shape_iou": float(iou.mean()),
        "centroid_error": float((pm["centroid"] - tm["centroid"]).norm(dim=-1).mean()),
        "area_mae": float((pm["area"] - tm["area"]).abs().mean()),
        "axis_error_degrees": float(degrees[eligible].mean()) if eligible.any() else None,
        "axis_eligible_images": int(eligible.sum()),
        "within_category_2afc": pair["accuracy"],
        "within_category_pairs": pair["pairs"],
        "axis_is_directionless": True,
    }


def retrieval_metrics(prediction, target, chunk_size=256):
    """Uniform tie-breaking expectation, including reciprocal rank."""
    pred, truth = F.normalize(prediction.float(), dim=-1), F.normalize(target.float(), dim=-1)
    if len(pred) != len(truth) or not len(pred):
        raise ValueError("expected aligned nonempty candidate pool")
    harmonic = torch.cat((torch.zeros(1), (1 / torch.arange(1, len(truth) + 1).float()).cumsum(0)))
    top1, top5, mrr = [], [], []
    for start in range(0, len(pred), chunk_size):
        scores = pred[start : start + chunk_size] @ truth.T
        match = scores[torch.arange(len(scores)), torch.arange(start, start + len(scores))][:, None]
        greater = (scores > match).sum(1).cpu()
        equal = (scores == match).sum(1).cpu().clamp_min(1)
        top1.append(((1 - greater).clamp(min=0) / equal).clamp(max=1))
        top5.append(((5 - greater).clamp(min=0) / equal).clamp(max=1))
        mrr.append((harmonic[greater + equal] - harmonic[greater]) / equal)
    return {
        "top1": float(torch.cat(top1).mean()),
        "top5": float(torch.cat(top5).mean()),
        "mrr": float(torch.cat(mrr).mean()),
        "candidates": len(truth),
        "tie_policy": "uniform expectation",
    }
