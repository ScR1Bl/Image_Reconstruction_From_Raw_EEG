from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import correlation_loss, dice_loss, focal_map_loss, image_gradients


@dataclass
class PyramidEdgeOutput:
    shape_logits16: torch.Tensor
    distance16: torch.Tensor
    semantic_logits32: torch.Tensor
    perceptual_logits64: torch.Tensor
    crisp_logits64: torch.Tensor
    confidence64: torch.Tensor
    uncertainty64: torch.Tensor
    orientation64: torch.Tensor


class ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, channels)
        while channels % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class PyramidEdgeDecoder(nn.Module):
    """Progressive DINO-grid decoder with explicit supervision at 16/32/64."""

    def __init__(self, input_dim: int = 192, widths=(192, 128, 96, 64)) -> None:
        super().__init__()
        self.input_dim, self.widths = input_dim, tuple(widths)
        w8, w16, w32, w64 = self.widths
        self.stem = nn.Sequential(nn.Conv2d(input_dim, w8, 1), ResidualConv(w8), ResidualConv(w8))
        self.up16 = nn.Sequential(
            nn.ConvTranspose2d(w8, w16, 4, 2, 1), ResidualConv(w16), ResidualConv(w16)
        )
        self.up32 = nn.Sequential(
            nn.ConvTranspose2d(w16, w32, 4, 2, 1), ResidualConv(w32), ResidualConv(w32)
        )
        self.up64 = nn.Sequential(
            nn.ConvTranspose2d(w32, w64, 4, 2, 1), ResidualConv(w64), ResidualConv(w64)
        )
        self.shape_head = nn.Conv2d(w16, 2, 3, padding=1)
        self.semantic_head = nn.Conv2d(w32, 1, 3, padding=1)
        self.fine_head = nn.Conv2d(w64, 6, 3, padding=1)

    def forward(self, grid: torch.Tensor) -> PyramidEdgeOutput:
        if grid.ndim != 4 or grid.shape[1:3] != (8, 8) or grid.shape[-1] != self.input_dim:
            raise ValueError(f"expected [B,8,8,{self.input_dim}], got {tuple(grid.shape)}")
        value = self.stem(grid.permute(0, 3, 1, 2).float())
        level16 = self.up16(value)
        coarse = self.shape_head(level16)
        level32 = self.up32(level16)
        semantic = self.semantic_head(level32)
        level64 = self.up64(level32)
        fine = self.fine_head(level64)
        orientation = F.normalize(torch.tanh(fine[:, 4:6]), dim=1, eps=1e-5)
        return PyramidEdgeOutput(
            shape_logits16=coarse[:, :1],
            distance16=coarse[:, 1:2].sigmoid(),
            semantic_logits32=semantic,
            perceptual_logits64=fine[:, :1],
            crisp_logits64=fine[:, 1:2],
            confidence64=fine[:, 2:3].sigmoid(),
            uncertainty64=fine[:, 3:4].sigmoid(),
            orientation64=orientation,
        )

    def config(self) -> dict:
        return {"input_dim": self.input_dim, "widths": self.widths}


def soft_boundary(value: torch.Tensor) -> torch.Tensor:
    dilation = F.max_pool2d(value, 3, stride=1, padding=1)
    erosion = -F.max_pool2d(-value, 3, stride=1, padding=1)
    return (dilation - erosion).clamp(0, 1)


def pyramid_edge_loss(
    output: PyramidEdgeOutput, target: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    shape_focal = focal_map_loss(output.shape_logits16, target["shape16"])
    shape_dice = dice_loss(output.shape_logits16, target["shape16"])
    distance = F.smooth_l1_loss(output.distance16, target["distance16"].float(), beta=0.04)
    distance_gradient = sum(
        F.l1_loss(a, b)
        for a, b in zip(
            image_gradients(output.distance16), image_gradients(target["distance16"].float())
        )
    )
    semantic_focal = focal_map_loss(output.semantic_logits32, target["semantic32"])
    semantic_dice = dice_loss(output.semantic_logits32, target["semantic32"])
    semantic_correlation = correlation_loss(
        output.semantic_logits32.sigmoid(), target["semantic32"]
    )

    confidence = target["confidence64"].float()
    perceptual_element = F.binary_cross_entropy_with_logits(
        output.perceptual_logits64.float(), target["perceptual64"].float(), reduction="none"
    )
    perceptual = (perceptual_element * (0.5 + confidence)).mean()
    crisp_focal = focal_map_loss(output.crisp_logits64, target["crisp64"])
    crisp_dice = dice_loss(output.crisp_logits64, target["crisp64"])
    confidence_prediction = output.confidence64.float().clamp(1e-6, 1 - 1e-6)
    confidence_target = target["confidence64"].float()
    confidence_loss = -(
        confidence_target * confidence_prediction.log()
        + (1 - confidence_target) * (1 - confidence_prediction).log()
    ).mean()
    uncertainty_loss = F.smooth_l1_loss(
        output.uncertainty64.float(), target["uncertainty64"].float(), beta=0.05
    )

    orientation_target = target["orientation64"].float()
    orientation_similarity = (output.orientation64 * orientation_target).sum(1, keepdim=True).abs()
    orientation_weight = F.max_pool2d(target["crisp64"].float(), 5, stride=1, padding=2)
    orientation_weight = orientation_weight * (0.25 + confidence)
    orientation = (
        (1 - orientation_similarity) * orientation_weight
    ).sum() / orientation_weight.sum().clamp_min(1)

    predicted_coarse_boundary = soft_boundary(output.shape_logits16.sigmoid())
    semantic_at_16 = F.adaptive_max_pool2d(output.semantic_logits32.sigmoid(), (16, 16))
    coarse_consistency = F.l1_loss(semantic_at_16, predicted_coarse_boundary)
    parts = {
        "shape_focal": shape_focal,
        "shape_dice": shape_dice,
        "distance": distance,
        "distance_gradient": distance_gradient,
        "semantic_focal": semantic_focal,
        "semantic_dice": semantic_dice,
        "semantic_correlation": semantic_correlation,
        "perceptual": perceptual,
        "crisp_focal": crisp_focal,
        "crisp_dice": crisp_dice,
        "confidence": confidence_loss,
        "uncertainty": uncertainty_loss,
        "orientation": orientation,
        "coarse_consistency": coarse_consistency,
    }
    total = (
        0.75 * shape_focal
        + 1.0 * shape_dice
        + 1.0 * distance
        + 0.5 * distance_gradient
        + 0.75 * semantic_focal
        + 1.25 * semantic_dice
        + 1.0 * semantic_correlation
        + 0.75 * perceptual
        + 0.75 * crisp_focal
        + 1.0 * crisp_dice
        + 0.35 * confidence_loss
        + 0.25 * uncertainty_loss
        + 0.35 * orientation
        + 0.25 * coarse_consistency
    )
    return total, parts


def _dice_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = logits.sigmoid().flatten(1)
    truth = target.float().flatten(1)
    return 1 - (2 * (prediction * truth).sum(1) + 1) / (prediction.sum(1) + truth.sum(1) + 1)


def _balanced_bce_per_sample(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    truth = target.float()
    positive = truth.flatten(1).mean(1).clamp(0.01, 0.99)
    weight = torch.where(
        truth > 0.25, (1 - positive)[:, None, None, None], positive[:, None, None, None]
    )
    value = F.binary_cross_entropy_with_logits(logits.float(), truth, reduction="none")
    return (value * weight).flatten(1).mean(1) / (2 * positive * (1 - positive))


def pyramid_edge_sample_errors(
    output: PyramidEdgeOutput, target: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-image task error, deliberately dominated by visible edge maps."""
    shape = _balanced_bce_per_sample(output.shape_logits16, target["shape16"])
    shape_dice = _dice_per_sample(output.shape_logits16, target["shape16"])
    distance = (output.distance16 - target["distance16"].float()).abs().flatten(1).mean(1)
    semantic = _balanced_bce_per_sample(output.semantic_logits32, target["semantic32"])
    semantic_dice = _dice_per_sample(output.semantic_logits32, target["semantic32"])
    perceptual = (
        (output.perceptual_logits64.sigmoid() - target["perceptual64"].float())
        .abs()
        .flatten(1)
        .mean(1)
    )
    crisp = _balanced_bce_per_sample(output.crisp_logits64, target["crisp64"])
    crisp_dice = _dice_per_sample(output.crisp_logits64, target["crisp64"])
    confidence = (output.confidence64 - target["confidence64"].float()).abs().flatten(1).mean(1)
    uncertainty = (output.uncertainty64 - target["uncertainty64"].float()).abs().flatten(1).mean(1)
    similarity = (output.orientation64 * target["orientation64"].float()).sum(1).abs()
    orientation_weight = F.max_pool2d(target["crisp64"].float(), 5, 1, 2).squeeze(1)
    orientation = ((1 - similarity) * orientation_weight).flatten(1).sum(
        1
    ) / orientation_weight.flatten(1).sum(1).clamp_min(1)
    error = (
        0.45 * shape
        + 0.75 * shape_dice
        + 0.20 * distance
        + 0.55 * semantic
        + 1.00 * semantic_dice
        + 0.45 * perceptual
        + 0.55 * crisp
        + 1.00 * crisp_dice
        + 0.12 * confidence
        + 0.08 * uncertainty
        + 0.20 * orientation
    )
    return error, {
        "shape": shape,
        "shape_dice": shape_dice,
        "distance": distance,
        "semantic": semantic,
        "semantic_dice": semantic_dice,
        "perceptual": perceptual,
        "crisp": crisp,
        "crisp_dice": crisp_dice,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "orientation": orientation,
    }


def load_pyramid_edge_decoder(path, device="cpu"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PyramidEdgeDecoder(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload
