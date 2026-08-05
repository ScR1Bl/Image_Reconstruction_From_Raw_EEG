"""Global/local multi-layer DINO edge renderer with iterative crisp refinement."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import dice_loss, focal_map_loss, image_gradients
from .pyramid_edge import PyramidEdgeOutput


class Residual(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        groups = min(16, width)
        while width % groups:
            groups -= 1
        self.block = nn.Sequential(
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(groups, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, value):
        return value + self.block(value)


class PixelDifferenceConv(nn.Module):
    """Learnable neighbor-minus-center convolution in the spirit of PiDiNet."""

    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(channels, channels, 3, 3) * (0.02 / channels**0.5))
        self.bias = nn.Parameter(torch.zeros(channels))
        mask = torch.ones(1, 1, 3, 3)
        mask[:, :, 1, 1] = 0
        center = torch.zeros(1, 1, 3, 3)
        center[:, :, 1, 1] = 1
        self.register_buffer("neighbor_mask", mask)
        self.register_buffer("center_mask", center)

    def forward(self, value):
        neighbor = self.weight * self.neighbor_mask
        kernel = neighbor - neighbor.sum((2, 3), keepdim=True) * self.center_mask
        return F.conv2d(value, kernel, self.bias, padding=1)


class FourierCrispRefiner(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.high_gain = nn.Parameter(torch.tensor(-1.0))
        self.fuse = nn.Sequential(nn.Conv2d(width * 2, width, 1), Residual(width))

    def forward(self, value):
        height, width = value.shape[-2:]
        frequency = torch.fft.rfft2(value.float(), norm="ortho")
        fy = torch.fft.fftfreq(height, device=value.device)[:, None]
        fx = torch.fft.rfftfreq(width, device=value.device)[None, :]
        radius = (fy.square() + fx.square()).sqrt()
        high = (radius / 0.5).clamp(0, 1).square()
        detail = torch.fft.irfft2(frequency * high, (height, width), norm="ortho").to(value.dtype)
        return self.fuse(torch.cat((value, torch.sigmoid(self.high_gain) * detail), 1))


@dataclass
class AdvancedEdgeOutput:
    pyramid: PyramidEdgeOutput
    side_crisp64: tuple[torch.Tensor, ...]
    layer_weights: torch.Tensor


class DINOEdgeOracleV2(nn.Module):
    """EDTER-style global/local fusion plus HED side supervision and MEMO refinement."""

    def __init__(
        self,
        input_dim: int = 384,
        layers: int = 4,
        width: int = 160,
        heads: int = 5,
        transformer_depth: int = 2,
        refinement_steps: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.input_dim, self.layers, self.width = input_dim, layers, width
        self.heads, self.transformer_depth, self.refinement_steps, self.dropout = (
            heads,
            transformer_depth,
            refinement_steps,
            dropout,
        )
        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
                for _ in range(layers)
            ]
        )
        self.cls_projections = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width))
                for _ in range(layers)
            ]
        )
        self.layer_logits = nn.Parameter(torch.zeros(3, layers))
        self.position = nn.Parameter(torch.randn(1, 256, width) * 0.01)
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.global_mixer = nn.TransformerEncoder(
            layer, transformer_depth, enable_nested_tensor=False
        )
        self.local_difference = nn.Sequential(
            PixelDifferenceConv(width), nn.GroupNorm(10, width), nn.SiLU(), Residual(width)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(width * 2, width, 1), Residual(width), Residual(width)
        )
        self.shape_head = nn.Conv2d(width, 2, 3, padding=1)
        self.up32 = nn.Sequential(
            nn.ConvTranspose2d(width, width // 2, 4, 2, 1),
            Residual(width // 2),
            Residual(width // 2),
        )
        self.semantic_head = nn.Conv2d(width // 2, 1, 3, padding=1)
        self.up64 = nn.Sequential(
            nn.ConvTranspose2d(width // 2, width // 4, 4, 2, 1),
            Residual(width // 4),
            Residual(width // 4),
        )
        fine = width // 4
        self.fourier = FourierCrispRefiner(fine)
        self.fine_head = nn.Conv2d(fine, 6, 3, padding=1)
        self.refiner = nn.Sequential(
            nn.Conv2d(fine + 3, fine, 3, padding=1),
            Residual(fine),
            Residual(fine),
            nn.Conv2d(fine, 1, 3, padding=1),
        )

    def _fuse_layers(self, values, projections, weights):
        projected = torch.stack(
            [projections[i](values[:, i].float()) for i in range(self.layers)], 1
        )
        shape = [1, self.layers] + [1] * (projected.ndim - 2)
        return (projected * weights.reshape(shape)).sum(1)

    def forward(self, patches, cls, mask_ratio: float = 0.0):
        if patches.ndim != 5 or patches.shape[1] != self.layers or patches.shape[2:4] != (16, 16):
            raise ValueError("patches must have shape [B,layers,16,16,features]")
        weights = self.layer_logits.softmax(-1)
        global_tokens = (
            self._fuse_layers(patches, self.projections, weights[0]).flatten(1, 2) + self.position
        )
        global_cls = self._fuse_layers(cls, self.cls_projections, weights[1])
        global_tokens = self.global_mixer(global_tokens + global_cls[:, None])
        global_map = global_tokens.transpose(1, 2).reshape(len(patches), self.width, 16, 16)
        local = self._fuse_layers(patches, self.projections, weights[2]).permute(0, 3, 1, 2)
        local = self.local_difference(local)
        level16 = self.fusion(torch.cat((global_map, local), 1))
        coarse = self.shape_head(level16)
        level32 = self.up32(level16)
        semantic = self.semantic_head(level32)
        level64 = self.fourier(self.up64(level32))
        fine = self.fine_head(level64)
        crisp = fine[:, 1:2]
        sides = []
        for step in range(self.refinement_steps):
            probability = crisp.sigmoid()
            confidence = (2 * (probability - 0.5).abs()).detach()
            if self.training and mask_ratio:
                visible = (torch.rand_like(probability) >= mask_ratio).float()
            else:
                threshold = 0.25 + 0.20 * step / max(self.refinement_steps - 1, 1)
                visible = (confidence >= threshold).float()
            delta = self.refiner(
                torch.cat((level64, probability * visible, confidence, visible), 1)
            )
            crisp = crisp + delta * (1 - 0.65 * confidence)
            sides.append(crisp)
        orientation = F.normalize(torch.tanh(fine[:, 4:6]), dim=1, eps=1e-5)
        pyramid = PyramidEdgeOutput(
            coarse[:, :1],
            coarse[:, 1:2].sigmoid(),
            semantic,
            fine[:, :1],
            crisp,
            fine[:, 2:3].sigmoid(),
            fine[:, 3:4].sigmoid(),
            orientation,
        )
        return AdvancedEdgeOutput(pyramid, tuple(sides), weights)

    def config(self):
        return {
            "input_dim": self.input_dim,
            "layers": self.layers,
            "width": self.width,
            "heads": self.heads,
            "transformer_depth": self.transformer_depth,
            "refinement_steps": self.refinement_steps,
            "dropout": self.dropout,
        }


def _soft_skeleton(value, iterations=4):
    value = value.clamp(0, 1)
    skeleton = F.relu(value - F.max_pool2d(-F.max_pool2d(-value, 3, 1, 1), 3, 1, 1))
    for _ in range(iterations):
        eroded = -F.max_pool2d(-value, 3, 1, 1)
        opened = F.max_pool2d(eroded, 3, 1, 1)
        delta = F.relu(eroded - opened)
        skeleton = skeleton + (1 - skeleton) * delta
        value = eroded
    return skeleton


def advanced_edge_loss(output: AdvancedEdgeOutput, target: dict[str, torch.Tensor]):
    value = output.pyramid
    shape = focal_map_loss(value.shape_logits16, target["shape16"]) + dice_loss(
        value.shape_logits16, target["shape16"]
    )
    distance = F.smooth_l1_loss(value.distance16, target["distance16"].float(), beta=0.03)
    distance_grad = sum(
        F.l1_loss(a, b)
        for a, b in zip(
            image_gradients(value.distance16), image_gradients(target["distance16"].float())
        )
    )
    semantic = focal_map_loss(value.semantic_logits32, target["semantic32"]) + 1.25 * dice_loss(
        value.semantic_logits32, target["semantic32"]
    )
    perceptual = F.binary_cross_entropy_with_logits(
        value.perceptual_logits64, target["perceptual64"].float()
    )
    crisp_target = target["crisp64"].float()
    crisp = focal_map_loss(value.crisp_logits64, crisp_target) + dice_loss(
        value.crisp_logits64, crisp_target
    )
    uncertainty = value.uncertainty64.clamp(1e-4, 1 - 1e-4)
    pixel_bce = F.binary_cross_entropy_with_logits(
        value.crisp_logits64, crisp_target, reduction="none"
    )
    heteroscedastic = (pixel_bce / (0.15 + uncertainty) + 0.25 * uncertainty.log1p()).mean()
    uncertainty_supervision = F.smooth_l1_loss(
        uncertainty, target["uncertainty64"].float(), beta=0.04
    )
    confidence = F.binary_cross_entropy(
        value.confidence64.clamp(1e-5, 1 - 1e-5), target["confidence64"].float()
    )
    weight = F.max_pool2d(crisp_target, 5, 1, 2) * (0.25 + target["confidence64"].float())
    similarity = (value.orientation64 * target["orientation64"].float()).sum(1, keepdim=True).abs()
    orientation = ((1 - similarity) * weight).sum() / weight.sum().clamp_min(1)
    pred_skeleton = _soft_skeleton(value.crisp_logits64.sigmoid())
    true_skeleton = _soft_skeleton(crisp_target)
    topology = 1 - (2 * (pred_skeleton * true_skeleton).sum() + 1) / (
        pred_skeleton.sum() + true_skeleton.sum() + 1
    )
    pred_fft = torch.fft.rfft2(value.crisp_logits64.sigmoid().float(), norm="ortho").abs()
    true_fft = torch.fft.rfft2(crisp_target.float(), norm="ortho").abs()
    frequency = F.l1_loss(pred_fft, true_fft)
    side = sum(
        F.binary_cross_entropy_with_logits(side, crisp_target) for side in output.side_crisp64
    ) / len(output.side_crisp64)
    consistency = F.l1_loss(
        F.adaptive_max_pool2d(value.crisp_logits64.sigmoid(), 32), value.semantic_logits32.sigmoid()
    )
    parts = {
        "shape": shape,
        "distance": distance,
        "distance_gradient": distance_grad,
        "semantic": semantic,
        "perceptual": perceptual,
        "crisp": crisp,
        "heteroscedastic": heteroscedastic,
        "uncertainty": uncertainty_supervision,
        "confidence": confidence,
        "orientation": orientation,
        "topology": topology,
        "frequency": frequency,
        "side": side,
        "consistency": consistency,
    }
    total = (
        1.0 * shape
        + 0.75 * distance
        + 0.30 * distance_grad
        + 1.3 * semantic
        + 0.55 * perceptual
        + 1.5 * crisp
        + 0.35 * heteroscedastic
        + 0.25 * uncertainty_supervision
        + 0.20 * confidence
        + 0.30 * orientation
        + 0.55 * topology
        + 0.08 * frequency
        + 0.35 * side
        + 0.20 * consistency
    )
    return total, parts
