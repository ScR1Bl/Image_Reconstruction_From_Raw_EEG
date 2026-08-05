"""Low-frequency color decoding from global and spatial visual features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .model_utils import correlation_loss, image_gradients


@dataclass
class ColorOutput:
    rgb32: torch.Tensor
    descriptor: torch.Tensor


class ResidualConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


class ColorDecoder(nn.Module):
    """Decode coarse color while keeping global palette and spatial layout separate."""

    def __init__(
        self, grid_dim: int = 192, global_dim: int = 384, width: int = 128, descriptor_dim: int = 63
    ) -> None:
        super().__init__()
        if width % 8:
            raise ValueError("width must be divisible by 8")
        self.grid_dim, self.global_dim = grid_dim, global_dim
        self.width, self.descriptor_dim = width, descriptor_dim
        self.stem = nn.Sequential(nn.Conv2d(grid_dim, width, 1), ResidualConv(width))
        self.film = nn.Sequential(nn.LayerNorm(global_dim), nn.Linear(global_dim, 2 * width))
        self.up16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width, 3, padding=1),
            ResidualConv(width),
        )
        self.up32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width // 2, 3, padding=1),
            ResidualConv(width // 2),
        )
        self.rgb_head = nn.Sequential(
            nn.GroupNorm(8, width // 2), nn.GELU(), nn.Conv2d(width // 2, 3, 1)
        )
        self.descriptor_head = nn.Sequential(
            nn.LayerNorm(width + global_dim),
            nn.Linear(width + global_dim, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, descriptor_dim),
            nn.Sigmoid(),
        )

    def forward(self, grid: torch.Tensor, global_feature: torch.Tensor) -> ColorOutput:
        if grid.ndim != 4 or grid.shape[1:3] != (8, 8):
            raise ValueError("grid must have shape [batch,8,8,features]")
        value = self.stem(grid.permute(0, 3, 1, 2).float())
        scale, bias = self.film(global_feature.float()).chunk(2, dim=-1)
        value = value * (1 + 0.2 * torch.tanh(scale[:, :, None, None]))
        value = value + 0.2 * bias[:, :, None, None]
        pooled = value.mean((2, 3))
        rgb = self.rgb_head(self.up32(self.up16(value))).sigmoid()
        descriptor = self.descriptor_head(torch.cat((pooled, global_feature.float()), -1))
        return ColorOutput(rgb, descriptor)

    def config(self) -> dict:
        return {
            "grid_dim": self.grid_dim,
            "global_dim": self.global_dim,
            "width": self.width,
            "descriptor_dim": self.descriptor_dim,
        }


def rgb_to_ycbcr(rgb: torch.Tensor) -> torch.Tensor:
    red, green, blue = rgb.float().unbind(1)
    y = 0.299 * red + 0.587 * green + 0.114 * blue
    cb = (blue - y) * 0.564 + 0.5
    cr = (red - y) * 0.713 + 0.5
    return torch.stack((y, cb, cr), 1)


def rgb_to_oklab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable sRGB to OKLab conversion for perceptual color errors."""
    value = rgb.float().clamp(0, 1)
    linear = torch.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055).pow(2.4))
    red, green, blue = linear.unbind(1)
    l = 0.4122214708 * red + 0.5363325363 * green + 0.0514459929 * blue
    m = 0.2119034982 * red + 0.6806995451 * green + 0.1073969566 * blue
    s = 0.0883024619 * red + 0.2817188376 * green + 0.6299787005 * blue
    l, m, s = (
        l.clamp_min(1e-8).pow(1 / 3),
        m.clamp_min(1e-8).pow(1 / 3),
        s.clamp_min(1e-8).pow(1 / 3),
    )
    lightness = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
    a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
    b = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
    return torch.stack((lightness, a, b), 1)


def soft_chroma_histogram(
    oklab: torch.Tensor, bins: int = 16, bandwidth: float = 0.035
) -> torch.Tensor:
    chroma = oklab[:, 1:].flatten(2)
    centers = torch.linspace(-0.4, 0.4, bins, device=oklab.device, dtype=oklab.dtype)
    distances = (chroma[:, :, :, None] - centers).square()
    assignments = (-distances / (2 * bandwidth**2)).softmax(-1)
    return assignments.mean(2)


def color_descriptor(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable counterpart of the stored 63-value RGB descriptor."""
    pixels = rgb.float().flatten(2)
    statistics = torch.cat(
        (
            pixels.mean(-1),
            pixels.std(-1, unbiased=False),
            torch.quantile(pixels, 0.10, dim=-1),
            torch.quantile(pixels, 0.50, dim=-1),
            torch.quantile(pixels, 0.90, dim=-1),
        ),
        -1,
    )
    palette = F.adaptive_avg_pool2d(rgb.float(), (4, 4)).permute(0, 2, 3, 1).flatten(1)
    return torch.cat((statistics, palette), -1)


def color_sample_errors(
    output: ColorOutput, rgb: torch.Tensor, descriptor: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-image visual errors used by residual-adapter training."""
    prediction, target = output.rgb32.float(), rgb.float()
    pred_oklab, true_oklab = rgb_to_oklab(prediction), rgb_to_oklab(target)
    pixel = (prediction - target).abs().mean((1, 2, 3))
    perceptual = (pred_oklab - true_oklab).abs().mean((1, 2, 3))
    chroma = (pred_oklab[:, 1:] - true_oklab[:, 1:]).abs().mean((1, 2, 3))
    palette = (
        (F.adaptive_avg_pool2d(prediction, 4) - F.adaptive_avg_pool2d(target, 4))
        .abs()
        .mean((1, 2, 3))
    )
    rendered = color_descriptor(prediction)
    descriptor_error = F.smooth_l1_loss(
        rendered, descriptor.float(), beta=0.04, reduction="none"
    ).mean(1)
    histogram = (
        (soft_chroma_histogram(pred_oklab) - soft_chroma_histogram(true_oklab)).abs().mean((1, 2))
    )
    error = (
        0.30 * pixel
        + 0.75 * perceptual
        + 0.80 * chroma
        + 0.90 * palette
        + 0.45 * descriptor_error
        + 0.50 * histogram
    )
    return error, {
        "pixel": pixel,
        "oklab": perceptual,
        "chroma_oklab": chroma,
        "palette": palette,
        "rendered_descriptor": descriptor_error,
        "chroma_histogram": histogram,
    }


def rendered_descriptor_contrastive(
    rgb: torch.Tensor, descriptor: torch.Tensor, temperature: float = 0.10
) -> torch.Tensor:
    prediction = color_descriptor(rgb.float())
    prediction = F.normalize(prediction - prediction.mean(0, keepdim=True), dim=-1)
    target = descriptor.float()
    target = F.normalize(target - target.mean(0, keepdim=True), dim=-1)
    logits = prediction @ target.T / temperature
    labels = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def color_loss(
    output: ColorOutput, rgb: torch.Tensor, descriptor: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    prediction, target = output.rgb32.float(), rgb.float()
    pred_yuv, true_yuv = rgb_to_ycbcr(prediction), rgb_to_ycbcr(target)
    pixel = F.smooth_l1_loss(prediction, target, beta=0.08)
    luminance = F.smooth_l1_loss(pred_yuv[:, :1], true_yuv[:, :1], beta=0.06)
    chroma = F.smooth_l1_loss(pred_yuv[:, 1:], true_yuv[:, 1:], beta=0.04)
    low4 = F.smooth_l1_loss(
        F.adaptive_avg_pool2d(prediction, 4), F.adaptive_avg_pool2d(target, 4), beta=0.05
    )
    low8 = F.smooth_l1_loss(
        F.adaptive_avg_pool2d(prediction, 8), F.adaptive_avg_pool2d(target, 8), beta=0.05
    )
    descriptor_target = descriptor.float()
    descriptor_direct = F.smooth_l1_loss(output.descriptor.float(), descriptor_target, beta=0.04)
    descriptor_consistency = F.smooth_l1_loss(
        color_descriptor(prediction), descriptor_target, beta=0.04
    )
    pred_centered = F.normalize(
        output.descriptor.float() - output.descriptor.float().mean(0, keepdim=True), dim=-1
    )
    true_centered = F.normalize(descriptor_target - descriptor_target.mean(0, keepdim=True), dim=-1)
    logits = pred_centered @ true_centered.T / 0.10
    labels = torch.arange(len(logits), device=logits.device)
    descriptor_contrastive = 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
    )
    gradient = sum(
        F.l1_loss(a, b) for a, b in zip(image_gradients(prediction), image_gradients(target))
    )
    palette_correlation = correlation_loss(
        F.adaptive_avg_pool2d(prediction, 4), F.adaptive_avg_pool2d(target, 4)
    )
    parts = {
        "pixel": pixel,
        "luminance": luminance,
        "chroma": chroma,
        "low4": low4,
        "low8": low8,
        "descriptor": descriptor_direct,
        "descriptor_consistency": descriptor_consistency,
        "descriptor_contrastive": descriptor_contrastive,
        "gradient": gradient,
        "palette_correlation": palette_correlation,
    }
    total = (
        0.5 * pixel
        + 0.35 * luminance
        + 1.25 * chroma
        + 1.2 * low4
        + 0.8 * low8
        + 0.8 * descriptor_direct
        + 0.5 * descriptor_consistency
        + 0.20 * descriptor_contrastive
        + 0.08 * gradient
        + 0.35 * palette_correlation
    )
    return total, parts


def color_metrics(
    output: ColorOutput,
    rgb: torch.Tensor,
    descriptor: torch.Tensor,
    train_mean: torch.Tensor | None = None,
) -> dict[str, float]:
    prediction, target = output.rgb32.float(), rgb.float()
    pred_yuv, true_yuv = rgb_to_ycbcr(prediction), rgb_to_ycbcr(target)
    mse = F.mse_loss(prediction, target)
    palette_prediction = F.adaptive_avg_pool2d(prediction, 4)
    palette_target = F.adaptive_avg_pool2d(target, 4)
    pred_descriptor = F.normalize(
        output.descriptor.float() - output.descriptor.float().mean(0, keepdim=True), dim=-1
    )
    true_descriptor = F.normalize(
        descriptor.float() - descriptor.float().mean(0, keepdim=True), dim=-1
    )
    similarities = pred_descriptor @ true_descriptor.T
    ranks = 1 + (similarities > similarities.diagonal()[:, None]).sum(1)
    result = {
        "rgb_mae": float(F.l1_loss(prediction, target)),
        "chroma_mae": float(F.l1_loss(pred_yuv[:, 1:], true_yuv[:, 1:])),
        "luminance_mae": float(F.l1_loss(pred_yuv[:, :1], true_yuv[:, :1])),
        "palette_mae": float(F.l1_loss(palette_prediction, palette_target)),
        "palette_pearson": float(1 - correlation_loss(palette_prediction, palette_target)),
        "descriptor_cosine": float((pred_descriptor * true_descriptor).sum(-1).mean()),
        "descriptor_top1": float((ranks == 1).float().mean()),
        "descriptor_top5": float((ranks <= 5).float().mean()),
        "psnr": float(-10 * torch.log10(mse.clamp_min(1e-10))),
        "between_image_std_ratio": float(
            prediction.flatten(1).std(0).mean() / target.flatten(1).std(0).mean().clamp_min(1e-8)
        ),
    }
    if train_mean is not None:
        baseline = F.mse_loss(train_mean.float().expand_as(target), target)
        result["rgb_gain_vs_train_mean"] = float(1 - mse / baseline.clamp_min(1e-10))
    return result


def load_color_decoder(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[ColorDecoder, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = ColorDecoder(**payload["model_config"])
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload
