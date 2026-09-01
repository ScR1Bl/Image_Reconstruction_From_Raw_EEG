"""Multi-layer DINO color renderer inspired by DDColor and ColTran."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .color import color_descriptor, rgb_to_oklab, soft_chroma_histogram


def oklab_to_linear_rgb(oklab: torch.Tensor) -> torch.Tensor:
    lightness, a, b = oklab.float().unbind(1)
    l = (lightness + 0.3963377774 * a + 0.2158037573 * b).pow(3)
    m = (lightness - 0.1055613458 * a - 0.0638541728 * b).pow(3)
    s = (lightness - 0.0894841775 * a - 1.2914855480 * b).pow(3)
    red = 4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    green = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    blue = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return torch.stack((red, green, blue), 1)


def linear_rgb_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    srgb = torch.where(
        linear <= 0.0031308, 12.92 * linear, 1.055 * linear.clamp_min(0).pow(1 / 2.4) - 0.055
    )
    return srgb.clamp(0, 1)


def oklab_to_rgb(oklab: torch.Tensor) -> torch.Tensor:
    return linear_rgb_to_srgb(oklab_to_linear_rgb(oklab))


class ConvResidual(nn.Module):
    def __init__(self, width: int) -> None:
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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.block(value)


@dataclass
class AdvancedColorOutput:
    rgb: torch.Tensor
    oklab: torch.Tensor
    linear_rgb: torch.Tensor
    descriptor: torch.Tensor
    palette: torch.Tensor
    masks16: torch.Tensor
    chroma_logits16: torch.Tensor
    layer_weights: torch.Tensor


@dataclass
class FullColorLatent:
    """Full-resolution oracle state used as a target for the EEG encoder."""

    grid: torch.Tensor  # [B,16,16,width]
    global_feature: torch.Tensor  # [B,width]


class DINOColorOracleV2(nn.Module):
    """Dual pixel/query decoder with a categorical low-resolution chroma path."""

    def __init__(
        self,
        input_dim: int = 384,
        layers: int = 4,
        width: int = 192,
        heads: int = 6,
        transformer_depth: int = 2,
        palette_queries: int = 16,
        chroma_bins: int = 32,
        output_size: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if output_size not in (32, 64):
            raise ValueError("output_size must be 32 or 64")
        self.input_dim, self.layers, self.width, self.heads = input_dim, layers, width, heads
        self.transformer_depth, self.palette_queries = transformer_depth, palette_queries
        self.chroma_bins, self.output_size, self.dropout = chroma_bins, output_size, dropout
        self.patch_projections = nn.ModuleList(
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
        encoder_layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.token_mixer = nn.TransformerEncoder(
            encoder_layer, transformer_depth, enable_nested_tensor=False
        )
        self.color_queries = nn.Parameter(torch.randn(1, palette_queries, width) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(
            width, heads, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.query_decoder = nn.TransformerDecoder(decoder_layer, 2)
        self.palette_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3)
        )
        self.mask_features = nn.Conv2d(width, width, 1)
        self.mask_queries = nn.Linear(width, width)
        self.chroma_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 2 * chroma_bins, 1))
        self.chroma_gate = nn.Parameter(torch.tensor(0.0))
        self.pixel16 = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), ConvResidual(width))
        self.up32 = nn.Sequential(
            nn.ConvTranspose2d(width, width // 2, 4, 2, 1),
            ConvResidual(width // 2),
            ConvResidual(width // 2),
        )
        if output_size == 64:
            self.up64 = nn.Sequential(
                nn.ConvTranspose2d(width // 2, width // 4, 4, 2, 1),
                ConvResidual(width // 4),
                ConvResidual(width // 4),
            )
            final_width = width // 4
        else:
            self.up64 = nn.Identity()
            final_width = width // 2
        self.residual_head = nn.Sequential(
            nn.GroupNorm(min(8, final_width), final_width),
            nn.SiLU(),
            nn.Conv2d(final_width, 3, 3, padding=1),
        )
        self.register_buffer(
            "chroma_centers", torch.linspace(-0.4, 0.4, chroma_bins), persistent=False
        )

    def _fuse(
        self, values: torch.Tensor, projections: nn.ModuleList, weights: torch.Tensor
    ) -> torch.Tensor:
        projected = torch.stack(
            [projections[i](values[:, i].float()) for i in range(self.layers)], 1
        )
        shape = [1, self.layers] + [1] * (projected.ndim - 2)
        return (projected * weights.reshape(shape)).sum(1)

    def _encode(self, patches: torch.Tensor, cls: torch.Tensor):
        if patches.ndim != 5 or patches.shape[1] != self.layers or patches.shape[2:4] != (16, 16):
            raise ValueError("patches must have shape [B,layers,16,16,features]")
        weights = self.layer_logits.softmax(-1)
        patch_values = []
        for scale in range(3):
            patch_values.append(self._fuse(patches, self.patch_projections, weights[scale]))
        tokens = patch_values[0].flatten(1, 2) + self.position
        global_feature = self._fuse(cls, self.cls_projections, weights[1])
        tokens = self.token_mixer(tokens + global_feature[:, None])
        feature16 = tokens.transpose(1, 2).reshape(len(tokens), self.width, 16, 16)
        # A second layer mixture enters as an explicit skip instead of being erased by attention.
        skip = patch_values[2].permute(0, 3, 1, 2)
        feature16 = feature16 + 0.25 * skip
        latent = FullColorLatent(tokens.reshape(len(tokens), 16, 16, self.width), global_feature)
        return latent, feature16, tokens, weights

    def encode_full_latent(self, patches: torch.Tensor, cls: torch.Tensor) -> FullColorLatent:
        """Return the full 16x16 state without the shallow pixel-decoder skip."""
        return self._encode(patches, cls)[0]

    def _render(
        self,
        latent: FullColorLatent,
        feature16: torch.Tensor,
        memory: torch.Tensor,
        weights: torch.Tensor,
        hard_chroma: bool = False,
    ) -> AdvancedColorOutput:
        tokens, global_feature = memory, latent.global_feature
        queries = self.query_decoder(
            self.color_queries.expand(len(tokens), -1, -1) + global_feature[:, None], tokens
        )
        palette_raw = self.palette_head(queries)
        palette = torch.cat((palette_raw[..., :1].sigmoid(), 0.4 * palette_raw[..., 1:].tanh()), -1)
        masks = (
            torch.einsum(
                "bchw,bkc->bkhw",
                self.mask_features(feature16),
                F.normalize(self.mask_queries(queries), dim=-1),
            )
            / self.width**0.5
        )
        masks = masks.softmax(1)
        base = torch.einsum("bkhw,bkc->bchw", masks, palette)

        logits = self.chroma_head(feature16).reshape(len(tokens), 2, self.chroma_bins, 16, 16)
        probabilities = logits.softmax(2)
        if hard_chroma:
            indices = probabilities.argmax(2)
            categorical = self.chroma_centers[indices]
        else:
            categorical = (probabilities * self.chroma_centers[None, None, :, None, None]).sum(2)
        gate = torch.sigmoid(self.chroma_gate)
        base = torch.cat((base[:, :1], (1 - gate) * base[:, 1:] + gate * categorical), 1)
        base = F.interpolate(
            base, (self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        residual = torch.tanh(self.residual_head(self.up64(self.up32(self.pixel16(feature16)))))
        scale = residual.new_tensor([0.12, 0.16, 0.16])[None, :, None, None]
        oklab = base + scale * residual
        oklab = torch.cat((oklab[:, :1].clamp(0, 1), oklab[:, 1:].clamp(-0.5, 0.5)), 1)
        linear_rgb = oklab_to_linear_rgb(oklab)
        rgb = linear_rgb_to_srgb(linear_rgb)
        return AdvancedColorOutput(
            rgb, oklab, linear_rgb, color_descriptor(rgb), palette, masks, logits, weights
        )

    def decode_full_latent(
        self, latent: FullColorLatent, hard_chroma: bool = False
    ) -> AdvancedColorOutput:
        """Render a predicted full latent using the frozen antiring decoder.

        The original shallow DINO skip is intentionally absent: it cannot be
        available from EEG. The 16x16 latent is used for both palette attention
        and the pixel path, so inference has no hidden image-derived input.
        """
        grid = latent.grid.float()
        if grid.ndim != 4 or grid.shape[1:3] != (16, 16) or grid.shape[-1] != self.width:
            raise ValueError("latent grid must have shape [B,16,16,width]")
        if latent.global_feature.shape != (len(grid), self.width):
            raise ValueError("latent global_feature must have shape [B,width]")
        memory = grid.flatten(1, 2)
        feature16 = grid.permute(0, 3, 1, 2)
        weights = self.layer_logits.softmax(-1)
        return self._render(latent, feature16, memory, weights, hard_chroma)

    def forward(
        self, patches: torch.Tensor, cls: torch.Tensor, hard_chroma: bool = False
    ) -> AdvancedColorOutput:
        latent, feature16, memory, weights = self._encode(patches, cls)
        return self._render(latent, feature16, memory, weights, hard_chroma)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "layers": self.layers,
            "width": self.width,
            "heads": self.heads,
            "transformer_depth": self.transformer_depth,
            "palette_queries": self.palette_queries,
            "chroma_bins": self.chroma_bins,
            "output_size": self.output_size,
            "dropout": self.dropout,
        }


def _ssim_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mu_x, mu_y = F.avg_pool2d(prediction, 3, 1, 1), F.avg_pool2d(target, 3, 1, 1)
    var_x = F.avg_pool2d(prediction.square(), 3, 1, 1) - mu_x.square()
    var_y = F.avg_pool2d(target.square(), 3, 1, 1) - mu_y.square()
    covariance = F.avg_pool2d(prediction * target, 3, 1, 1) - mu_x * mu_y
    ssim = (
        (2 * mu_x * mu_y + 0.01**2)
        * (2 * covariance + 0.03**2)
        / ((mu_x.square() + mu_y.square() + 0.01**2) * (var_x + var_y + 0.03**2)).clamp_min(1e-6)
    )
    return 1 - ssim.mean()


def _artifact_terms(
    output: AdvancedColorOutput, target_oklab: torch.Tensor, hard_fraction: float = 0.08
) -> dict[str, torch.Tensor]:
    pred_chroma, target_chroma = output.oklab[:, 1:], target_oklab[:, 1:]
    pred_grad = (
        pred_chroma[..., 1:, :] - pred_chroma[..., :-1, :],
        pred_chroma[..., :, 1:] - pred_chroma[..., :, :-1],
    )
    true_grad = (
        target_chroma[..., 1:, :] - target_chroma[..., :-1, :],
        target_chroma[..., :, 1:] - target_chroma[..., :, :-1],
    )
    chroma_gradient = sum(F.smooth_l1_loss(p, t, beta=0.01) for p, t in zip(pred_grad, true_grad))
    spurious = sum(F.relu(p.abs() - t.abs() - 0.008).mean() for p, t in zip(pred_grad, true_grad))
    delta = (output.oklab - target_oklab).square().sum(1).add(1e-8).sqrt().flatten(1)
    count = max(1, round(delta.shape[1] * hard_fraction))
    hard_pixels = delta.topk(count, dim=1).values.mean()
    linear = output.linear_rgb
    gamut = (F.relu(-linear) + F.relu(linear - 1)).mean()
    return {
        "chroma_gradient": chroma_gradient,
        "chroma_edge_artifact": spurious,
        "hard_oklab": hard_pixels,
        "gamut_violation": gamut,
    }


def color_artifact_metrics(
    output_oklab: torch.Tensor, target_rgb: torch.Tensor, linear_rgb: torch.Tensor | None = None
) -> dict[str, float]:
    target_oklab = rgb_to_oklab(target_rgb.float())
    pred = output_oklab.float()
    pred_grad = (
        pred[:, 1:, ..., 1:, :] - pred[:, 1:, ..., :-1, :],
        pred[:, 1:, ..., :, 1:] - pred[:, 1:, ..., :, :-1],
    )
    true_grad = (
        target_oklab[:, 1:, ..., 1:, :] - target_oklab[:, 1:, ..., :-1, :],
        target_oklab[:, 1:, ..., :, 1:] - target_oklab[:, 1:, ..., :, :-1],
    )
    artifact = sum(F.relu(p.abs() - t.abs() - 0.008).mean() for p, t in zip(pred_grad, true_grad))
    gradient = sum(F.l1_loss(p, t) for p, t in zip(pred_grad, true_grad))
    delta = (pred - target_oklab).square().sum(1).sqrt().flatten()
    gamut = (
        pred.new_zeros(())
        if linear_rgb is None
        else (F.relu(-linear_rgb.float()) + F.relu(linear_rgb.float() - 1)).mean()
    )
    return {
        "chroma_edge_artifact": float(artifact),
        "chroma_gradient_mae": float(gradient),
        "oklab_delta_p95": float(torch.quantile(delta, 0.95)),
        "gamut_violation": float(gamut),
    }


def advanced_color_loss(
    output: AdvancedColorOutput,
    target_rgb: torch.Tensor,
    chroma_gradient_weight: float = 0.45,
    hard_pixel_weight: float = 0.60,
    spurious_edge_weight: float = 0.65,
    gamut_weight: float = 0.30,
    hard_fraction: float = 0.08,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target_rgb.float()
    target_oklab = rgb_to_oklab(target)
    rgb = F.smooth_l1_loss(output.rgb, target, beta=0.04)
    lightness = F.smooth_l1_loss(output.oklab[:, :1], target_oklab[:, :1], beta=0.025)
    chroma = F.smooth_l1_loss(output.oklab[:, 1:], target_oklab[:, 1:], beta=0.015)
    multi = (
        sum(
            F.smooth_l1_loss(
                F.adaptive_avg_pool2d(output.oklab, size),
                F.adaptive_avg_pool2d(target_oklab, size),
                beta=0.02,
            )
            for size in (4, 8, 16)
        )
        / 3
    )
    ssim = _ssim_loss(output.rgb, target)
    target16 = F.adaptive_avg_pool2d(target_oklab[:, 1:], 16)
    index = (
        ((target16 + 0.4) / 0.8 * (output.chroma_logits16.shape[2] - 1))
        .round()
        .long()
        .clamp(0, output.chroma_logits16.shape[2] - 1)
    )
    categorical = (
        sum(
            F.cross_entropy(output.chroma_logits16[:, channel], index[:, channel])
            for channel in range(2)
        )
        / 2
    )
    target_palette = F.adaptive_avg_pool2d(target_oklab, 4).flatten(2).transpose(1, 2)
    distances = torch.cdist(output.palette.float(), target_palette.float(), p=1)
    palette = distances.min(2).values.mean() + distances.min(1).values.mean()
    histogram = (
        (soft_chroma_histogram(output.oklab) - soft_chroma_histogram(target_oklab)).abs().mean()
    )
    predicted_chroma = output.oklab[:, 1:].square().sum(1).sqrt()
    true_chroma = target_oklab[:, 1:].square().sum(1).sqrt()
    colorfulness = (
        (predicted_chroma.mean((1, 2)) - true_chroma.mean((1, 2))).abs()
        + (predicted_chroma.std((1, 2)) - true_chroma.std((1, 2))).abs()
    ).mean()
    usage = output.masks16.mean((2, 3)).clamp_min(1e-8)
    effective = (-usage * usage.log()).sum(1).exp()
    query_collapse = F.relu(4 - effective).mean() / 4
    descriptor = F.smooth_l1_loss(output.descriptor, color_descriptor(target), beta=0.03)
    artifacts = _artifact_terms(output, target_oklab, hard_fraction)
    parts = {
        "rgb": rgb,
        "lightness": lightness,
        "chroma": chroma,
        "multi": multi,
        "ssim": ssim,
        "categorical": categorical,
        "palette": palette,
        "histogram": histogram,
        "colorfulness": colorfulness,
        "query_collapse": query_collapse,
        "descriptor": descriptor,
        **artifacts,
    }
    total = (
        1.0 * rgb
        + 0.55 * lightness
        + 1.5 * chroma
        + 1.0 * multi
        + 0.25 * ssim
        + 0.18 * categorical
        + 0.35 * palette
        + 0.45 * histogram
        + 0.35 * colorfulness
        + 0.08 * query_collapse
        + 0.35 * descriptor
        + chroma_gradient_weight * artifacts["chroma_gradient"]
        + hard_pixel_weight * artifacts["hard_oklab"]
        + spurious_edge_weight * artifacts["chroma_edge_artifact"]
        + gamut_weight * artifacts["gamut_violation"]
    )
    return total, parts


class PatchDiscriminator(nn.Module):
    def __init__(self, width: int = 48) -> None:
        super().__init__()
        layers = []
        channels = 3
        for out in (width, width * 2, width * 4):
            layers += [
                nn.utils.spectral_norm(nn.Conv2d(channels, out, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = out
        layers.append(nn.utils.spectral_norm(nn.Conv2d(channels, 1, 3, padding=1)))
        self.net = nn.Sequential(*layers)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.net(image.float())
