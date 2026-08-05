"""Compact, EEG-predictable color tokens distilled from the rich DINO oracle."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .advanced_color import (
    AdvancedColorOutput,
    ConvResidual,
    linear_rgb_to_srgb,
    oklab_to_linear_rgb,
)
from .color import color_descriptor


@dataclass
class CompactColorTokens:
    """Factorized interface that the EEG model will learn to predict later."""

    scene: torch.Tensor  # [B, 8, 8, D], shared with the future edge branch
    color: torch.Tensor  # [B, K, D], color-only residual tokens


@dataclass
class CompactColorForward:
    image: AdvancedColorOutput
    tokens: CompactColorTokens
    decoded_tokens: CompactColorTokens
    scene_lightness: torch.Tensor


class RichDINOColorTokenizer(nn.Module):
    """Compress four 16x16 DINO maps into a spatial scene grid and color tokens."""

    def __init__(
        self,
        input_dim: int = 384,
        layers: int = 4,
        width: int = 192,
        heads: int = 6,
        depth: int = 2,
        color_tokens: int = 16,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim, self.layers, self.width = input_dim, layers, width
        self.heads, self.depth, self.color_tokens = heads, depth, color_tokens
        self.dropout = dropout
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
        # Separate mixtures stop the scene grid and color residual from being aliases.
        self.layer_logits = nn.Parameter(torch.zeros(3, layers))
        self.scene_position = nn.Parameter(torch.randn(1, 64, width) * 0.02)
        self.memory_position = nn.Parameter(torch.randn(1, 256, width) * 0.02)
        encoder = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.scene_mixer = nn.TransformerEncoder(encoder, depth, enable_nested_tensor=False)
        self.color_queries = nn.Parameter(torch.randn(1, color_tokens, width) * 0.02)
        decoder = nn.TransformerDecoderLayer(
            width, heads, width * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.color_mixer = nn.TransformerDecoder(decoder, max(1, depth))
        self.scene_norm = nn.LayerNorm(width)
        self.color_norm = nn.LayerNorm(width)

    def _project(
        self, value: torch.Tensor, projections: nn.ModuleList, weights: torch.Tensor
    ) -> torch.Tensor:
        projected = torch.stack(
            [projections[index](value[:, index].float()) for index in range(self.layers)], 1
        )
        shape = [1, self.layers] + [1] * (projected.ndim - 2)
        return (projected * weights.reshape(shape)).sum(1)

    def forward(self, patches: torch.Tensor, cls: torch.Tensor) -> CompactColorTokens:
        if patches.ndim != 5 or patches.shape[1] != self.layers or patches.shape[2:4] != (16, 16):
            raise ValueError("patches must have shape [B,layers,16,16,features]")
        if cls.shape[:2] != patches.shape[:2]:
            raise ValueError("cls must have shape [B,layers,features]")
        weights = self.layer_logits.softmax(-1)
        scene16 = self._project(patches, self.patch_projections, weights[0])
        scene8 = F.avg_pool2d(scene16.permute(0, 3, 1, 2), 2).flatten(2).transpose(1, 2)
        global_scene = self._project(cls, self.cls_projections, weights[1])
        scene = self.scene_mixer(scene8 + self.scene_position + global_scene[:, None])
        scene = self.scene_norm(scene)

        memory = self._project(patches, self.patch_projections, weights[2]).flatten(1, 2)
        memory = memory + self.memory_position
        queries = self.color_queries.expand(len(patches), -1, -1) + global_scene[:, None]
        # Full-resolution memory determines the palette; scene tokens supply placement context.
        color = self.color_mixer(queries, torch.cat((memory, scene), 1))
        color = self.color_norm(color)
        return CompactColorTokens(scene.reshape(len(patches), 8, 8, self.width), color)


class CompactColorRenderer(nn.Module):
    """Render RGB using only compact scene and color tokens (never DINO features)."""

    def __init__(
        self,
        width: int = 192,
        heads: int = 6,
        color_tokens: int = 16,
        chroma_bins: int = 32,
        output_size: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if output_size not in (32, 64):
            raise ValueError("output_size must be 32 or 64")
        self.width, self.heads, self.color_tokens = width, heads, color_tokens
        self.chroma_bins, self.output_size, self.dropout = chroma_bins, output_size, dropout
        self.scene_stem = nn.Sequential(nn.Conv2d(width, width, 1), ConvResidual(width))
        self.up16 = nn.Sequential(
            nn.ConvTranspose2d(width, width, 4, 2, 1), ConvResidual(width), ConvResidual(width)
        )
        self.pixel_color_attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.color_norm = nn.LayerNorm(width)
        self.palette_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 3)
        )
        self.mask_features = nn.Conv2d(width, width, 1)
        self.mask_queries = nn.Linear(width, width)
        self.scene_lightness_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 1, 1))
        self.chroma_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 2 * chroma_bins, 1))
        self.chroma_gate = nn.Parameter(torch.tensor(0.0))
        self.pixel16 = nn.Sequential(ConvResidual(width), nn.Conv2d(width, width, 3, padding=1))
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
        self.chroma_residual = nn.Sequential(
            nn.GroupNorm(min(8, final_width), final_width),
            nn.SiLU(),
            nn.Conv2d(final_width, 2, 3, padding=1),
        )
        self.register_buffer(
            "chroma_centers", torch.linspace(-0.4, 0.4, chroma_bins), persistent=False
        )

    def forward(
        self, tokens: CompactColorTokens, hard_chroma: bool = False
    ) -> tuple[AdvancedColorOutput, torch.Tensor]:
        scene, color = tokens.scene.float(), tokens.color.float()
        if scene.ndim != 4 or scene.shape[1:3] != (8, 8) or scene.shape[-1] != self.width:
            raise ValueError("scene tokens must have shape [B,8,8,width]")
        if color.ndim != 3 or color.shape[1:] != (self.color_tokens, self.width):
            raise ValueError("color tokens must have shape [B,color_tokens,width]")
        scene16 = self.up16(self.scene_stem(scene.permute(0, 3, 1, 2)))
        spatial = scene16.flatten(2).transpose(1, 2)
        color_context, _ = self.pixel_color_attention(spatial, color, color, need_weights=False)
        combined16 = (
            self.color_norm(spatial + color_context)
            .transpose(1, 2)
            .reshape(len(scene), self.width, 16, 16)
        )
        # Values come only from color tokens; scene queries decide where to place them.
        # This prevents the shared scene grid from becoming a second palette code.
        color16 = (
            self.color_norm(color_context).transpose(1, 2).reshape(len(scene), self.width, 16, 16)
        )

        palette_raw = self.palette_head(color)
        palette = torch.cat((palette_raw[..., :1].sigmoid(), 0.4 * palette_raw[..., 1:].tanh()), -1)
        masks = (
            torch.einsum(
                "bchw,bkc->bkhw",
                self.mask_features(combined16),
                F.normalize(self.mask_queries(color), dim=-1),
            )
            / self.width**0.5
        )
        masks = masks.softmax(1)
        palette_chroma = torch.einsum("bkhw,bkc->bchw", masks, palette[..., 1:])

        logits = self.chroma_head(color16).reshape(len(scene), 2, self.chroma_bins, 16, 16)
        probabilities = logits.softmax(2)
        if hard_chroma:
            indices = probabilities.argmax(2)
            categorical = self.chroma_centers[indices]
        else:
            categorical = (probabilities * self.chroma_centers[None, None, :, None, None]).sum(2)
        gate = torch.sigmoid(self.chroma_gate)
        chroma16 = (1 - gate) * palette_chroma + gate * categorical
        scene_lightness = self.scene_lightness_head(scene16).sigmoid()
        base = torch.cat((scene_lightness, chroma16), 1)
        base = F.interpolate(
            base, (self.output_size, self.output_size), mode="bilinear", align_corners=False
        )

        pixels = self.up64(self.up32(self.pixel16(color16)))
        lightness = base[:, :1]
        chroma = base[:, 1:] + 0.14 * torch.tanh(self.chroma_residual(pixels))
        oklab = torch.cat((lightness.clamp(0, 1), chroma.clamp(-0.5, 0.5)), 1)
        linear_rgb = oklab_to_linear_rgb(oklab)
        rgb = linear_rgb_to_srgb(linear_rgb)
        layer_weights = rgb.new_empty((0, 0))
        image = AdvancedColorOutput(
            rgb, oklab, linear_rgb, color_descriptor(rgb), palette, masks, logits, layer_weights
        )
        return image, scene_lightness


class CompactColorAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 384,
        layers: int = 4,
        width: int = 192,
        heads: int = 6,
        tokenizer_depth: int = 2,
        color_tokens: int = 16,
        chroma_bins: int = 32,
        output_size: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim, self.layers, self.width = input_dim, layers, width
        self.heads, self.tokenizer_depth, self.color_tokens = heads, tokenizer_depth, color_tokens
        self.chroma_bins, self.output_size, self.dropout = chroma_bins, output_size, dropout
        self.tokenizer = RichDINOColorTokenizer(
            input_dim, layers, width, heads, tokenizer_depth, color_tokens, dropout
        )
        self.renderer = CompactColorRenderer(
            width, heads, color_tokens, chroma_bins, output_size, dropout
        )

    def encode(self, patches: torch.Tensor, cls: torch.Tensor) -> CompactColorTokens:
        return self.tokenizer(patches, cls)

    @staticmethod
    def corrupt(
        tokens: CompactColorTokens, noise: float = 0.0, dropout: float = 0.0
    ) -> CompactColorTokens:
        scene, color = tokens.scene, tokens.color
        if noise:
            scene = scene + noise * torch.randn_like(scene)
            color = color + noise * torch.randn_like(color)
        if dropout:
            scene = scene * (torch.rand((*scene.shape[:-1], 1), device=scene.device) >= dropout)
            color = color * (torch.rand((*color.shape[:-1], 1), device=color.device) >= dropout)
        return CompactColorTokens(scene, color)

    def decode(
        self, tokens: CompactColorTokens, hard_chroma: bool = False
    ) -> tuple[AdvancedColorOutput, torch.Tensor]:
        return self.renderer(tokens, hard_chroma)

    def forward(
        self,
        patches: torch.Tensor,
        cls: torch.Tensor,
        hard_chroma: bool = False,
        token_noise: float = 0.0,
        token_dropout: float = 0.0,
    ) -> CompactColorForward:
        tokens = self.encode(patches, cls)
        decode_tokens = (
            self.corrupt(tokens, token_noise, token_dropout) if self.training else tokens
        )
        image, scene_lightness = self.decode(decode_tokens, hard_chroma)
        return CompactColorForward(image, tokens, decode_tokens, scene_lightness)

    def config(self) -> dict:
        return {
            "input_dim": self.input_dim,
            "layers": self.layers,
            "width": self.width,
            "heads": self.heads,
            "tokenizer_depth": self.tokenizer_depth,
            "color_tokens": self.color_tokens,
            "chroma_bins": self.chroma_bins,
            "output_size": self.output_size,
            "dropout": self.dropout,
        }


def load_compact_color(
    path: str, device: torch.device | str = "cpu", ema: bool = True
) -> tuple[CompactColorAutoencoder, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = CompactColorAutoencoder(**payload["model_config"])
    model.load_state_dict(payload["ema"] if ema and "ema" in payload else payload["model"])
    return model.to(device).eval(), payload
