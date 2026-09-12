"""Self-contained mixture-of-experts EEG color encoder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .advanced_color import (
    AdvancedColorOutput,
    ConvResidual,
    linear_rgb_to_srgb,
    oklab_to_linear_rgb,
)
from .color import ColorDecoder, color_descriptor, rgb_to_oklab
from .color_specialist import ColorEEGCondition, EEGColorSpecialist, MultiscaleEEGColorEncoder
from .model_utils import CrossAttentionBlock
from .semantic_edge import EEGDINOGridPredictor
from .semantic_encoder import SubjectAdaptiveSemanticEncoder
from .task_adapter import TaskResidualAdapter


@dataclass
class LegacyColorOutput:
    image: AdvancedColorOutput
    condition: ColorEEGCondition


@dataclass
class HybridColorOutput:
    image: AdvancedColorOutput
    condition: ColorEEGCondition
    expert_weights: torch.Tensor  # [B, OKLab channels, experts, H, W]
    legacy_image: AdvancedColorOutput
    direct_image: AdvancedColorOutput
    proposal_oklab: torch.Tensor
    prior_grid8: torch.Tensor


class LegacyColorRefiner(nn.Module):
    """Exact renderer used by the previous color specialist checkpoint."""

    def __init__(
        self,
        style_dim: int = 192,
        width: int = 128,
        heads: int = 4,
        style_tokens: int = 16,
        chroma_bins: int = 32,
        output_size: int = 64,
    ) -> None:
        super().__init__()
        self.style_dim, self.width, self.heads = style_dim, width, heads
        self.style_tokens, self.chroma_bins, self.output_size = (
            style_tokens,
            chroma_bins,
            output_size,
        )
        self.structure = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1), ConvResidual(width), ConvResidual(width)
        )
        self.style_projection = nn.Sequential(nn.LayerNorm(style_dim), nn.Linear(style_dim, width))
        self.pixel_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.combined = nn.Sequential(nn.Conv2d(width * 2, width, 1), ConvResidual(width))
        self.palette_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2)
        )
        self.mask_features = nn.Conv2d(width, width, 1)
        self.mask_queries = nn.Linear(width, width)
        self.chroma_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 2 * chroma_bins, 1))
        self.palette_categorical_gate = nn.Parameter(torch.tensor(0.0))
        self.residual_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 2, 3, padding=1))
        self.alpha_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.alpha_head[-1].weight)
        nn.init.zeros_(self.alpha_head[-1].bias)
        self.global_alpha = nn.Parameter(torch.tensor(-3.0))
        self.register_buffer(
            "chroma_centers", torch.linspace(-0.4, 0.4, chroma_bins), persistent=False
        )

    def forward(
        self, baseline_rgb: torch.Tensor, condition: ColorEEGCondition, hard_chroma: bool = False
    ) -> LegacyColorOutput:
        baseline = F.interpolate(
            baseline_rgb.float(),
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        baseline_oklab = rgb_to_oklab(baseline)
        low = F.adaptive_avg_pool2d(baseline_oklab, 16)
        edges = F.pad((low[:, :1, :, 1:] - low[:, :1, :, :-1]).abs(), (0, 1, 0, 0))
        structure = self.structure(torch.cat((low, edges), 1))
        style = self.style_projection(condition.style_tokens.float())
        pixels = structure.flatten(2).transpose(1, 2)
        attended, _ = self.pixel_attention(pixels, style, style, need_weights=False)
        color_features = attended.transpose(1, 2).reshape(len(baseline), self.width, 16, 16)
        combined = self.combined(torch.cat((structure, color_features), 1))
        palette_ab = 0.4 * self.palette_head(style).tanh()
        masks = (
            torch.einsum(
                "bchw,bkc->bkhw",
                self.mask_features(combined),
                F.normalize(self.mask_queries(style), dim=-1),
            )
            / self.width**0.5
        ).softmax(1)
        palette_chroma = torch.einsum("bkhw,bkc->bchw", masks, palette_ab)
        logits = self.chroma_head(color_features).reshape(
            len(baseline), 2, self.chroma_bins, 16, 16
        )
        probabilities = logits.softmax(2)
        categorical = (
            self.chroma_centers[probabilities.argmax(2)]
            if hard_chroma
            else (probabilities * self.chroma_centers[None, None, :, None, None]).sum(2)
        )
        mixture = torch.sigmoid(self.palette_categorical_gate)
        proposal = (1 - mixture) * palette_chroma + mixture * categorical
        proposal = proposal + 0.10 * torch.tanh(self.residual_head(color_features))
        baseline_chroma = low[:, 1:]
        alpha = torch.sigmoid(self.global_alpha) * torch.sigmoid(self.alpha_head(combined))
        chroma16 = baseline_chroma + alpha * (proposal - baseline_chroma)
        chroma = F.interpolate(
            chroma16, (self.output_size, self.output_size), mode="bilinear", align_corners=False
        )
        oklab = torch.cat((baseline_oklab[:, :1], chroma.clamp(-0.5, 0.5)), 1)
        linear_rgb = oklab_to_linear_rgb(oklab)
        rgb = linear_rgb_to_srgb(linear_rgb)
        mass = masks.sum((2, 3)).clamp_min(1e-6)
        palette_l = torch.einsum("bkhw,bhw->bk", masks, low[:, 0]) / mass
        palette = torch.cat((palette_l[..., None], palette_ab), -1)
        image = AdvancedColorOutput(
            rgb,
            oklab,
            linear_rgb,
            color_descriptor(rgb),
            palette,
            masks,
            logits,
            rgb.new_empty((0, 0)),
        )
        return LegacyColorOutput(image, condition)

    def config(self) -> dict:
        return {
            "style_dim": self.style_dim,
            "width": self.width,
            "heads": self.heads,
            "style_tokens": self.style_tokens,
            "chroma_bins": self.chroma_bins,
            "output_size": self.output_size,
        }


class LegacyColorSpecialist(nn.Module):
    def __init__(self, encoder_config: dict, refiner_config: dict) -> None:
        super().__init__()
        self.encoder = MultiscaleEEGColorEncoder(**encoder_config)
        self.refiner = LegacyColorRefiner(**refiner_config)

    def forward(self, eeg: torch.Tensor, baseline_rgb: torch.Tensor, hard_chroma: bool = False):
        return self.refiner(baseline_rgb, self.encoder(eeg), hard_chroma)


class LegacyColorPrior(nn.Module):
    """A private copy of the old EEG spatial-color pipeline."""

    def __init__(
        self,
        semantic_config: dict,
        predictor_config: dict,
        adapter_config: dict,
        decoder_config: dict,
        specialist_config: dict,
    ) -> None:
        super().__init__()
        semantic = SubjectAdaptiveSemanticEncoder(**semantic_config)
        self.predictor = EEGDINOGridPredictor(semantic, **predictor_config)
        self.adapter = TaskResidualAdapter(**adapter_config)
        self.decoder = ColorDecoder(**decoder_config)
        self.specialist = LegacyColorSpecialist(**specialist_config)

    def forward(self, eeg: torch.Tensor, hard_chroma: bool = False):
        batch, subjects = eeg.shape[:2]
        grid, contexts = self.predictor.forward_ensemble(eeg)
        ids = torch.arange(subjects, device=eeg.device).repeat(batch)
        with torch.no_grad():
            semantic = self.predictor.encoder(eeg.flatten(0, 1), ids)
        semantic = F.normalize(semantic.reshape(batch, subjects, -1).mean(1), dim=-1)
        adapted = self.adapter(grid, contexts, semantic)
        baseline = self.decoder(adapted.grid, adapted.global_feature).rgb32
        return self.specialist(eeg, baseline, hard_chroma), adapted


class TokenColorFusion(nn.Module):
    """Bidirectional token fusion followed by channel-wise spatial expert routing."""

    def __init__(
        self,
        direct_dim: int = 192,
        prior_dim: int = 192,
        width: int = 128,
        heads: int = 6,
        chroma_bins: int = 32,
        output_size: int = 64,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.direct_dim, self.prior_dim, self.width = direct_dim, prior_dim, width
        self.heads, self.chroma_bins, self.output_size, self.dropout = (
            heads,
            chroma_bins,
            output_size,
            dropout,
        )
        self.prior_projection = nn.Linear(prior_dim, direct_dim)
        self.direct_to_prior = CrossAttentionBlock(direct_dim, direct_dim, heads, dropout)
        self.prior_to_direct = CrossAttentionBlock(direct_dim, direct_dim, heads, dropout)
        self.token_fusion = nn.Sequential(
            nn.LayerNorm(2 * direct_dim), nn.Linear(2 * direct_dim, width), nn.GELU()
        )
        self.stem = nn.Sequential(
            nn.Conv2d(width + 6, width, 3, padding=1), ConvResidual(width), ConvResidual(width)
        )
        self.chroma_head = nn.Sequential(ConvResidual(width), nn.Conv2d(width, 2 * chroma_bins, 1))
        self.categorical_gate = nn.Parameter(torch.tensor(-1.5))
        self.register_buffer(
            "chroma_centers", torch.linspace(-0.4, 0.4, chroma_bins), persistent=False
        )
        self.up32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(width, width // 2, 3, padding=1),
            ConvResidual(width // 2),
        )
        if output_size == 64:
            self.up64 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(width // 2, width // 4, 3, padding=1),
                ConvResidual(width // 4),
            )
            final_width = width // 4
        elif output_size == 32:
            self.up64 = nn.Identity()
            final_width = width // 2
        else:
            raise ValueError("output_size must be 32 or 64")
        self.proposal_head = nn.Conv2d(final_width, 3, 3, padding=1)
        self.router_head = nn.Conv2d(final_width, 9, 3, padding=1)
        nn.init.zeros_(self.proposal_head.weight)
        nn.init.zeros_(self.proposal_head.bias)
        nn.init.zeros_(self.router_head.weight)
        # Per OKLab channel: prefer the proven prior, but retain gradients for both alternatives.
        self.router_head.bias.data.copy_(torch.tensor([2.0, -1.0, 0.0] * 3))

    def forward(
        self,
        direct_condition: ColorEEGCondition,
        prior_grid8: torch.Tensor,
        legacy: AdvancedColorOutput,
        direct: AdvancedColorOutput,
        hard_chroma: bool = False,
    ) -> tuple[AdvancedColorOutput, torch.Tensor, torch.Tensor]:
        direct_tokens = direct_condition.grid16.flatten(1, 2).float()
        prior16 = F.interpolate(
            prior_grid8.permute(0, 3, 1, 2).float(), (16, 16), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        prior_tokens = self.prior_projection(prior16.flatten(1, 2))
        direct_attended = self.direct_to_prior(direct_tokens, prior_tokens)
        prior_attended = self.prior_to_direct(prior_tokens, direct_tokens)
        fused = self.token_fusion(torch.cat((direct_attended, prior_attended), -1))
        feature16 = fused.transpose(1, 2).reshape(len(fused), self.width, 16, 16)
        legacy_low = F.adaptive_avg_pool2d(legacy.oklab.float(), 16)
        direct_low = F.adaptive_avg_pool2d(direct.oklab.float(), 16)
        feature16 = self.stem(torch.cat((feature16, legacy_low, direct_low), 1))
        feature64 = self.up64(self.up32(feature16))
        legacy_oklab = F.interpolate(
            legacy.oklab.float(),
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        direct_oklab = F.interpolate(
            direct.oklab.float(),
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        # The proposal may improve either parent, but cannot invent a distant
        # color field unsupported by both EEG experts.  This local envelope is
        # deliberately wider for chroma than ordinary numerical noise.
        scale = feature64.new_tensor([0.18, 0.12, 0.12])[None, :, None, None]
        proposal = legacy_oklab + scale * torch.tanh(self.proposal_head(feature64))
        logits = self.chroma_head(feature16).reshape(len(feature16), 2, self.chroma_bins, 16, 16)
        probabilities = logits.softmax(2)
        categorical16 = (
            self.chroma_centers[probabilities.argmax(2)]
            if hard_chroma
            else (probabilities * self.chroma_centers[None, None, :, None, None]).sum(2)
        )
        categorical = F.interpolate(
            categorical16,
            (self.output_size, self.output_size),
            mode="bilinear",
            align_corners=False,
        )
        proposal = torch.cat(
            (
                proposal[:, :1],
                proposal[:, 1:]
                + torch.sigmoid(self.categorical_gate) * (categorical - proposal[:, 1:]),
            ),
            1,
        )
        envelope = proposal.new_tensor([0.08, 0.06, 0.06])[None, :, None, None]
        lower = torch.minimum(legacy_oklab, direct_oklab) - envelope
        upper = torch.maximum(legacy_oklab, direct_oklab) + envelope
        proposal = torch.maximum(lower, torch.minimum(proposal, upper))
        proposal = torch.cat((proposal[:, :1].clamp(0, 1), proposal[:, 1:].clamp(-0.5, 0.5)), 1)
        weights = (
            self.router_head(feature64)
            .reshape(len(feature64), 3, 3, self.output_size, self.output_size)
            .softmax(2)
        )
        experts = torch.stack((legacy_oklab, direct_oklab, proposal), 2)
        oklab = (weights * experts).sum(2)
        oklab = torch.cat((oklab[:, :1].clamp(0, 1), oklab[:, 1:].clamp(-0.5, 0.5)), 1)
        linear_rgb = oklab_to_linear_rgb(oklab)
        rgb = linear_rgb_to_srgb(linear_rgb)
        palette = F.adaptive_avg_pool2d(oklab, 4).flatten(2).transpose(1, 2)
        image = AdvancedColorOutput(
            rgb,
            oklab,
            linear_rgb,
            color_descriptor(rgb),
            palette,
            direct.masks16,
            logits,
            weights.mean((1, 3, 4)),
        )
        return image, weights, proposal

    def config(self) -> dict:
        return {
            "direct_dim": self.direct_dim,
            "prior_dim": self.prior_dim,
            "width": self.width,
            "heads": self.heads,
            "chroma_bins": self.chroma_bins,
            "output_size": self.output_size,
            "dropout": self.dropout,
        }


class EEGHybridColorEncoder(nn.Module):
    def __init__(
        self,
        direct_config: dict,
        semantic_config: dict,
        predictor_config: dict,
        adapter_config: dict,
        prior_decoder_config: dict,
        legacy_specialist_config: dict,
        fusion_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.direct = EEGColorSpecialist(**direct_config)
        self.prior = LegacyColorPrior(
            semantic_config,
            predictor_config,
            adapter_config,
            prior_decoder_config,
            legacy_specialist_config,
        )
        self.fusion = TokenColorFusion(**(fusion_config or {}))

    def forward(self, eeg: torch.Tensor, hard_chroma: bool = False) -> HybridColorOutput:
        direct = self.direct(eeg, hard_chroma)
        legacy, adapted = self.prior(eeg, hard_chroma)
        image, weights, proposal = self.fusion(
            direct.condition,
            adapted.grid,
            legacy.image,
            direct.image,
            hard_chroma,
        )
        return HybridColorOutput(
            image, direct.condition, weights, legacy.image, direct.image, proposal, adapted.grid
        )

    def config(self) -> dict:
        return {
            "direct_config": self.direct.config(),
            "semantic_config": self.prior.predictor.encoder.config(),
            "predictor_config": self.prior.predictor.config(),
            "adapter_config": self.prior.adapter.config(),
            "prior_decoder_config": self.prior.decoder.config(),
            "legacy_specialist_config": {
                "encoder_config": self.prior.specialist.encoder.config(),
                "refiner_config": self.prior.specialist.refiner.config(),
            },
            "fusion_config": self.fusion.config(),
        }


def initialize_hybrid_color_encoder(
    direct_root: str | Path,
    legacy_root: str | Path,
    device: torch.device | str = "cpu",
) -> EEGHybridColorEncoder:
    direct_root, legacy_root = Path(direct_root), Path(legacy_root)
    direct_state = torch.load(direct_root / "best.pt", map_location="cpu", weights_only=False)
    legacy_state = torch.load(legacy_root / "best.pt", map_location="cpu", weights_only=False)
    components = legacy_root / "components"
    semantic_state = torch.load(
        components / "semantic_encoder.pt", map_location="cpu", weights_only=False
    )
    adapter_state = torch.load(components / "adapter.pt", map_location="cpu", weights_only=False)
    decoder_state = torch.load(
        components / "color_decoder.pt", map_location="cpu", weights_only=False
    )
    model = EEGHybridColorEncoder(
        direct_state["model_config"],
        semantic_state["model_config"],
        adapter_state["predictor_config"],
        adapter_state["adapter_config"],
        decoder_state["model_config"],
        legacy_state["model_config"],
    )
    model.direct.load_state_dict(direct_state.get("ema", direct_state["model"]), strict=True)
    model.prior.predictor.load_state_dict(adapter_state["predictor"], strict=True)
    model.prior.adapter.load_state_dict(adapter_state["adapter"], strict=True)
    model.prior.decoder.load_state_dict(decoder_state["model"], strict=True)
    missing, unexpected = model.prior.specialist.load_state_dict(
        legacy_state.get("ema", legacy_state["model"]), strict=False
    )
    allowed = ("encoder.spatial_",)
    if unexpected or any(not key.startswith(allowed) for key in missing):
        raise RuntimeError(
            f"incompatible legacy specialist: missing={missing}, unexpected={unexpected}"
        )
    return model.to(device)


def load_hybrid_color_encoder(
    root: str | Path = "runs/eeg_color_encoder_hybrid",
    device: torch.device | str = "cpu",
    allow_unqualified: bool = False,
) -> tuple[EEGHybridColorEncoder, dict]:
    root = Path(root)
    checkpoint = root / "best_qualified.pt"
    state = (
        torch.load(checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.exists()
        else None
    )
    safely_qualified = bool(
        state and state.get("objective_version") == 2 and state.get("record", {}).get("qualified")
    )
    if not safely_qualified:
        if not allow_unqualified:
            raise RuntimeError(
                "no safely qualified v2 checkpoint: the hybrid has not yet beaten its "
                "parent-relative constraints"
            )
        checkpoint = root / "best.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("architecture") != "independent_hybrid_color_v1":
        raise RuntimeError("checkpoint is not an independent hybrid color encoder")
    model = EEGHybridColorEncoder(**state["model_config"])
    model.load_state_dict(state.get("ema", state["model"]), strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, state
