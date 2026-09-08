"""DiT adapter for frozen PixArt-alpha (Step 3): decoupled cross-attention.

Every transformer block's ``attn2`` gets a second K,V pair for 8 image tokens
projected from a 768-d CLIP-image embedding. Q is shared; the original text
K,V stay untouched; block output = text attention + lambda_i * image attention
with a LEARNED PER-BLOCK lambda (28 params), zero-initialized so the adapter is
invisible at start (bitwise-equal generation at lambda=0 — verified before
training). New K,V start as copies of the block's text K,V weights.

Training (no EEG anywhere): epsilon-prediction MSE on VAE latents of the SAME
image whose CLIP embedding conditions the adapter; the text slot is pinned to
PixArt's own learned null (``caption_projection.y_embedding``); conditioning
dropout ~10% (CLIP embedding zeroed BEFORE projection) enables CFG.
"""

from __future__ import annotations

import lightning as L
import torch
from torch import nn
from torch.nn import functional as F


class IPTokenHolder:
    """Mutable channel delivering per-batch image tokens to the processors."""

    def __init__(self) -> None:
        self.tokens: torch.Tensor | None = None
        self.scale_override: float | None = None


class IPAttnProcessor(nn.Module):
    """Decoupled cross-attention via DELEGATION to the factory processor.

    Text path = the ORIGINAL installed ``AttnProcessor2_0`` (bitwise equality at
    lambda=0 holds by construction, whatever the diffusers version does inside).
    Image contribution is added through the linearity of ``to_out``:
    ``to_out(text + s*ip) == to_out(text) + s*W_out@ip`` (bias counted once in
    the factory call). Requires ``residual_connection=False`` and
    ``rescale_output_factor==1`` on the wrapped Attention — asserted at wiring.
    """

    def __init__(self, hidden_size: int, holder: IPTokenHolder) -> None:
        super().__init__()
        from diffusers.models.attention_processor import AttnProcessor2_0

        self.base = AttnProcessor2_0()
        self.holder = holder
        self.to_k_ip = nn.Linear(hidden_size, hidden_size, bias=True)
        self.to_v_ip = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ip_scale = nn.Parameter(torch.zeros(()))

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        text_out = self.base(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        tokens = self.holder.tokens
        scale = self.ip_scale if self.holder.scale_override is None else self.holder.scale_override
        if tokens is None or (not torch.is_tensor(scale) and scale == 0.0):
            return text_out

        batch_size = hidden_states.shape[0]
        head_dim = attn.to_q.out_features // attn.heads
        # Sciezka obrazowa w fp32 (parametry adaptera fp32, transformer bf16).
        query = attn.to_q(hidden_states).float()
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        ip_key = (
            self.to_k_ip(tokens.float()).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        )
        ip_value = (
            self.to_v_ip(tokens.float()).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        )
        ip_hidden = F.scaled_dot_product_attention(
            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
        )
        ip_hidden = ip_hidden.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        ip_out = F.linear(ip_hidden, attn.to_out[0].weight.float())  # bez biasu (liniowosc)
        return text_out + (scale * ip_out).to(text_out.dtype)


class ImageProjection(nn.Module):
    """CLIP 768-d -> N image tokens x hidden (default 8 x 1152)."""

    def __init__(self, clip_dim: int = 768, hidden_size: int = 1152, tokens: int = 8) -> None:
        super().__init__()
        self.tokens = tokens
        self.hidden_size = hidden_size
        self.proj = nn.Linear(clip_dim, tokens * hidden_size)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, clip_embeds: torch.Tensor) -> torch.Tensor:
        out = self.proj(clip_embeds).reshape(len(clip_embeds), self.tokens, self.hidden_size)
        return self.norm(out)


class DiTAdapterLightning(L.LightningModule):
    def __init__(
        self,
        repo: str = "PixArt-alpha/PixArt-XL-2-512x512",
        null_embeddings_file: str = "artifacts/dit_adapter/t5_embeddings.pt",
        null_key: str = "null_learned",
        clip_dim: int = 768,
        image_tokens: int = 8,
        cond_dropout: float = 0.10,
        gradient_checkpointing: bool = False,  # konflikt z custom attn processorem
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        from diffusers import DDPMScheduler, PixArtTransformer2DModel

        self.transformer = PixArtTransformer2DModel.from_pretrained(
            repo, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.transformer.requires_grad_(False)
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        self.noise_scheduler = DDPMScheduler.from_pretrained(repo, subfolder="scheduler")

        hidden = self.transformer.config.num_attention_heads * (
            self.transformer.config.attention_head_dim
        )
        self.holder = IPTokenHolder()
        self.projection = ImageProjection(clip_dim, hidden, image_tokens)
        processors = []
        for block in self.transformer.transformer_blocks:
            if block.attn2.residual_connection or block.attn2.rescale_output_factor != 1.0:
                raise RuntimeError(
                    "IPAttnProcessor zaklada residual_connection=False i "
                    "rescale_output_factor=1 na attn2 (liniowosc to_out)"
                )
            processor = IPAttnProcessor(hidden, self.holder)
            with torch.no_grad():
                processor.to_k_ip.weight.copy_(block.attn2.to_k.weight.float())
                processor.to_k_ip.bias.copy_(block.attn2.to_k.bias.float())
                processor.to_v_ip.weight.copy_(block.attn2.to_v.weight.float())
                processor.to_v_ip.bias.copy_(block.attn2.to_v.bias.float())
            block.attn2.set_processor(processor)
            processors.append(processor)
        self.processors = nn.ModuleList(processors)

        blob = torch.load(null_embeddings_file, map_location="cpu", weights_only=False)
        null = blob[null_key]
        self.register_buffer("null_text", null["embeds"].float(), persistent=False)
        self.register_buffer("null_mask", null["mask"], persistent=False)

    # --- adapter state (bez zamrozonego transformera) ---
    def adapter_parameters(self) -> list[nn.Parameter]:
        return list(self.projection.parameters()) + list(self.processors.parameters())

    def state_dict(self, *args, **kwargs):
        full = super().state_dict(*args, **kwargs)
        return {k: v for k, v in full.items() if not k.startswith("transformer.")}

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=False, assign=assign)
        missing, unexpected = result.missing_keys, result.unexpected_keys
        real_missing = [k for k in missing if not k.startswith("transformer.")]
        if strict and (real_missing or unexpected):
            raise RuntimeError(
                f"adapter checkpoint mismatch: missing={real_missing} unexpected={unexpected}"
            )
        # Lightning oczekuje _IncompatibleKeys, nie krotki; zamrozony transformer
        # nigdy nie jest w checkpoincie, wiec jego klucze nie sa brakami.
        from torch.nn.modules.module import _IncompatibleKeys

        return _IncompatibleKeys(real_missing, unexpected)

    def lambda_profile(self) -> list[float]:
        return [float(p.ip_scale.detach()) for p in self.processors]

    def denoise_forward(
        self, latents: torch.Tensor, timesteps: torch.Tensor, clip_embeds: torch.Tensor | None
    ) -> torch.Tensor:
        batch = len(latents)
        if clip_embeds is not None:
            self.holder.tokens = self.projection(clip_embeds.float())
        else:
            self.holder.tokens = None
        text = self.null_text.to(self.device, torch.bfloat16).expand(batch, -1, -1)
        mask = self.null_mask.to(self.device).expand(batch, -1)
        out = self.transformer(
            latents.to(torch.bfloat16),
            encoder_hidden_states=text,
            encoder_attention_mask=mask,
            timestep=timesteps,
            added_cond_kwargs={"resolution": None, "aspect_ratio": None},
        ).sample
        self.holder.tokens = None
        return out.chunk(2, dim=1)[0]  # PixArt zwraca [eps, sigma] w kanalach

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        moments, clip_embeds = batch
        mean, std = moments[:, :4].float(), moments[:, 4:].float()
        latents = (mean + std * torch.randn_like(std)) * 0.18215
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (len(latents),), device=self.device
        )
        noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
        drop = torch.rand(len(latents), device=self.device) < self.hparams.cond_dropout
        clip_in = clip_embeds.float().clone()
        clip_in[drop] = 0.0
        prediction = self.denoise_forward(noisy, timesteps, clip_in)
        loss = F.mse_loss(prediction.float(), noise)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        moments, clip_embeds = batch
        mean = moments[:, :4].float() * 0.18215  # deterministycznie: srodek posteriora
        generator = torch.Generator(self.device.type).manual_seed(20260714 + batch_idx)
        noise = torch.randn(mean.shape, generator=generator, device=self.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (len(mean),),
            generator=generator,
            device=self.device,
        )
        noisy = self.noise_scheduler.add_noise(mean, noise, timesteps)
        prediction = self.denoise_forward(noisy, timesteps, clip_embeds.float())
        loss = F.mse_loss(prediction.float(), noise)
        self.log("val/loss", loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.adapter_parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )

    def _vae(self):
        if not hasattr(self, "_vae_model"):
            from diffusers import AutoencoderKL

            self._vae_model = (
                AutoencoderKL.from_pretrained(
                    self.hparams.repo, subfolder="vae", torch_dtype=torch.bfloat16
                )
                .to(self.device)
                .eval()
            )
        return self._vae_model

    @torch.no_grad()
    def decode_and_grid(self, latents: torch.Tensor):
        """Decode [B,4,64,64] latents to a horizontal PIL grid of 512px images."""
        from PIL import Image

        images = self._vae().decode((latents / 0.18215).to(torch.bfloat16), return_dict=False)[0]
        images = ((images.float() / 2 + 0.5).clamp(0, 1) * 255).round().byte()
        images = images.permute(0, 2, 3, 1).cpu().numpy()
        grid = Image.new("RGB", (len(images) * 512, 512))
        for i, arr in enumerate(images):
            grid.paste(Image.fromarray(arr), (i * 512, 0))
        return grid

    @torch.no_grad()
    def sample_images(
        self,
        clip_embeds: torch.Tensor,
        steps: int = 20,
        guidance: float = 4.5,
        seed: int = 20260714,
        lambda_override: float | None = None,
    ) -> torch.Tensor:
        """DPM-Solver sampling with image-CFG (uncond = zeroed CLIP embedding).

        Returns latents [B, 4, 64, 64] (divide by 0.18215 and VAE-decode outside).
        """
        from diffusers import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            self.hparams.repo, subfolder="scheduler"
        )
        scheduler.set_timesteps(steps, device=self.device)
        generator = torch.Generator(self.device.type).manual_seed(seed)
        latents = torch.randn(
            (len(clip_embeds), 4, 64, 64), generator=generator, device=self.device
        )
        latents = latents * scheduler.init_noise_sigma
        self.holder.scale_override = lambda_override
        uncond = torch.zeros_like(clip_embeds)
        for t in scheduler.timesteps:
            model_in = scheduler.scale_model_input(latents, t)
            tt = t.expand(len(latents))
            eps_cond = self.denoise_forward(model_in, tt, clip_embeds)
            eps_uncond = self.denoise_forward(model_in, tt, uncond)
            eps = eps_uncond + guidance * (eps_cond - eps_uncond)
            latents = scheduler.step(eps.float(), t, latents).prev_sample
        self.holder.scale_override = None
        return latents
