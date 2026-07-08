"""Stage 6 — Video Latent Diffusion Model backbone.

Wraps a frozen Stable Diffusion U-Net + VAE and injects EEG embeddings via
cross-attention. Only the attention/normalisation layers are fine-tuned. A
:class:`TemporalAttentionLayer` refines the rolling EEG context so the conditioning
query carries temporal information across frames.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import CFG, DEVICE
from .temporal import TemporalAttentionLayer


class VAEImageDecoder(nn.Module):
    """Decode latent tensors to RGB frames.

    This uses the Stable Diffusion VAE decoder where available.
    Latent shape expected: (B, 4, H, W)
    Output: (B, 3, H*8, W*8) approximately.
    """
    def __init__(self, vae: object):
        super().__init__()
        self.vae = vae

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents / 0.18215
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs


class VideoLDMDenoiser(nn.Module):
    """Guide-aligned video-conditioned denoiser.

    Uses a Stable Diffusion U-Net backbone and injects EEG embeddings via
    cross-attention. TemporalAttentionLayer refines the EEG context across
    a rolling frame window to improve inter-frame coherence.
    """
    def __init__(self,
                 model_id: str = CFG.stable_diffusion_model,
                 cross_attention_dim: int = CFG.unet_cross_attention_dim):
        super().__init__()
        # Import heavy diffusers classes lazily to avoid import-time errors
        from diffusers import AutoencoderKL
        from diffusers.models import UNet2DConditionModel

        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet.enable_gradient_checkpointing()
        torch.cuda.empty_cache()   # <-- add this line

        # EEG conditioning: 512 -> 768
        self.eeg_proj = nn.Linear(512, cross_attention_dim)
        self.temporal_attn = TemporalAttentionLayer(d_model=cross_attention_dim, nhead=8)

        # Fine-tune only attention + norm layers in U-Net
        for name, param in self.unet.named_parameters():
            if ("attn" in name) or ("norm" in name):
                param.requires_grad = True

    def forward(self,
                noisy_latents: torch.Tensor,
                timesteps: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # encoder_hidden_states: (B, 1, 512)
        cond = self.eeg_proj(encoder_hidden_states)  # (B, 1, 768)
        if temporal_context is not None:
            # temporal_context: (B, T, 512)
            ctx = self.eeg_proj(temporal_context)
            ctx = self.temporal_attn(ctx)
            cond = ctx[:, -1:, :]
        return self.unet(noisy_latents, timesteps, encoder_hidden_states=cond).sample


def build_video_denoiser() -> VideoLDMDenoiser:
    return VideoLDMDenoiser().to(DEVICE)
