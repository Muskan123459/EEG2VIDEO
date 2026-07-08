"""Ablation hooks for the temporal components.

Toggles the causal temporal-attention layer and/or the EMA momentum smoothing at
generation time so their individual contributions can be compared.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEVICE
from .pipeline import Group4Pipeline


def run_ablation(pipeline: Group4Pipeline,
                 eeg_windows: torch.Tensor,
                 with_temporal_attention: bool = True,
                 with_momentum: bool = True) -> List[torch.Tensor]:
    """Ablation hook. Use this to compare:
    - full model



    - without temporal attention
    - without momentum smoothing
    """
    orig_temporal = pipeline.st6.temporal_attn
    if not with_temporal_attention:
        pipeline.st6.temporal_attn = nn.Identity().to(DEVICE)

    frames: List[torch.Tensor] = []
    try:
        if with_momentum:
            frames = pipeline.generate_video_sequence(eeg_windows)
        else:
            # Temporarily force momentum alpha = 0 behavior by bypassing EMA blend
            old_generate = pipeline.generate_video_sequence

            @torch.no_grad()
            def no_momentum_generate(eeg_windows, num_inference_steps=None, context_len=None):
                pipeline.models_eval()
                num_inference_steps = num_inference_steps or pipeline.cfg.num_inference_steps
                context_len = context_len or pipeline.cfg.context_len
                pipeline.scheduler.set_timesteps(num_inference_steps)
                z_history = []
                latents = []
                take = min(eeg_windows.size(0), pipeline.cfg.max_video_frames)
                for i in range(take):
                    x_i = eeg_windows[i:i+1].to(DEVICE)
                    z_i = pipeline.st5(pipeline.st4(pipeline.st3(pipeline.st2(x_i))))
                    z_i = F.normalize(z_i, p=2, dim=1)
                    z_history.append(z_i)
                    ctx_start = max(0, len(z_history) - context_len)
                    temporal_ctx = torch.cat(z_history[ctx_start:], dim=0).unsqueeze(0)
                    latent = torch.randn(1, 4, pipeline.cfg.latent_h, pipeline.cfg.latent_w, device=DEVICE)
                    for t in pipeline.scheduler.timesteps:
                        ts = torch.tensor([t], device=DEVICE).long()
                        noise_pred = pipeline.st6(latent, ts, encoder_hidden_states=z_i.unsqueeze(1), temporal_context=temporal_ctx)
                        latent = pipeline.scheduler.step(noise_pred, t, latent).prev_sample
                    latents.append(latent.squeeze(0).cpu())
                lat_batch = torch.stack(latents, dim=0).to(DEVICE)
                decoded = pipeline.st6.vae.decode(lat_batch / 0.18215).sample
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                return [decoded[i].detach().cpu() for i in range(decoded.size(0))]

            pipeline.generate_video_sequence = no_momentum_generate  # type: ignore[assignment]
            frames = pipeline.generate_video_sequence(eeg_windows)
            pipeline.generate_video_sequence = old_generate  # restore
    finally:
        pipeline.st6.temporal_attn = orig_temporal

    return frames
