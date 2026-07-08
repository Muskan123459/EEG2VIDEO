"""Stages 4 & 5 — latent projection and CLIP semantic alignment.

Stage 4 flattens the CARD encoder output and projects it to a 512-d embedding on
the unit hypersphere (L2-normalised). Stage 5 is a two-layer MLP that maps this
EEG embedding into CLIP's image-embedding space for cosine alignment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import CFG


class Stage4LatentProjection(nn.Module):
    def __init__(self, channels: int = CFG.n_channels, n_patches: int = CFG.n_patches, d_model: int = CFG.d_model):
        super().__init__()
        self.proj = nn.Linear(channels * n_patches * d_model, 512)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        z = self.proj(x)
        return F.normalize(z, p=2, dim=1)


class Stage5CLIPAlignment(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.f_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z_eeg: torch.Tensor) -> torch.Tensor:
        return self.f_proj(z_eeg)
