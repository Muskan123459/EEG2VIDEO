"""Stage 2 — temporal tokenisation.

Splits each 128×440 EEG window into ``n_patches`` non-overlapping temporal
patches of length ``patch_len`` and linearly projects every patch to a
``d_model``-dim token, yielding a structured (B, 128, 10, 512) token array.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CFG


class Stage2Tokenization(nn.Module):
    def __init__(self, d_model: int = CFG.d_model, patch_len: int = CFG.patch_len):
        super().__init__()
        self.d_model = d_model
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 128, 440)
        B, C, T = x.shape
        assert C == CFG.n_channels, f"Expected {CFG.n_channels} channels, got {C}"
        assert T == CFG.window_samples, f"Expected {CFG.window_samples} samples, got {T}"
        x = x.view(B, C, CFG.n_patches, CFG.patch_len)
        return self.proj(x)  # (B, 128, 10, 512)
