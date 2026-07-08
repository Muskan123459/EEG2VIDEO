"""Stage 3 — CARD (Channel-Aligned Robust Blend) encoder.

Each CARD block applies, with residual + LayerNorm:
  1. intra-channel temporal multi-head self-attention (per channel, across tokens);
  2. inter-channel spatial multi-head self-attention (per token, across channels);
  3. 1D convolutional token blending across the temporal axis.
Three blocks are stacked into the full encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import CFG


class CARDBlock(nn.Module):
    def __init__(self, d_model: int = CFG.d_model, nhead: int = CFG.nhead_card):
        super().__init__()
        self.intra_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.inter_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.conv_blender = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C=128, N=10, D=512)
        B, C, N, D = x.shape

        # Intra-channel temporal attention (per channel across 10 tokens)
        intra_in = x.view(B * C, N, D)
        a1, _ = self.intra_attn(intra_in, intra_in, intra_in)
        x = self.norm1(x + a1.view(B, C, N, D))

        # Inter-channel attention (per token across 128 channels)
        inter_in = x.transpose(1, 2).contiguous().view(B * N, C, D)
        a2, _ = self.inter_attn(inter_in, inter_in, inter_in)
        x = self.norm2(x + a2.view(B, N, C, D).transpose(1, 2))

        # 1D convolutional token blending
        conv_in = x.view(B * C, N, D).transpose(1, 2)  # (B*C, D, N)
        conv_out = self.conv_blender(conv_in).transpose(1, 2).view(B, C, N, D)
        x = self.norm3(x + conv_out)
        return x


class Stage3CARDEncoder(nn.Module):
    def __init__(self, num_blocks: int = CFG.num_card_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([CARDBlock() for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
