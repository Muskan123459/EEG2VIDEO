"""Temporal conditioning components for coherent video generation.

``TemporalAttentionLayer`` is a causal multi-head attention block that processes
the rolling window of recent EEG embeddings; ``EMAEmbeddingBuffer`` maintains an
exponential-moving-average momentum target that smooths the embedding trajectory.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..config import CFG


class TemporalAttentionLayer(nn.Module):
    def __init__(self, d_model: int = CFG.unet_cross_attention_dim, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, causal_mask: bool = True) -> torch.Tensor:
        T = x.size(1)
        attn_mask = None
        if causal_mask and T > 1:
            attn_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + out)
        x = self.norm2(x + self.ff(x))
        return x


class EMAEmbeddingBuffer:
    def __init__(self, alpha: float = 0.9):
        self.alpha = alpha
        self.running: Optional[torch.Tensor] = None

    def update(self, z: torch.Tensor) -> torch.Tensor:
        z_mean = z.detach().mean(dim=0, keepdim=True)
        if self.running is None:
            self.running = z_mean.clone()
        else:
            self.running = self.alpha * self.running + (1.0 - self.alpha) * z_mean
        return self.running.expand_as(z)

    def reset(self):
        self.running = None
