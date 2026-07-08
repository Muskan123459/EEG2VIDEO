"""Composite training objective and the diffusion + temporal-coherence loss.

``CompositeLoss`` combines the MAE momentum term, InfoNCE contrastive term, and
CLIP cosine-alignment term (diffusion is passed in). ``VideoLDMLoss`` returns the
DDPM noise-prediction MSE plus a temporal-coherence penalty on adjacent frames.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CFG


class CompositeLoss(nn.Module):
    """Guide-style total loss: MAE + contrastive + CLIP + diffusion."""
    def __init__(self,
                 lambda_mae: float = CFG.lambda_mae,
                 lambda_contrastive: float = CFG.lambda_contrastive,
                 lambda_clip: float = CFG.lambda_clip,
                 lambda_diffusion: float = CFG.lambda_diffusion):
        super().__init__()
        self.lambda_mae = lambda_mae
        self.lambda_contrastive = lambda_contrastive
        self.lambda_clip = lambda_clip
        self.lambda_diffusion = lambda_diffusion
        self.l1 = nn.L1Loss()

    def forward(self,
                z: torch.Tensor,
                z_img: torch.Tensor,
                mae_input: Optional[torch.Tensor] = None,
                mae_target: Optional[torch.Tensor] = None,
                contrastive_loss: Optional[torch.Tensor] = None,
                diffusion_loss: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cosine = F.cosine_similarity(z, z_img, dim=1)
        l_clip = (1.0 - cosine).mean()

        if mae_input is not None and mae_target is not None:
            l_mae = self.l1(mae_input, mae_target)
        else:
            l_mae = torch.tensor(0.0, device=z.device)

        if contrastive_loss is None:
            contrastive_loss = torch.tensor(0.0, device=z.device)
        if diffusion_loss is None:
            diffusion_loss = torch.tensor(0.0, device=z.device)

        total = (
            self.lambda_mae * l_mae +
            self.lambda_contrastive * contrastive_loss +
            self.lambda_clip * l_clip +
            self.lambda_diffusion * diffusion_loss
        )
        return total, {"l_mae": l_mae, "l_contrastive": contrastive_loss, "l_clip": l_clip, "l_diffusion": diffusion_loss}


class VideoLDMLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, noise_pred: torch.Tensor, noise_target: torch.Tensor, adj_frames: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        l_diff = self.mse(noise_pred, noise_target)
        l_temp = torch.tensor(0.0, device=noise_pred.device)
        if adj_frames is not None and adj_frames.size(1) > 1:
            l_temp = self.mse(adj_frames[:, :-1], adj_frames[:, 1:])
        total = l_diff + CFG.lambda_temporal * l_temp
        return total, {"l_diffusion": l_diff, "l_temporal": l_temp}
