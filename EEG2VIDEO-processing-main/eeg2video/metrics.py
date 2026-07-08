"""Evaluation metrics: SSIM, PSNR, FID, LPIPS, CLIP similarity.

``MetricsBundle`` lazily constructs the (optional) FID / LPIPS torchmetrics
estimators. ``ClipScorer`` embeds frames with CLIP for semantic similarity.
``evaluate_frame_pair_metrics`` runs the full suite over aligned real/fake frames.
"""

from __future__ import annotations

import math
import os
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import CFG

# Optional metrics packages
try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except Exception:
    FrechetInceptionDistance = None

try:

     from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
except Exception:
    LearnedPerceptualImagePatchSimilarity = None

# clip is imported lazily to avoid import-time torchvision issues.
clip = None


class MetricsBundle:
    def __init__(self, device: torch.device):
        self.device = device
        self._fid = None   # lazily created on first use to avoid downloading inception at startup
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(device) if LearnedPerceptualImagePatchSimilarity is not None else None

    @property
    def fid(self):
        if self._fid is None and FrechetInceptionDistance is not None:
            self._fid = FrechetInceptionDistance(feature=2048, normalize=True).to(self.device)
        return self._fid

    def reset(self):
        if self._fid is not None:
            self._fid.reset()
        if self.lpips is not None:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(self.device)

    @staticmethod
    def ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
        # pred/target: (B, 3, H, W) in [0,1]
        if skimage_ssim is None:
            return float("nan")
        vals = []
        p = pred.detach().cpu().numpy()
        t = target.detach().cpu().numpy()
        for i in range(p.shape[0]):
            pi = np.transpose(p[i], (1, 2, 0))
            ti = np.transpose(t[i], (1, 2, 0))
            vals.append(skimage_ssim(pi, ti, channel_axis=2, data_range=1.0))
        return float(np.mean(vals))

    @staticmethod
    def psnr_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
        mse = F.mse_loss(pred, target).item()
        return float(10.0 * math.log10(1.0 / max(mse, 1e-8)))

    def update_fid(self, real: torch.Tensor, fake: torch.Tensor):
        if self.fid is None:
            return
        real_u8 = (real.clamp(0, 1) * 255).to(torch.uint8)
        fake_u8 = (fake.clamp(0, 1) * 255).to(torch.uint8)
        self.fid.update(real_u8, real=True)
        self.fid.update(fake_u8, real=False)

    def compute_fid(self) -> float:
        if self.fid is None:
            return float("nan")
        return float(self.fid.compute().item())

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        if self.lpips is None:
            return float("nan")
        return float(self.lpips(pred, target).item())


class ClipScorer:
    def __init__(self, device: torch.device, clip_model_name: str = CFG.clip_model_name):
        self.device = device
        import importlib
        global clip
        if clip is None:
            clip = importlib.import_module('clip')
        clip_device = "cpu" if os.environ.get("CLIP_FORCE_CPU", "1") != "0" else device
        self.clip_device = torch.device(clip_device)
        self.model, self.preprocess = clip.load(clip_model_name, device=clip_device)
        self.model.eval()

    @torch.no_grad()
    def embed_frames(self, frames: List[torch.Tensor]) -> torch.Tensor:
        pil_frames = []
        for fr in frames:
            fr = fr.detach().cpu().clamp(0, 1)
            if fr.ndim == 3:
                arr = (fr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                arr = (fr.numpy() * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(arr))
        batch = torch.stack([self.preprocess(im) for im in pil_frames]).to(self.clip_device)  # CPU
        emb = self.model.encode_image(batch).float().mean(dim=0, keepdim=True)
        emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return emb.to(self.device)  # move result back to CUDA

    @torch.no_grad()
    def cosine_similarity(self, frames_a: List[torch.Tensor], frames_b: List[torch.Tensor]) -> float:
        ea = self.embed_frames(frames_a)
        eb = self.embed_frames(frames_b)
        return float(F.cosine_similarity(ea, eb).mean().item())


def evaluate_frame_pair_metrics(real_frames: List[torch.Tensor], fake_frames: List[torch.Tensor], device: torch.device) -> Dict[str, float]:
    """Compute SSIM, PSNR, FID, LPIPS, Top-5, and CLIP similarity.

    real_frames and fake_frames should be aligned lists of RGB tensors in [0,1],
    each tensor shaped (3,H,W).
    """
    metrics = MetricsBundle(device)
    clip_scorer = ClipScorer(device)

    import torch.nn.functional as F

    target_size = real_frames[0].shape[-2:]  # (H, W)

    fake_frames = [
            F.interpolate(fr.unsqueeze(0), size=target_size, mode="bilinear", align_corners=False).squeeze(0)
            for fr in fake_frames
        ]

    if len(real_frames) == 0 or len(fake_frames) == 0:
        return {"ssim": float("nan"), "psnr": float("nan"), "fid": float("nan"), "lpips": float("nan"), "top5": float("nan"), "clip_sim": float("nan")}

    real = torch.stack([fr.to(device) for fr in real_frames], dim=0)
    fake = torch.stack([fr.to(device) for fr in fake_frames], dim=0)

    ssim = metrics.ssim_batch(fake, real)
    psnr = metrics.psnr_batch(fake, real)
    metrics.update_fid(real, fake)
    fid = metrics.compute_fid()
    lpips = metrics.compute_lpips(fake, real)
    clip_sim = clip_scorer.cosine_similarity(fake_frames, real_frames)

    # Top-5 Inception accuracy requires labels and a classifier; placeholder here
    # for the standard evaluation hook. Replace with dataset class labels if available.
    top5 = float("nan")

    return {"ssim": ssim, "psnr": psnr, "fid": fid, "lpips": lpips, "top5": top5, "clip_sim": clip_sim}
