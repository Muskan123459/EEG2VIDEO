"""Video frame sampling and CLIP visual-feature extraction.

Provides the ground-truth semantic targets used for CLIP alignment: frames are
sampled uniformly from each stimulus clip, encoded with CLIP, and averaged into a
single L2-normalised 512-d embedding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
import cv2

from ..config import CFG

# clip is imported lazily to avoid import-time torchvision issues.
clip = None


def sample_video_frames(video_path: Path, num_frames: int = 5) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    frames: List[Image.Image] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return frames

    idxs = np.linspace(0, max(total - 1, 0), num_frames).astype(int).tolist()
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


class VideoFeatureExtractor:
    def __init__(self, device: torch.device, clip_model_name: str = CFG.clip_model_name):
        self.device = device
        # lazy import of clip to avoid import-time torchvision issues
        import importlib
        global clip
        if clip is None:
            clip = importlib.import_module('clip')

        clip_device = "cpu" if os.environ.get("CLIP_FORCE_CPU", "1") != "0" else device
        self.clip_device = torch.device(clip_device)
        self.model, self.preprocess = clip.load(clip_model_name, device=clip_device)
        self.model.eval()

    @torch.no_grad()
    def extract_clip_embedding(self, video_path: Path) -> torch.Tensor:
        frames = sample_video_frames(video_path, num_frames=5)
        if not frames:
            return torch.zeros(1, 512, device=self.device)
        # AFTER
        batch = torch.stack([self.preprocess(fr) for fr in frames]).to(self.clip_device)
        feat = self.model.encode_image(batch).float().mean(dim=0, keepdim=True)
        feat = feat.to(self.device)   # move result back to main device
        feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feat
