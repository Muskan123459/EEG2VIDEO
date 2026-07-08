"""Reproducibility helpers and lightweight tensor / image / video I/O utilities.

Kept free of any dependency on :mod:`eeg2video.config` so that it can be imported
first (``config`` itself depends on :func:`seed_everything`).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence

import numpy as np
import cv2
import torch
from PIL import Image

# torchvision utilities are imported lazily to avoid version-mismatch errors at
# module import time.
make_grid = None
save_image = None


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------

def list_files(root: Path, ext: str) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob(f"*{ext}"))


def safe_mean(xs: Sequence[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def to_uint8_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().clamp(0, 1)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    # lazy import to avoid torchvision version mismatches at import time
    global make_grid
    if make_grid is None:
        from torchvision.utils import make_grid as _make_grid
        make_grid = _make_grid
    grid = make_grid(x, nrow=1)
    grid = (grid * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    return grid


def save_video_frames(frames: List[torch.Tensor], out_dir: Path, prefix: str = "frame") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, fr in enumerate(frames):
        img = fr.detach().cpu().clamp(0, 1)
        # lazy import to avoid torchvision import issues at module import time
        global save_image
        if save_image is None:
            from torchvision.utils import save_image as _save_image
            save_image = _save_image
        save_image(img, out_dir / f"{prefix}_{i:04d}.png")


def load_frame_for_vae(video_path: str, frame_idx: int, total_frames: int, device: torch.device) -> torch.Tensor:
    """Load one video frame, resize to 512x512, normalise to [-1, 1] for VAE input."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return torch.zeros(1, 3, 512, 512, device=device)
    pick = int(frame_idx * max(total - 1, 0) / max(total_frames - 1, 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, pick)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return torch.zeros(1, 3, 512, 512, device=device)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb).resize((512, 512))
    arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
