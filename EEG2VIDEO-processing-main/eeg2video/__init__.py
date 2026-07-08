"""EEG-to-Video reconstruction via the CARD Transformer + Video Latent Diffusion.

A six-stage pipeline that decodes dynamic visual perception from 128-channel EEG:

    Stage 1  EEG preprocessing        (:mod:`eeg2video.data.preprocessing`)
    Stage 2  Temporal tokenisation    (:mod:`eeg2video.models.tokenization`)
    Stage 3  CARD encoder             (:mod:`eeg2video.models.card_encoder`)
    Stage 4  Latent projection        (:mod:`eeg2video.models.projection`)
    Stage 5  CLIP semantic alignment  (:mod:`eeg2video.models.projection`)
    Stage 6  Video latent diffusion   (:mod:`eeg2video.models.video_ldm`)

The :class:`~eeg2video.pipeline.Group4Pipeline` ties the stages together for
training, validation, and autoregressive video generation.
"""

from __future__ import annotations

from .config import Config, CFG, DEVICE
from .pipeline import Group4Pipeline
from .engine import build_dataloaders, train_group4_pipeline, load_checkpoint
from .ablation import run_ablation
from .metrics import evaluate_frame_pair_metrics

__all__ = [
    "Config",
    "CFG",
    "DEVICE",
    "Group4Pipeline",
    "build_dataloaders",
    "train_group4_pipeline",
    "load_checkpoint",
    "run_ablation",
    "evaluate_frame_pair_metrics",
]
