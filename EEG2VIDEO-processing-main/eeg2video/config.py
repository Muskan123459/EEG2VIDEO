"""Global configuration for the CARD Transformer + Video LDM pipeline.

Defines the :class:`Config` dataclass together with the module-level singletons
(``CFG``, ``DEVICE``) that the rest of the package reads from. Many model
sub-modules use ``CFG.*`` values as default constructor arguments, so this module
is imported early and instantiated exactly once.
"""

from __future__ import annotations

import os

# Reduce CUDA fragmentation. Must be set before torch initialises the allocator.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from dataclasses import dataclass
from pathlib import Path

import torch

from .utils import seed_everything


@dataclass
class Config:
    eeg_root: Path = Path("EEG")
    video_root: Path = Path("Video")
    output_dir: Path = Path("outputs_group4")

    # Guide-spec preprocessing
    original_sfreq: float = 200.0
    target_sfreq: float = 880.0
    window_samples: int = 440
    overlap: float = 0.5
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    n_channels: int = 128
    n_patches: int = 10
    patch_len: int = 44

    # Model sizes
    d_model: int = 512
    unet_cross_attention_dim: int = 768
    num_card_blocks: int = 3
    nhead_card: int = 8

    # Training
    batch_size: int = 1
    epochs: int = 500
    lr_encoder: float = 1e-4
    lr_unet: float = 1e-5
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    mixed_precision: bool = True
    # Gradient accumulation to simulate larger batch sizes when GPU memory is limited
    grad_accum_steps: int = 4
    num_train_timesteps: int = 1000
    num_inference_steps: int = 100


    # Loss weights from guide
    lambda_mae: float = 0.1
    lambda_contrastive: float = 0.1
    lambda_clip: float = 1.0
    lambda_diffusion: float = 1.0
    lambda_temporal: float = 0.1
    lambda_recon: float = 0.5

    # Video generation
    latent_h: int = 64          # smaller latents = less VRAM
    latent_w: int = 64
    max_video_frames: int = 4   # fewer frames per sample during training
    context_len: int = 4
    grad_accum_steps: int = 8   # compensate with more accumulation steps

    # LR scheduler
    lr_min_factor: float = 0.01      # ← new: cosine annealing floor = lr * this factor
    warmup_epochs: int = 10          # ← new: linear warmup before cosine decay kicks in

    # Misc
    dataset_split_seed: int = 123
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model_name: str = "ViT-B/32"
    stable_diffusion_model: str = "runwayml/stable-diffusion-v1-5"


CFG = Config()
CFG.output_dir.mkdir(parents=True, exist_ok=True)
seed_everything(42)
DEVICE = torch.device(CFG.device)
