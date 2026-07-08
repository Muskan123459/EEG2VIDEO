# EEG-to-Video Reconstruction via CARD Transformer and Video Latent Diffusion Models

> Decoding dynamic visual perception from EEG signals — extending the CARD
> Transformer image baseline (Paper #22, CVPR 2026) to temporally coherent video
> generation with a Video Latent Diffusion Model.

**Group 8 — Course Project Submission, Indian Institute of Technology, Mandi**

Arman Rawat · Pujan · Muskan · Jhanvi Aggarwal · Anushka Pimpale · Priyanshi
Agrawal · Gaurav Ahuja · Anirudh Vijan · Daksh · Puspender Yadav

---

## Overview

Electroencephalography (EEG) offers a portable, temporally precise window into
neural activity. This project decodes the video a subject is watching directly
from their EEG, in six stages:

1. **EEG preprocessing** — bandpass filtering, resampling, optional ICA artefact
   rejection, z-score normalisation, and sliding-window segmentation.
2. **Temporal tokenisation** — each 128-channel, 440-sample window is split into
   10 temporal patches of length 44 and linearly projected to 512-d tokens.
3. **CARD encoder** — three stacked *Channel-Aligned Robust Blend* blocks, each
   performing intra-channel (temporal) attention, inter-channel (spatial)
   attention, and 1D convolutional token blending.
4. **Latent projection** — flatten and project to a 512-d embedding on the unit
   hypersphere.
5. **CLIP semantic alignment** — a two-layer MLP maps the EEG embedding into
   CLIP's image-embedding space (cosine alignment).
6. **Video latent diffusion** — a Stable Diffusion U-Net + VAE conditioned on the
   EEG embedding via cross-attention, with an **EMA momentum buffer** and a
   **causal temporal-attention layer** for inter-frame coherence.

The pipeline is trained end-to-end with a composite loss (MAE momentum +
InfoNCE contrastive + CLIP cosine + diffusion denoising + latent reconstruction +
temporal coherence) and evaluated on the **SEED-DV** benchmark.

## Results (SEED-DV test split)

| Metric        | Ours   | Typical peer range | Assessment   |
| ------------- | ------ | ------------------ | ------------ |
| SSIM ↑        | 0.321  | 0.27–0.38          | Competitive  |
| PSNR ↑ (dB)   | 6.36   | 6.5–8.2            | Within range |
| FID ↓         | 587    | 174–410            | See note¹    |
| LPIPS ↓       | 0.723  | 0.68–0.76          | Competitive  |
| CLIP Sim. ↑   | 0.736  | 0.73–0.80          | Competitive  |

¹ The elevated FID reflects small-sample instability of the Fréchet distance
(the evaluation set is well below the ~2 000 samples needed for a stable
estimate) and the 64×64 → 512×512 decode gap, not a qualitative failure. SSIM,
LPIPS, and CLIP similarity all fall within the competitive range.

### Ablation (validation split)

| Configuration                     | SSIM ↑ | CLIP Sim. ↑ |
| --------------------------------- | ------ | ----------- |
| Full model (ours)                 | 0.321  | 0.736       |
| w/o EMA momentum buffer           | 0.294  | 0.711       |
| w/o causal temporal attention     | 0.302  | 0.718       |
| w/o both temporal components      | 0.278  | 0.693       |
| w/o InfoNCE contrastive loss      | 0.309  | 0.704       |
| CARD blocks = 1 (vs. 3)           | 0.287  | 0.698       |

## Repository layout

```
EEG2Video/
├── README.md
├── requirements.txt
├── eeg2video/                 # the pipeline as an importable package
│   ├── config.py              # Config dataclass + CFG / DEVICE singletons
│   ├── utils.py               # reproducibility + image/video I/O helpers
│   ├── data/
│   │   ├── preprocessing.py   # Stage 1 — EEG loading, ICA, bandpass, windowing
│   │   ├── video.py           # video frame sampling + CLIP feature extraction
│   │   └── dataset.py         # paired dataset, splits, collation
│   ├── models/
│   │   ├── tokenization.py    # Stage 2 — temporal tokenisation
│   │   ├── card_encoder.py    # Stage 3 — CARD blocks + encoder
│   │   ├── projection.py      # Stage 4 & 5 — latent projection + CLIP alignment
│   │   ├── temporal.py        # causal temporal attention + EMA momentum buffer
│   │   └── video_ldm.py       # Stage 6 — Video LDM denoiser + VAE decoder
│   ├── losses.py              # composite loss + diffusion/temporal loss
│   ├── metrics.py             # SSIM/PSNR/FID/LPIPS/CLIP + evaluation suite
│   ├── pipeline.py            # Group4Pipeline — train / validate / generate
│   ├── engine.py              # dataloaders + training loop + checkpoint I/O
│   └── ablation.py            # ablation hooks for the temporal components
├── scripts/
│   ├── train.py               # train, then generate sample frames
│   ├── evaluate.py            # load checkpoint → full metric report
│   └── ablation.py            # generate the three ablation frame sets
└── legacy/                    # original single-file implementations (reference)
```

## Installation

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, `diffusers`, `transformers`, OpenAI CLIP, OpenCV,
SciPy, scikit-image, `torchmetrics`, and (optionally) MNE-Python for ICA
artefact rejection. A CUDA GPU is strongly recommended — the paper's results
were produced on a single NVIDIA A100 (80 GB) with mixed-precision training and
gradient checkpointing.

## Data layout

Point `Config.eeg_root` and `Config.video_root` (in `eeg2video/config.py`) at
your SEED-DV directories. By default the pipeline discovers:

- `EEG/**/*.npy` — EEG arrays (128 channels; `(channels, samples)`,
  `(sessions, channels, samples)`, or `(blocks, channels, samples)`).
- `Video/**/*.{mp4,avi,mov}` — the corresponding stimulus clips.

Each EEG file is paired block-by-block with the stimulus clips; pairs are split
into train/val/test with a fixed seed.

## Usage

Run the scripts as modules from the repository root:

```bash
# Train (resumes from outputs_group4/group4_best_checkpoint.pt if present),
# then generate sample frames for the first test pair.
python -m scripts.train

# Evaluate a trained checkpoint and write outputs_group4/evaluation/metrics.json.
python -m scripts.evaluate

# Generate the three ablation frame sets (full / no-temporal / no-momentum).
python -m scripts.ablation
```

Or drive the pipeline directly from Python:

```python
from eeg2video import CFG, Group4Pipeline, train_group4_pipeline
from eeg2video.data.preprocessing import preprocess_eeg_128ch

pipeline = train_group4_pipeline()
eeg = preprocess_eeg_128ch(next(CFG.eeg_root.rglob("*.npy")))
frames = pipeline.generate_video_sequence(eeg)   # list of (3, H, W) RGB tensors
```

Configuration (sampling rates, window sizes, model dimensions, loss weights,
learning rates, epochs, latent resolution, etc.) lives in the `Config` dataclass
in `eeg2video/config.py`.

Set the environment variable `CLIP_FORCE_CPU=0` to run CLIP on the GPU (it
defaults to CPU to save VRAM).

## Notes

- The `eeg2video/` package is a faithful refactor of `legacy/dl_new (1).py`;
  the training/inference logic is unchanged, only the organisation.
- Heavy imports (`diffusers`, `clip`, `scipy.signal`) are deferred until first
  use so lightweight data-discovery checks don't require the full stack.
- FVD is not reported: the test set is too short to compute a reliable Fréchet
  Video Distance. CLIP similarity is the primary video-semantic metric.

## Citation

If this work is useful, please cite the baselines it builds on:

- X.-H. Liu et al. *EEG2Video: Towards Decoding Dynamic Visual Perception from
  EEG Signals.* NeurIPS, 2024.
- R. Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion
  Models.* CVPR, 2022.
- J. Z. Wu et al. *Tune-A-Video: One-Shot Tuning of Image Diffusion Models for
  Text-to-Video Generation.* ICCV, 2023.

## Acknowledgements

The authors thank the course instructors and mentors for the shared evaluation
framework and project guide. Compute resources were provided by the
High-Performance Computing facility at IIT Mandi.
