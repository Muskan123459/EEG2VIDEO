"""Evaluate a trained checkpoint on the SEED-DV test split.

Loads ``group4_best_checkpoint.pt``, generates frames for the first test pair,
computes the full metric suite (SSIM / PSNR / FID / LPIPS / CLIP similarity)
against the ground-truth clip, and writes a JSON report plus the real/fake frames.

    python -m scripts.evaluate
"""

from __future__ import annotations

import json

import numpy as np
import torch

from eeg2video import CFG, DEVICE, Group4Pipeline, load_checkpoint, evaluate_frame_pair_metrics
from eeg2video.data import paired_indices, split_pairs, sample_video_frames
from eeg2video.data.preprocessing import preprocess_eeg_128ch
from eeg2video.utils import list_files, save_video_frames


def main():
    # Build pipeline structure, then load trained weights
    pipeline = Group4Pipeline(CFG)
    ckpt_path = CFG.output_dir / "group4_best_checkpoint.pt"
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    load_checkpoint(pipeline, ckpt_path)
    pipeline.models_eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    eeg_files = list_files(CFG.eeg_root, ".npy")
    video_files = (
        list_files(CFG.video_root, ".mp4")
        + list_files(CFG.video_root, ".avi")
        + list_files(CFG.video_root, ".mov")
    )
    if not eeg_files or not video_files:
        print("No EEG/video files found.")
        return

    pairs = paired_indices(eeg_files, video_files)
    _, _, test_pairs = split_pairs(pairs, CFG.dataset_split_seed, CFG.val_ratio, CFG.test_ratio)
    if not test_pairs:
        test_pairs = pairs[:1]

    eeg_path, video_path, block = test_pairs[0]
    eeg_windows = preprocess_eeg_128ch(eeg_path, block=block)

    fake_frames = pipeline.generate_video_sequence(eeg_windows)
    gen_dir = CFG.output_dir / "sample_generation"
    save_video_frames(fake_frames, gen_dir, prefix="fake")

    real_pil = sample_video_frames(video_path, num_frames=len(fake_frames))
    if len(real_pil) == 0:
        print(f"Could not read frames from: {video_path}")
        return

    real_frames = []
    for im in real_pil:
        arr = np.asarray(im, dtype=np.float32) / 255.0
        real_frames.append(torch.from_numpy(arr).permute(2, 0, 1))

    n = min(len(real_frames), len(fake_frames))
    real_frames = real_frames[:n]
    fake_frames = fake_frames[:n]

    eval_device = DEVICE  # switch to torch.device("cpu") if VRAM is tight
    report = evaluate_frame_pair_metrics(real_frames, fake_frames, eval_device)

    report_dir = CFG.output_dir / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(
            {
                "checkpoint": str(ckpt_path),
                "eeg_path": str(eeg_path),
                "video_path": str(video_path),
                "block": int(block),
                "n_frames": int(n),
                "metrics": report,
            },
            f,
            indent=2,
        )

    save_video_frames(real_frames, report_dir / "real_frames", prefix="real")
    save_video_frames(fake_frames, report_dir / "fake_frames", prefix="fake")

    print("Saved generated frames to:", gen_dir)
    print("Saved evaluation report to:", report_path)
    print("Metrics:", report)


if __name__ == "__main__":
    main()
