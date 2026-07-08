"""Generate ablation frame sets for the temporal components.

Loads a trained checkpoint (if present, otherwise a freshly initialised pipeline)
and generates frame sequences for the first test pair under three settings:
full model, without causal temporal attention, and without EMA momentum smoothing.

    python -m scripts.ablation
"""

from __future__ import annotations

from eeg2video import CFG, Group4Pipeline, load_checkpoint, run_ablation
from eeg2video.data import paired_indices, split_pairs
from eeg2video.data.preprocessing import preprocess_eeg_128ch
from eeg2video.utils import list_files, save_video_frames


def main():
    pipeline = Group4Pipeline(CFG)
    ckpt_path = CFG.output_dir / "group4_best_checkpoint.pt"
    if ckpt_path.exists():
        load_checkpoint(pipeline, ckpt_path)
        print(f"Loaded checkpoint: {ckpt_path}")
    pipeline.models_eval()

    eeg_files = list_files(CFG.eeg_root, ".npy")
    video_files = list_files(CFG.video_root, ".mp4")
    if not eeg_files or not video_files:
        print("No EEG/video files found.")
        return

    pairs = paired_indices(eeg_files, video_files)
    _, _, test_pairs = split_pairs(pairs, CFG.dataset_split_seed, CFG.val_ratio, CFG.test_ratio)
    if not test_pairs:
        test_pairs = pairs[:1]

    eeg_path, video_path, block = test_pairs[0]
    eeg_windows = preprocess_eeg_128ch(eeg_path, block=block)

    ab_full = run_ablation(pipeline, eeg_windows, with_temporal_attention=True, with_momentum=True)
    ab_no_temp = run_ablation(pipeline, eeg_windows, with_temporal_attention=False, with_momentum=True)
    ab_no_mom = run_ablation(pipeline, eeg_windows, with_temporal_attention=True, with_momentum=False)

    save_video_frames(ab_full, CFG.output_dir / "ablation_full", prefix="full")
    save_video_frames(ab_no_temp, CFG.output_dir / "ablation_no_temporal", prefix="notemp")
    save_video_frames(ab_no_mom, CFG.output_dir / "ablation_no_momentum", prefix="nomom")

    print("Ablation frames saved.")
    print("For true SSIM/PSNR/FID/FVD/LPIPS/Top-5 reporting, pass real test frames into evaluate_frame_pair_metrics().")


if __name__ == "__main__":
    main()
