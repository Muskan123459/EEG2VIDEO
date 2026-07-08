"""Train the CARD + Video LDM pipeline, then generate sample frames.

Trains on all discovered EEG/video pairs (see :class:`eeg2video.Config` for the
``EEG`` / ``Video`` roots), then generates and saves a short frame sequence for
the first test pair as a smoke test of the trained model.

    python -m scripts.train
"""

from __future__ import annotations

from eeg2video import CFG, DEVICE, train_group4_pipeline
from eeg2video.data import paired_indices, split_pairs
from eeg2video.data.preprocessing import preprocess_eeg_128ch
from eeg2video.utils import list_files, save_video_frames


def main():
    pipeline = train_group4_pipeline()

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

    _, _, test_pairs = split_pairs(
        pairs,
        CFG.dataset_split_seed,
        CFG.val_ratio,
        CFG.test_ratio
    )

    if not test_pairs:
        test_pairs = pairs[:1]

    eeg_path, video_path, block = test_pairs[0]

    eeg_windows = preprocess_eeg_128ch(eeg_path, block=block)

    generated_frames = pipeline.generate_video_sequence(eeg_windows)

    out_dir = CFG.output_dir / "sample_generation"

    save_video_frames(generated_frames, out_dir)

    print(f"Saved generated frames to: {out_dir}")


if __name__ == "__main__":
    main()
