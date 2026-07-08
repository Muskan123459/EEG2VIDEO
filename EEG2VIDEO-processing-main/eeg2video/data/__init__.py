"""Data loading, EEG preprocessing, and video/CLIP feature extraction."""

from .preprocessing import (
    load_eeg_numpy,
    apply_ica_artifact_rejection,
    preprocess_eeg_128ch,
)
from .video import sample_video_frames, VideoFeatureExtractor
from .dataset import (
    paired_indices,
    split_pairs,
    EEGVideoPairDataset,
    collate_pairs,
)

__all__ = [
    "load_eeg_numpy",
    "apply_ica_artifact_rejection",
    "preprocess_eeg_128ch",
    "sample_video_frames",
    "VideoFeatureExtractor",
    "paired_indices",
    "split_pairs",
    "EEGVideoPairDataset",
    "collate_pairs",
]
