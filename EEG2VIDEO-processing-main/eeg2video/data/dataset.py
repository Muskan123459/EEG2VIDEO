"""EEG–video paired dataset, train/val/test splitting, and collation.

Each EEG file contributes one block-aligned pair per stimulus clip; pairs are
shuffled with a fixed seed and partitioned into train/val/test splits.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .preprocessing import preprocess_eeg_128ch
from .video import VideoFeatureExtractor


def paired_indices(eeg_files: List[Path], video_files: List[Path]) -> List[Tuple[Path, Path, int]]:
    """All block×video pairs: each EEG file contributes one pair per video (block i → video i)."""
    if not video_files:
        return []
    pairs = []
    for eeg in eeg_files:
        for block, video in enumerate(video_files):
            pairs.append((eeg, video, block))
    return pairs


def split_pairs(pairs: List[Tuple[Path, Path, int]], seed: int, val_ratio: float, test_ratio: float):
    rnd = random.Random(seed)
    idxs = list(range(len(pairs)))
    rnd.shuffle(idxs)
    n = len(idxs)
    n_test = max(1, int(n * test_ratio)) if n >= 3 else 0
    n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
    test_ids = idxs[:n_test]
    val_ids = idxs[n_test:n_test + n_val]
    train_ids = idxs[n_test + n_val:]
    return [pairs[i] for i in train_ids], [pairs[i] for i in val_ids], [pairs[i] for i in test_ids]


class EEGVideoPairDataset(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path, int]], clip_extractor: VideoFeatureExtractor, max_windows_per_sample: Optional[int] = None):
        self.pairs = pairs
        self.clip_extractor = clip_extractor
        self.max_windows_per_sample = max_windows_per_sample

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        eeg_path, video_path, block = self.pairs[idx]
        eeg_windows = preprocess_eeg_128ch(eeg_path, block=block)
        if self.max_windows_per_sample is not None:
            eeg_windows = eeg_windows[: self.max_windows_per_sample]
        clip_feat = self.clip_extractor.extract_clip_embedding(video_path)
        return {
            "eeg_windows": eeg_windows,
            "clip_feat": clip_feat,
            "eeg_path": str(eeg_path),
            "video_path": str(video_path),
        }


def collate_pairs(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, object]:
    # Variable number of windows per EEG sample is allowed.
    return {
        "batch": batch,
    }
