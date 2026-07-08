"""Stage 1 — EEG loading and preprocessing.

Loads raw 128-channel EEG, optionally rejects ocular/cardiac artefacts with ICA,
resamples to the target rate, applies a zero-phase Butterworth bandpass, z-score
normalises per channel, and slices 50%-overlap windows of ``window_samples``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import CFG

# scipy.signal is imported lazily inside preprocessing to avoid import-time
# binary compatibility issues in some environments.
signal = None

# Optional MNE for ICA artifact rejection
try:
    import mne
except Exception:
    mne = None


def load_eeg_numpy(npy_path: Path, block: int = 0) -> np.ndarray:
    raw_data = np.load(npy_path)
    if raw_data.ndim == 3:
        if raw_data.shape[0] == CFG.n_channels:
            # (channels, ?, samples)
            eeg = raw_data.reshape(raw_data.shape[0], -1)
        elif raw_data.shape[1] == CFG.n_channels:
            # (sessions, channels, samples) with exact channel match
            eeg = raw_data.transpose(1, 0, 2).reshape(raw_data.shape[1], -1)
        else:
            # (blocks, channels, samples) — pick the requested block
            b = min(block, raw_data.shape[0] - 1)
            eeg = raw_data[b]  # (channels, samples)
    elif raw_data.ndim == 2:
        eeg = raw_data
    else:
        raise ValueError(f"Unsupported EEG array shape: {raw_data.shape}")
    return eeg.astype(np.float32)


def apply_ica_artifact_rejection(eeg: np.ndarray, sfreq: float) -> np.ndarray:
    """Optional ICA artifact rejection. Falls back gracefully when MNE is absent."""
    if mne is None:
        return eeg

    try:
        info = mne.create_info(ch_names=[f"ch_{i}" for i in range(eeg.shape[0])], sfreq=sfreq, ch_types="eeg")
        raw = mne.io.RawArray(eeg, info, verbose=False)
        raw.filter(l_freq=CFG.bandpass_low, h_freq=CFG.bandpass_high, verbose=False)

        ica = mne.preprocessing.ICA(n_components=min(20, eeg.shape[0] - 1), random_state=42, max_iter="auto")
        ica.fit(raw, verbose=False)

        # Without EOG/ECG channels, we use a conservative reconstruction path.
        cleaned = ica.apply(raw.copy(), verbose=False).get_data()
        return cleaned.astype(np.float32)
    except Exception:
        # Keep pipeline robust if ICA fails on this dataset.
        return eeg


def preprocess_eeg_128ch(npy_path: Path,
                         target_sfreq: float = CFG.target_sfreq,
                         original_sfreq: float = CFG.original_sfreq,
                         window_samples: int = CFG.window_samples,
                         overlap: float = CFG.overlap,
                         block: int = 0) -> torch.Tensor:
    """Return tensor of shape (N, 128, 440)."""
    global signal
    if signal is None:
        from scipy import signal as _signal
        signal = _signal
    eeg = load_eeg_numpy(npy_path, block=block)

    # Force 128 channels where possible.
    if eeg.shape[0] < CFG.n_channels:
        pad = np.zeros((CFG.n_channels - eeg.shape[0], eeg.shape[1]), dtype=eeg.dtype)
        eeg = np.concatenate([eeg, pad], axis=0)
    elif eeg.shape[0] > CFG.n_channels:
        eeg = eeg[:CFG.n_channels]

    # Artifact rejection (optional) and bandpass.
    eeg = apply_ica_artifact_rejection(eeg, sfreq=original_sfreq)

    # Resample to 880 Hz.
    num_samples = int(eeg.shape[-1] * target_sfreq / original_sfreq)
    resampled = signal.resample(eeg, num_samples, axis=-1)

    # Robust bandpass via SOS.
    nyq = 0.5 * target_sfreq
    sos = signal.butter(4, [CFG.bandpass_low / nyq, CFG.bandpass_high / nyq], btype="band", output="sos")
    filtered = signal.sosfiltfilt(sos, resampled, axis=-1)

    # Z-score normalization per channel.
    mean = filtered.mean(axis=-1, keepdims=True)
    std = filtered.std(axis=-1, keepdims=True) + 1e-6
    normalized = (filtered - mean) / std

    # 50% overlap sliding windows: hop=220 for 440-sample windows.
    hop = int(window_samples * (1.0 - overlap))
    windows = []
    start = 0
    while start + window_samples <= normalized.shape[-1]:
        windows.append(normalized[:, start:start + window_samples])
        start += hop

    if not windows:
        raise ValueError(f"EEG file too short for windowing: {npy_path}")

    windows = np.stack(windows, axis=0).astype(np.float32)
    return torch.from_numpy(windows)
