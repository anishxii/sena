from __future__ import annotations

import math

import numpy as np
from scipy.signal import welch


FS = 128
EPOCH_SEC = 30
EPOCH_SAMPLES = FS * EPOCH_SEC

CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
IDX_F3 = CHANNELS.index("F3")
IDX_F4 = CHANNELS.index("F4")

BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

EEG_FEATURE_NAMES = [
    "eeg_theta_mean",
    "eeg_alpha_mean",
    "eeg_beta_mean",
    "eeg_gamma_mean",
    "eeg_frontal_alpha_asymmetry",
    "eeg_frontal_alpha_asymmetry_abs",
    "eeg_frontal_theta_alpha_ratio_mean",
    "eeg_load_score",
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _bandpower(signal_1d: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    nperseg = min(len(signal_1d), fs * 4)
    freqs, psd = welch(signal_1d, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 0.0
    return float(np.mean(psd[mask]))


def compute_bandpower_matrix(window: np.ndarray, fs: int = FS) -> tuple[np.ndarray, np.ndarray]:
    if window.ndim != 2 or window.shape[1] != len(CHANNELS):
        raise ValueError(f"expected window shape (n_samples, {len(CHANNELS)}), got {window.shape}")

    abs_bp = np.zeros((len(BANDS), len(CHANNELS)), dtype=np.float64)
    for band_index, (_, (flo, fhi)) in enumerate(BANDS.items()):
        for channel_index in range(len(CHANNELS)):
            abs_bp[band_index, channel_index] = _bandpower(window[:, channel_index], fs, flo, fhi)

    total_power = abs_bp.sum(axis=0, keepdims=True) + 1e-12
    rel_bp = abs_bp / total_power
    return abs_bp, rel_bp


def cognitive_load_score(frontal_theta_alpha_ratio_mean: float) -> float:
    normalized = (float(frontal_theta_alpha_ratio_mean) - 0.5) / 2.5
    score = 1.0 / (1.0 + math.exp(-4.0 * (normalized - 0.5)))
    return _clip01(score)


def compute_eeg_summary(window: np.ndarray, fs: int = FS) -> list[float]:
    abs_bp, rel_bp = compute_bandpower_matrix(window, fs=fs)

    theta_mean = float(rel_bp[0].mean())
    alpha_mean = float(rel_bp[1].mean())
    beta_mean = float(rel_bp[2].mean())
    gamma_mean = float(rel_bp[3].mean())

    frontal_alpha_asymmetry = float(rel_bp[1, IDX_F4] - rel_bp[1, IDX_F3])
    frontal_alpha_asymmetry_abs = float(abs_bp[1, IDX_F4] - abs_bp[1, IDX_F3])

    frontal_channel_indices = [CHANNELS.index(name) for name in ["F7", "F3", "F4", "F8"]]
    theta_alpha_ratio = rel_bp[0, frontal_channel_indices] / (rel_bp[1, frontal_channel_indices] + 1e-12)
    frontal_theta_alpha_ratio_mean = float(theta_alpha_ratio.mean())
    load_score = cognitive_load_score(frontal_theta_alpha_ratio_mean)

    return [
        round(theta_mean, 6),
        round(alpha_mean, 6),
        round(beta_mean, 6),
        round(gamma_mean, 6),
        round(frontal_alpha_asymmetry, 6),
        round(frontal_alpha_asymmetry_abs, 6),
        round(frontal_theta_alpha_ratio_mean, 6),
        round(load_score, 6),
    ]
