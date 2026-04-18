from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from .ingest_cog_bci import load_eeglab_header
from .schema import BenchmarkConfig, WindowedTrial


def extract_eeg_feature_dict(window: WindowedTrial, config: BenchmarkConfig) -> dict[str, float]:
    """Return interpretable EEG summary features for one window.

    The initial scaffold keeps the interface stable while leaving the actual
    spectral pipeline to be filled in against the real data layout.
    """

    eeg_samples = _resolve_window_samples(window)
    if eeg_samples.size == 0:
        return _empty_eeg_features(config)

    flattened = eeg_samples.reshape(-1)
    if flattened.size == 0:
        return _empty_eeg_features(config)
    payload = window.behavior_payload
    srate = int(payload.get("srate", config.sample_rate_hz))
    set_path = payload.get("eeg_set_path")
    chan_labels = load_eeglab_header(set_path)["chan_labels"] if set_path else []
    frontal_indices = [index for index, label in enumerate(chan_labels) if _is_frontal_label(label)]
    posterior_indices = [index for index, label in enumerate(chan_labels) if _is_posterior_label(label)]

    features = _empty_eeg_features(config)
    fft = np.fft.rfft(eeg_samples, axis=1)
    power = (np.abs(fft) ** 2) / max(eeg_samples.shape[1], 1)
    freqs = np.fft.rfftfreq(eeg_samples.shape[1], d=1.0 / srate)
    broad_mask = (freqs >= 4.0) & (freqs < 45.0)
    broad_power = power[:, broad_mask]
    total_global = float(np.mean(broad_power)) if broad_power.size else 1.0
    total_frontal = float(np.mean(broad_power[frontal_indices, :])) if broad_power.size and frontal_indices else 1.0
    total_posterior = float(np.mean(broad_power[posterior_indices, :])) if broad_power.size and posterior_indices else 1.0
    theta_global = 0.0
    alpha_global = 0.0
    for band_name, (low, high) in config.frequency_bands.items():
        mask = (freqs >= low) & (freqs < high)
        band_power = power[:, mask]
        global_mean = float(np.mean(band_power)) if band_power.size else 0.0
        features[f"{band_name}_global_power"] = global_mean / max(total_global, 1e-8)
        if frontal_indices:
            frontal_mean = float(np.mean(band_power[frontal_indices, :])) if band_power.size else 0.0
            features[f"{band_name}_frontal_power"] = frontal_mean / max(total_frontal, 1e-8)
        else:
            features[f"{band_name}_frontal_power"] = 0.0
        if posterior_indices:
            posterior_mean = float(np.mean(band_power[posterior_indices, :])) if band_power.size else 0.0
            features[f"{band_name}_posterior_power"] = posterior_mean / max(total_posterior, 1e-8)
        else:
            features[f"{band_name}_posterior_power"] = 0.0
        if band_name == "theta":
            theta_global = features[f"{band_name}_global_power"]
        if band_name == "alpha":
            alpha_global = features[f"{band_name}_global_power"]

    features["theta_alpha_ratio"] = theta_global / alpha_global if alpha_global > 1e-8 else 0.0
    features["signal_variance_proxy"] = float(np.log1p(np.var(flattened)))
    return features


def eeg_feature_names(config: BenchmarkConfig) -> list[str]:
    names = []
    for band_name in config.frequency_bands:
        names.extend(
            [
                f"{band_name}_global_power",
                f"{band_name}_frontal_power",
                f"{band_name}_posterior_power",
            ]
        )
    names.extend(["theta_alpha_ratio", "signal_variance_proxy"])
    return names


def _empty_eeg_features(config: BenchmarkConfig) -> dict[str, float]:
    return {name: 0.0 for name in eeg_feature_names(config)}


def _resolve_window_samples(window: WindowedTrial) -> np.ndarray:
    if window.eeg_samples is not None:
        return np.asarray(window.eeg_samples, dtype=np.float32)

    payload = window.behavior_payload
    eeg_fdt_path = payload.get("eeg_fdt_path")
    nbchan = payload.get("nbchan")
    sample_start = payload.get("sample_start")
    sample_end = payload.get("sample_end")
    total_points = payload.get("total_points")
    if None in {eeg_fdt_path, nbchan, sample_start, sample_end, total_points}:
        return np.empty((0, 0), dtype=np.float32)

    raw = np.memmap(
        Path(str(eeg_fdt_path)),
        dtype=np.float32,
        mode="r",
        shape=(int(nbchan), int(total_points)),
        order="F",
    )
    return np.asarray(raw[:, int(sample_start):int(sample_end)], dtype=np.float32)


def _is_frontal_label(label: str) -> bool:
    normalized = label.upper()
    return bool(re.match(r"^(FP|AF|F|FC)", normalized))


def _is_posterior_label(label: str) -> bool:
    normalized = label.upper()
    return bool(re.match(r"^(P|PO|O|OZ)", normalized))
