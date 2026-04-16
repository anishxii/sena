from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .eeg_features import CHANNELS, EPOCH_SAMPLES, FS, compute_eeg_summary


@dataclass(frozen=True)
class IndexedEEGWindow:
    subject_id: str
    window_id: str
    epoch_index: int
    start_offset_s: float
    features: list[float]
    normalized_features: list[float]


@dataclass(frozen=True)
class STEWFeatureIndex:
    feature_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    windows_by_subject: dict[str, list[IndexedEEGWindow]]
    epoch_sec: int | None = None
    stride_sec: int | None = None


class STEWSubjectLoader:
    def __init__(self, stew_dir: str | Path, epoch_sec: int = 30, stride_sec: int | None = None) -> None:
        self.stew_dir = Path(stew_dir)
        self.fs = FS
        self.epoch_sec = epoch_sec
        self.epoch_samples = self.fs * self.epoch_sec
        self.stride_sec = stride_sec if stride_sec is not None else epoch_sec
        self.stride_samples = self.fs * self.stride_sec
        self._subject_cache: dict[str, np.ndarray] = {}

    def subject_ids(self) -> list[str]:
        return sorted(path.name.replace("_hi.txt", "") for path in self.stew_dir.glob("sub*_hi.txt"))

    def load_epochs(self, subject_id: str) -> np.ndarray:
        if subject_id not in self._subject_cache:
            path = self.stew_dir / f"{subject_id}_hi.txt"
            delimiter = _detect_delimiter(path)
            raw = np.loadtxt(path, delimiter=delimiter, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != len(CHANNELS):
                raise ValueError(f"{path}: expected shape (n_samples, {len(CHANNELS)}), got {raw.shape}")
            windows = []
            last_start = len(raw) - self.epoch_samples
            if last_start < 0:
                raise ValueError(f"{path}: not enough samples for a {self.epoch_samples}-sample epoch")
            for start in range(0, last_start + 1, self.stride_samples):
                windows.append(raw[start : start + self.epoch_samples])
            self._subject_cache[subject_id] = np.stack(windows, axis=0)
        return self._subject_cache[subject_id]

    def load_epoch(self, subject_id: str, epoch_index: int) -> np.ndarray:
        epochs = self.load_epochs(subject_id)
        return epochs[epoch_index]


def build_stew_feature_index(
    stew_dir: str | Path,
    feature_names: list[str],
    epoch_sec: int = 30,
    stride_sec: int | None = None,
) -> STEWFeatureIndex:
    loader = STEWSubjectLoader(stew_dir, epoch_sec=epoch_sec, stride_sec=stride_sec)
    raw_windows_by_subject: dict[str, list[dict[str, Any]]] = {}
    all_features: list[list[float]] = []

    for subject_id in loader.subject_ids():
        subject_windows: list[dict[str, Any]] = []
        epochs = loader.load_epochs(subject_id)
        for epoch_index, window in enumerate(epochs):
            features = compute_eeg_summary(window, fs=FS)
            all_features.append(features)
            subject_windows.append(
                {
                    "subject_id": subject_id,
                    "window_id": f"{subject_id}_epoch_{epoch_index:04d}",
                    "epoch_index": epoch_index,
                    "start_offset_s": float(epoch_index * (loader.stride_samples / FS)),
                    "features": features,
                }
            )
        raw_windows_by_subject[subject_id] = subject_windows

    feature_matrix = np.array(all_features, dtype=np.float64)
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = np.maximum(feature_matrix.std(axis=0), 1e-6)

    windows_by_subject: dict[str, list[IndexedEEGWindow]] = {}
    for subject_id, subject_windows in raw_windows_by_subject.items():
        indexed_windows: list[IndexedEEGWindow] = []
        for window in subject_windows:
            normalized_features = ((np.array(window["features"]) - feature_mean) / feature_std).tolist()
            indexed_windows.append(
                IndexedEEGWindow(
                    subject_id=subject_id,
                    window_id=str(window["window_id"]),
                    epoch_index=int(window["epoch_index"]),
                    start_offset_s=float(window["start_offset_s"]),
                    features=[float(value) for value in window["features"]],
                    normalized_features=[float(value) for value in normalized_features],
                )
            )
        windows_by_subject[subject_id] = indexed_windows

    return STEWFeatureIndex(
        feature_names=list(feature_names),
        feature_mean=[float(value) for value in feature_mean.tolist()],
        feature_std=[float(value) for value in feature_std.tolist()],
        windows_by_subject=windows_by_subject,
        epoch_sec=epoch_sec,
        stride_sec=loader.stride_sec,
    )


def save_feature_index(index: STEWFeatureIndex, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_names": index.feature_names,
        "feature_mean": index.feature_mean,
        "feature_std": index.feature_std,
        "epoch_sec": index.epoch_sec,
        "stride_sec": index.stride_sec,
        "windows_by_subject": {
            subject_id: [asdict(window) for window in windows]
            for subject_id, windows in index.windows_by_subject.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_feature_index(index_path: str | Path) -> STEWFeatureIndex:
    payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
    return STEWFeatureIndex(
        feature_names=list(payload["feature_names"]),
        feature_mean=[float(value) for value in payload["feature_mean"]],
        feature_std=[float(value) for value in payload["feature_std"]],
        windows_by_subject={
            subject_id: [IndexedEEGWindow(**window) for window in windows]
            for subject_id, windows in payload["windows_by_subject"].items()
        },
        epoch_sec=payload.get("epoch_sec"),
        stride_sec=payload.get("stride_sec"),
    )


def _detect_delimiter(path: Path) -> str | None:
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    if "\t" in first_line:
        return "\t"
    if "," in first_line:
        return ","
    return None
