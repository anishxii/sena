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
    condition: str
    window_id: str
    epoch_index: int
    start_offset_s: float
    workload_rating: float | None
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
        self._cache: dict[str, np.ndarray] = {}

    def recording_ids(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for path in sorted(self.stew_dir.glob("sub*_*.txt")):
            stem = path.stem
            subject_id, condition = stem.split("_", 1)
            if condition not in {"hi", "lo"}:
                continue
            pairs.append((subject_id, condition))
        return pairs

    def load_epochs(self, subject_id: str, condition: str) -> np.ndarray:
        cache_key = f"{subject_id}_{condition}"
        if cache_key not in self._cache:
            path = self.stew_dir / f"{subject_id}_{condition}.txt"
            raw = np.loadtxt(path, dtype=np.float32)
            if raw.ndim != 2 or raw.shape[1] != len(CHANNELS):
                raise ValueError(f"{path}: expected shape (n_samples, {len(CHANNELS)}), got {raw.shape}")
            windows = []
            last_start = len(raw) - self.epoch_samples
            if last_start < 0:
                raise ValueError(f"{path}: not enough samples for a {self.epoch_samples}-sample epoch")
            for start in range(0, last_start + 1, self.stride_samples):
                windows.append(raw[start : start + self.epoch_samples])
            self._cache[cache_key] = np.stack(windows, axis=0)
        return self._cache[cache_key]

    def load_epoch(self, subject_id: str, condition: str, epoch_index: int) -> np.ndarray:
        return self.load_epochs(subject_id, condition)[epoch_index]


def build_stew_feature_index(
    stew_dir: str | Path,
    feature_names: list[str],
    epoch_sec: int = 30,
    stride_sec: int | None = None,
) -> STEWFeatureIndex:
    loader = STEWSubjectLoader(stew_dir, epoch_sec=epoch_sec, stride_sec=stride_sec)
    ratings = _load_stew_ratings(stew_dir)

    raw_windows_by_subject: dict[str, list[dict[str, Any]]] = {}
    all_features: list[list[float]] = []

    for subject_id, condition in loader.recording_ids():
        subject_windows = raw_windows_by_subject.setdefault(subject_id, [])
        epochs = loader.load_epochs(subject_id, condition)
        workload_rating = ratings.get(subject_id, {}).get(condition)
        for epoch_index, window in enumerate(epochs):
            features = compute_eeg_summary(window, fs=FS)
            all_features.append(features)
            subject_windows.append(
                {
                    "subject_id": subject_id,
                    "condition": condition,
                    "window_id": f"{subject_id}_{condition}_epoch_{epoch_index:04d}",
                    "epoch_index": epoch_index,
                    "start_offset_s": float(epoch_index * (loader.stride_samples / FS)),
                    "workload_rating": workload_rating,
                    "features": features,
                }
            )

    feature_matrix = np.array(all_features, dtype=np.float64)
    feature_mean = feature_matrix.mean(axis=0)
    feature_std = np.maximum(feature_matrix.std(axis=0), 1e-6)

    windows_by_subject: dict[str, list[IndexedEEGWindow]] = {}
    for subject_id, subject_windows in raw_windows_by_subject.items():
        indexed: list[IndexedEEGWindow] = []
        for window in subject_windows:
            normalized_features = ((np.array(window["features"]) - feature_mean) / feature_std).tolist()
            indexed.append(
                IndexedEEGWindow(
                    subject_id=str(window["subject_id"]),
                    condition=str(window["condition"]),
                    window_id=str(window["window_id"]),
                    epoch_index=int(window["epoch_index"]),
                    start_offset_s=float(window["start_offset_s"]),
                    workload_rating=float(window["workload_rating"]) if window["workload_rating"] is not None else None,
                    features=[float(value) for value in window["features"]],
                    normalized_features=[float(value) for value in normalized_features],
                )
            )
        windows_by_subject[subject_id] = indexed

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


def _load_stew_ratings(stew_dir: str | Path) -> dict[str, dict[str, float]]:
    ratings_path = Path(stew_dir) / "ratings.txt"
    ratings: dict[str, dict[str, float]] = {}
    if not ratings_path.exists():
        return ratings
    for line in ratings_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = [part.strip() for part in stripped.split(",")]
        if len(parts) != 3:
            continue
        subject_number, lo_rating, hi_rating = parts
        subject_id = f"sub{int(subject_number):02d}"
        ratings[subject_id] = {"lo": float(lo_rating), "hi": float(hi_rating)}
    return ratings
