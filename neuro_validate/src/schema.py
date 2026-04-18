from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark_name: str
    dataset_root: Path
    task_name: str
    window_length_s: float
    window_step_s: float
    sample_rate_hz: int
    grouped_cv_folds: int
    model_family: str
    output_dir: Path
    frequency_bands: dict[str, tuple[float, float]]
    metrics: tuple[str, ...]


@dataclass(frozen=True)
class WindowedTrial:
    subject_id: str
    session_id: str
    task_name: str
    workload_label: int
    window_start_s: float
    window_end_s: float
    eeg_samples: list[list[float]] | None
    behavior_payload: dict[str, float | int | str | None]


@dataclass(frozen=True)
class BenchmarkSample:
    subject_id: str
    session_id: str
    task_name: str
    workload_label: int
    window_start_s: float
    window_end_s: float
    eeg_features: dict[str, float]
    behavior_features: dict[str, float]


@dataclass(frozen=True)
class ModelEvaluation:
    feature_family: str
    mean_balanced_accuracy: float
    mean_macro_f1: float
    per_fold_balanced_accuracy: tuple[float, ...]
    per_fold_macro_f1: tuple[float, ...]
    sample_count: int
    subject_count: int
