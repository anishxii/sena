from __future__ import annotations

from .behavior_features import BEHAVIOR_FEATURE_NAMES, extract_behavior_feature_dict
from .eeg_features import eeg_feature_names, extract_eeg_feature_dict
from .schema import BenchmarkConfig, BenchmarkSample, WindowedTrial


def build_benchmark_samples(windows: list[WindowedTrial], config: BenchmarkConfig) -> list[BenchmarkSample]:
    samples = []
    for window in windows:
        samples.append(
            BenchmarkSample(
                subject_id=window.subject_id,
                session_id=window.session_id,
                task_name=window.task_name,
                workload_label=window.workload_label,
                window_start_s=window.window_start_s,
                window_end_s=window.window_end_s,
                eeg_features=extract_eeg_feature_dict(window, config),
                behavior_features=extract_behavior_feature_dict(window),
            )
        )
    return samples


def build_feature_matrices(
    samples: list[BenchmarkSample],
    config: BenchmarkConfig,
) -> dict[str, dict[str, list]]:
    eeg_names = eeg_feature_names(config)
    matrices = {
        "behavior_only": {"X": [], "y": [], "groups": [], "feature_names": list(BEHAVIOR_FEATURE_NAMES)},
        "eeg_only": {"X": [], "y": [], "groups": [], "feature_names": list(eeg_names)},
        "fused": {"X": [], "y": [], "groups": [], "feature_names": list(BEHAVIOR_FEATURE_NAMES) + list(eeg_names)},
    }

    for sample in samples:
        behavior_vector = [sample.behavior_features[name] for name in BEHAVIOR_FEATURE_NAMES]
        eeg_vector = [sample.eeg_features[name] for name in eeg_names]

        matrices["behavior_only"]["X"].append(behavior_vector)
        matrices["eeg_only"]["X"].append(eeg_vector)
        matrices["fused"]["X"].append(behavior_vector + eeg_vector)

        for family in matrices.values():
            family["y"].append(sample.workload_label)
            family["groups"].append(sample.subject_id)

    return matrices

