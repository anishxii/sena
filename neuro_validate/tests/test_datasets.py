from pathlib import Path

from neuro_validate.src.config import load_benchmark_config
from neuro_validate.src.datasets import build_benchmark_samples, build_feature_matrices
from neuro_validate.src.schema import WindowedTrial


def test_build_feature_matrices_includes_three_feature_families() -> None:
    config = load_benchmark_config(
        Path("/Users/anish/PERSONAL/emotiv_learn/neuro_validate/configs/cog_bci_nback_workload.yaml")
    )
    windows = [
        WindowedTrial(
            subject_id="01",
            session_id="ses-01",
            task_name="N-Back",
            workload_label=2,
            window_start_s=0.0,
            window_end_s=4.0,
            eeg_samples=[[0.1, 0.2, 0.3], [0.0, 0.1, 0.2]],
            behavior_payload={
                "task_condition_norm": 1.0,
                "trial_progress_norm": 0.25,
                "rolling_accuracy": 0.8,
                "rolling_rt_percentile": 0.3,
                "lapse_rate": 0.1,
                "session_index_norm": 0.0,
            },
        )
    ]

    samples = build_benchmark_samples(windows, config)
    matrices = build_feature_matrices(samples, config)

    assert set(matrices) == {"behavior_only", "eeg_only", "fused"}
    assert matrices["behavior_only"]["y"] == [2]
    assert matrices["behavior_only"]["groups"] == ["01"]
    assert len(matrices["fused"]["X"][0]) == (
        len(matrices["behavior_only"]["feature_names"]) + len(matrices["eeg_only"]["feature_names"])
    )
