from __future__ import annotations

import numpy as np

from emotiv_learn.eeg import EEGObservationContext, build_eeg_provider
from emotiv_learn.eeg_mapper import (
    fit_stew_workload_feature_mapper,
    load_stew_workload_feature_mapper,
    save_stew_workload_feature_mapper,
)
from emotiv_learn.stew_index import build_stew_feature_index


def _make_tiny_stew_dataset(stew_dir) -> None:
    stew_dir.mkdir()
    (stew_dir / "ratings.txt").write_text("1, 2, 8\n", encoding="utf-8")
    hi = np.tile(np.linspace(0.4, 1.2, 3840, dtype=np.float32).reshape(-1, 1), (1, 14))
    lo = np.tile(np.linspace(0.0, 0.6, 3840, dtype=np.float32).reshape(-1, 1), (1, 14))
    np.savetxt(stew_dir / "sub01_hi.txt", hi)
    np.savetxt(stew_dir / "sub01_lo.txt", lo)


def test_fit_and_reload_workload_mapper(tmp_path) -> None:
    stew_dir = tmp_path / "stew"
    _make_tiny_stew_dataset(stew_dir)
    index = build_stew_feature_index(stew_dir=stew_dir, feature_names=[
        "eeg_theta_mean",
        "eeg_alpha_mean",
        "eeg_beta_mean",
        "eeg_gamma_mean",
        "eeg_frontal_alpha_asymmetry",
        "eeg_frontal_alpha_asymmetry_abs",
        "eeg_frontal_theta_alpha_ratio_mean",
        "eeg_load_score",
    ], epoch_sec=10, stride_sec=10)

    mapper = fit_stew_workload_feature_mapper(index)
    path = tmp_path / "mapper.json"
    save_stew_workload_feature_mapper(mapper, path)
    loaded = load_stew_workload_feature_mapper(path)

    assert loaded.feature_names == mapper.feature_names
    assert len(loaded.coefficients) == len(mapper.coefficients)
    assert len(loaded.coefficients[0]) == len(mapper.coefficients[0])


def test_build_provider_uses_learned_mapper(tmp_path) -> None:
    stew_dir = tmp_path / "stew"
    _make_tiny_stew_dataset(stew_dir)
    index = build_stew_feature_index(stew_dir=stew_dir, feature_names=[
        "eeg_theta_mean",
        "eeg_alpha_mean",
        "eeg_beta_mean",
        "eeg_gamma_mean",
        "eeg_frontal_alpha_asymmetry",
        "eeg_frontal_alpha_asymmetry_abs",
        "eeg_frontal_theta_alpha_ratio_mean",
        "eeg_load_score",
    ], epoch_sec=10, stride_sec=10)
    mapper = fit_stew_workload_feature_mapper(index)
    mapper_path = tmp_path / "mapper.json"
    save_stew_workload_feature_mapper(mapper, mapper_path)

    provider = build_eeg_provider(
        eeg_mode="synthetic",
        seed=7,
        mapper_path=str(mapper_path),
    )
    eeg_window = provider.observe(
        EEGObservationContext(
            timestamp=1,
            user_id="u1",
            concept_id="gradient",
            action_id="worked_example",
            tutor_message="Use a concrete worked example and then summarize the update rule.",
            time_on_chunk=55.0,
            hidden_state={
                "knowledge_state": {"concept_mastery": {"gradient": 0.25}, "confidence": 0.4},
                "neuro_state": {"workload": 0.41, "fatigue": 0.15, "attention": 0.7, "vigilance": 0.66, "stress": 0.22},
            },
            observable_signals={"confusion_score": 0.65, "engagement_score": 0.75},
        )
    )

    assert eeg_window.metadata["source"] == "synthetic"
    assert "proxy_state" in eeg_window.metadata
    assert "target_features" in eeg_window.metadata
    assert len(eeg_window.features) == 8
