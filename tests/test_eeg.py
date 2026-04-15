from __future__ import annotations

import numpy as np

from emotiv_learn.eeg import EEGObservationContext, EEG_SUMMARY_FEATURE_NAMES, SyntheticEEGProvider, estimate_time_on_chunk
from emotiv_learn.eeg_features import CHANNELS, EPOCH_SAMPLES, compute_eeg_summary
from emotiv_learn.eeg_retrieval import NearestNeighborPatientEEGRetriever
from emotiv_learn.stew_index import IndexedEEGWindow, STEWFeatureIndex


def test_time_on_chunk_increases_with_length_and_complexity() -> None:
    short_simple = "Gradient points uphill."
    long_dense = (
        "Gradient descent iteratively updates parameters by moving opposite the derivative "
        "while balancing learning-rate magnitude, local curvature, and convergence behavior "
        "across a potentially nonconvex optimization surface."
    )

    assert estimate_time_on_chunk(long_dense) > estimate_time_on_chunk(short_simple)


def test_compute_eeg_summary_emits_expected_schema() -> None:
    window = np.tile(np.linspace(0.0, 1.0, EPOCH_SAMPLES, dtype=np.float32).reshape(-1, 1), (1, len(CHANNELS)))
    features = compute_eeg_summary(window)

    assert len(features) == len(EEG_SUMMARY_FEATURE_NAMES)
    assert features[-1] >= 0.0


def test_synthetic_eeg_provider_emits_expected_schema() -> None:
    provider = SyntheticEEGProvider(seed=5)

    eeg_window = provider.observe(
        EEGObservationContext(
            timestamp=4,
            user_id="u1",
            concept_id="gradient",
            action_id="step_by_step",
            tutor_message="Step 1: identify the gradient. Step 2: follow the update carefully.",
            time_on_chunk=82.0,
            hidden_state={
                "concept_mastery": {"gradient": 0.35},
                "fatigue": 0.25,
                "attention": 0.62,
                "confidence": 0.48,
                "engagement": 0.66,
            },
            observable_signals={
                "confusion_score": 0.68,
                "engagement_score": 0.66,
                "confidence": 0.48,
                "attention": 0.62,
            },
        )
    )

    assert eeg_window.feature_names == EEG_SUMMARY_FEATURE_NAMES
    assert len(eeg_window.features) == len(EEG_SUMMARY_FEATURE_NAMES)
    assert eeg_window.metadata["source"] == "synthetic"


def test_retriever_returns_closest_patient_window() -> None:
    index = STEWFeatureIndex(
        feature_names=EEG_SUMMARY_FEATURE_NAMES,
        feature_mean=[0.0] * len(EEG_SUMMARY_FEATURE_NAMES),
        feature_std=[1.0] * len(EEG_SUMMARY_FEATURE_NAMES),
        windows_by_subject={
            "sub01": [
                IndexedEEGWindow("sub01", "w0", 0, 0.0, [0.1] * 8, [0.1] * 8),
                IndexedEEGWindow("sub01", "w1", 1, 30.0, [0.9] * 8, [0.9] * 8),
            ]
        },
    )
    retriever = NearestNeighborPatientEEGRetriever(feature_index=index, seed=1)
    matches = retriever.retrieve("sub01", [0.12] * 8, k=1)

    assert len(matches) == 1
    assert matches[0].window.window_id == "w0"
