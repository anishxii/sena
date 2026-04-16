from emotiv_learn.eeg import EEGObservationContext, EEG_SUMMARY_FEATURE_NAMES, HeuristicTargetEEGMapper, SyntheticEEGProvider, build_eeg_provider, estimate_time_on_chunk
def test_time_on_chunk_increases_with_length_and_complexity() -> None:
    short_simple = "Gradient points uphill."
    long_dense = (
        "Gradient descent iteratively updates parameters by moving opposite the derivative "
        "while balancing learning-rate magnitude, local curvature, and convergence behavior "
        "across a potentially nonconvex optimization surface."
    )

    assert estimate_time_on_chunk(long_dense) > estimate_time_on_chunk(short_simple)


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
                "knowledge_state": {
                    "concept_mastery": {"gradient": 0.35},
                    "confidence": 0.48,
                },
                "neuro_state": {
                    "workload": 0.42,
                    "fatigue": 0.25,
                    "attention": 0.62,
                    "engagement": 0.66,
                    "vigilance": 0.58,
                    "stress": 0.30,
                },
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
    assert 0.0 <= eeg_window.features[0] <= 1.0
    assert 0.0 <= eeg_window.features[1] <= 1.0
    assert eeg_window.features[-1] >= 0.0


def test_heuristic_target_mapper_emits_proxy_and_feature_targets() -> None:
    mapper = HeuristicTargetEEGMapper()
    context = mapper.predict_proxy_state(
        type("Ctx", (), {
            "user_id": "u1",
            "concept_id": "gradient",
            "action_id": "worked_example",
            "tutor_message": "Work through the gradient step by step with a concrete example.",
            "time_on_chunk": 70.0,
            "hidden_state": {
                "knowledge_state": {"concept_mastery": {"gradient": 0.3}, "confidence": 0.45},
                "neuro_state": {"workload": 0.48, "fatigue": 0.2, "attention": 0.65, "vigilance": 0.62, "stress": 0.28},
            },
            "observable_signals": {"confusion_score": 0.6, "engagement_score": 0.7},
        })()
    )
    features = mapper.predict_features(
        type("Ctx", (), {
            "user_id": "u1",
            "concept_id": "gradient",
            "action_id": "worked_example",
            "tutor_message": "Work through the gradient step by step with a concrete example.",
            "time_on_chunk": 70.0,
            "hidden_state": {
                "knowledge_state": {"concept_mastery": {"gradient": 0.3}, "confidence": 0.45},
                "neuro_state": {"workload": 0.48, "fatigue": 0.2, "attention": 0.65, "vigilance": 0.62, "stress": 0.28},
            },
            "observable_signals": {"confusion_score": 0.6, "engagement_score": 0.7},
        })()
    )

    assert 0.0 <= context.workload <= 1.0
    assert len(features) == len(EEG_SUMMARY_FEATURE_NAMES)
def test_build_eeg_provider_supports_synthetic_mode_only() -> None:
    provider = build_eeg_provider(eeg_mode="synthetic", seed=3)
    assert isinstance(provider, SyntheticEEGProvider)
