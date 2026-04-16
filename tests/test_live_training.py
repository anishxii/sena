from emotiv_learn.live_training import (
    LIVE_FEATURE_NAMES,
    STATE_PROFILE_BEHAVIOR_ONLY,
    STATE_PROFILE_CURRENT_EEG,
    STATE_PROFILE_TUTOR_PROXY_EEG,
    LiveLLMStateBuilder,
    LiveStateInput,
)


def test_live_state_builder_uses_interpreted_signals_without_eeg() -> None:
    state = LiveLLMStateBuilder().build_state(
        LiveStateInput(
            timestamp=1,
            user_id="live_user",
            topic_id="gradient_descent",
            task_type="learn",
            difficulty="medium",
            turn_index=3,
            max_turns=10,
            interpreted={
                "followup_type": "clarify",
                "confusion_score": 0.7,
                "comprehension_score": 0.4,
                "engagement_score": 0.6,
                "progress_signal": 0.2,
                "pace_fast_score": 0.0,
                "pace_slow_score": 0.3,
            },
            student_response={"self_reported_confidence": 0.35},
            previous_reward=-0.23,
        )
    )

    assert state.feature_names == LIVE_FEATURE_NAMES
    assert len(state.features) == len(LIVE_FEATURE_NAMES)
    assert state.features[LIVE_FEATURE_NAMES.index("last_response_clarify")] == 1.0
    assert state.features[LIVE_FEATURE_NAMES.index("student_confidence")] == 0.35
    assert state.features[LIVE_FEATURE_NAMES.index("difficulty_medium")] == 1.0


def test_live_state_builder_defaults_are_valid_for_first_turn() -> None:
    state = LiveLLMStateBuilder().build_state(
        LiveStateInput(
            timestamp=1,
            user_id="new_user",
            topic_id="gradient_descent",
            task_type="learn",
            difficulty="hard",
            turn_index=1,
            max_turns=5,
            interpreted=None,
            student_response=None,
            previous_reward=0.0,
        )
    )

    assert len(state.features) == len(LIVE_FEATURE_NAMES)
    assert state.features[LIVE_FEATURE_NAMES.index("last_response_other")] == 1.0
    assert state.features[LIVE_FEATURE_NAMES.index("student_confidence")] == 0.5
    assert state.features[LIVE_FEATURE_NAMES.index("difficulty_hard")] == 1.0


def test_live_state_builder_behavior_only_zeroes_eeg_and_tutor_proxy_features() -> None:
    state = LiveLLMStateBuilder(state_profile=STATE_PROFILE_BEHAVIOR_ONLY).build_state(
        LiveStateInput(
            timestamp=1,
            user_id="behavior_only",
            topic_id="gradient_descent",
            task_type="learn",
            difficulty="medium",
            turn_index=2,
            max_turns=5,
            interpreted={
                "followup_type": "continue",
                "confusion_score": 0.2,
                "comprehension_score": 0.8,
                "engagement_score": 0.7,
                "progress_signal": 0.6,
                "pace_fast_score": 0.4,
                "pace_slow_score": 0.0,
            },
            student_response={"self_reported_confidence": 0.8},
            previous_reward=0.4,
            eeg_features=[0.9] * 8,
            eeg_proxy_estimates={
                "workload_estimate": 0.3,
                "rolling_accuracy": 0.8,
                "rolling_rt_percentile": 0.2,
                "lapse_rate": 0.1,
            },
        )
    )

    assert len(state.features) == len(LIVE_FEATURE_NAMES)
    assert state.features[LIVE_FEATURE_NAMES.index("eeg_theta_mean")] == 0.0
    assert state.features[LIVE_FEATURE_NAMES.index("proxy_workload_estimate")] == 0.0
    assert state.features[LIVE_FEATURE_NAMES.index("tutor_overload_risk")] == 0.0


def test_live_state_builder_tutor_proxy_profile_adds_control_relevant_proxy_features() -> None:
    state = LiveLLMStateBuilder(state_profile=STATE_PROFILE_TUTOR_PROXY_EEG).build_state(
        LiveStateInput(
            timestamp=1,
            user_id="proxy_user",
            topic_id="gradient_descent",
            task_type="learn",
            difficulty="medium",
            turn_index=4,
            max_turns=10,
            interpreted={
                "followup_type": "branch",
                "confusion_score": 0.25,
                "comprehension_score": 0.72,
                "engagement_score": 0.81,
                "progress_signal": 0.66,
                "pace_fast_score": 0.45,
                "pace_slow_score": 0.05,
                "checkpoint_score": 1.0,
            },
            student_response={"self_reported_confidence": 0.77},
            previous_reward=0.55,
            eeg_features=[0.2] * 8,
            eeg_proxy_estimates={
                "workload_estimate": 0.28,
                "rolling_accuracy": 0.88,
                "rolling_rt_percentile": 0.18,
                "lapse_rate": 0.07,
            },
        )
    )

    assert state.features[LIVE_FEATURE_NAMES.index("proxy_workload_estimate")] == 0.28
    assert state.features[LIVE_FEATURE_NAMES.index("tutor_challenge_readiness")] > 0.5
    assert state.features[LIVE_FEATURE_NAMES.index("tutor_curiosity_headroom")] > 0.5


def test_live_state_builder_current_eeg_omits_tutor_proxy_layer() -> None:
    state = LiveLLMStateBuilder(state_profile=STATE_PROFILE_CURRENT_EEG).build_state(
        LiveStateInput(
            timestamp=1,
            user_id="current_eeg",
            topic_id="gradient_descent",
            task_type="learn",
            difficulty="medium",
            turn_index=2,
            max_turns=6,
            interpreted={"followup_type": "clarify"},
            student_response={"self_reported_confidence": 0.4},
            previous_reward=0.0,
            eeg_features=[0.4] * 8,
            eeg_proxy_estimates={
                "workload_estimate": 0.6,
                "rolling_accuracy": 0.55,
                "rolling_rt_percentile": 0.7,
                "lapse_rate": 0.2,
            },
        )
    )

    assert state.features[LIVE_FEATURE_NAMES.index("proxy_workload_estimate")] == 0.6
    assert state.features[LIVE_FEATURE_NAMES.index("tutor_repair_need")] == 0.0
