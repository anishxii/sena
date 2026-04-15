from emotiv_learn.live_training import LIVE_FEATURE_NAMES, LiveLLMStateBuilder, LiveStateInput


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
    assert state.features[LIVE_FEATURE_NAMES.index("difficulty_medium")] == 1.0
    assert state.features[LIVE_FEATURE_NAMES.index("student_confidence")] == 0.35


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
    assert state.features[LIVE_FEATURE_NAMES.index("difficulty_hard")] == 1.0
