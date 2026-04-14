import math

import pytest

from emotiv_learn.decision_engine import DecisionEngine
from emotiv_learn.schemas import ACTION_BANK, Outcome, RewardEvent, SemanticSignals, State, StateMetadata, TaskResult


def build_state(user_id: str = "user_a") -> State:
    feature_names = [
        "eeg_theta",
        "eeg_alpha",
        "eeg_beta",
        "eeg_beta_alpha_ratio",
        "eeg_load_proxy",
        "turn_index_norm",
        "difficulty_easy",
        "difficulty_medium",
        "difficulty_hard",
        "prev_correct",
        "prev_latency_norm",
        "prev_reread",
        "recent_clarify_count_norm",
        "recent_progress_norm",
        "checkpoint_mode_flag",
    ]
    features = [
        0.42,
        0.55,
        0.61,
        1.11,
        0.67,
        0.30,
        0.00,
        1.00,
        0.00,
        1.00,
        0.25,
        0.00,
        0.20,
        0.75,
        0.00,
    ]
    return State(
        timestamp=1,
        user_id=user_id,
        features=features,
        feature_names=feature_names,
        metadata=StateMetadata(task_type="learn", difficulty="medium", topic_id="algebra"),
    )


def build_reward_event(state: State, action_id: str, reward: float) -> RewardEvent:
    return RewardEvent(
        timestamp=state.timestamp + 1,
        user_id=state.user_id,
        state_features=state.features,
        action_id=action_id,
        reward=reward,
        outcome=Outcome(
            timestamp=state.timestamp + 1,
            user_id=state.user_id,
            action_id=action_id,
            task_result=TaskResult(
                correct=1,
                latency_s=4.0,
                reread=0,
                completed=1,
                abandoned=0,
            ),
            semantic_signals=SemanticSignals(
                followup_text="continue",
                followup_type="continue",
                confusion_score=0.0,
                comprehension_score=0.8,
                engagement_score=0.7,
                pace_fast_score=0.1,
                pace_slow_score=0.2,
            ),
            raw={},
        ),
    )


def test_score_actions_returns_all_actions() -> None:
    state = build_state()
    engine = DecisionEngine(feature_dim=len(state.features), epsilon=0.0, seed=7)

    action_scores = engine.score_actions(state, ACTION_BANK)

    assert list(action_scores.scores.keys()) == [action.action_id for action in ACTION_BANK]
    assert action_scores.selected_action == "no_change"
    assert action_scores.policy_info.policy_type == "personalized"
    assert action_scores.policy_info.exploration is False


def test_update_changes_generic_and_personalized_scores() -> None:
    state = build_state()
    engine = DecisionEngine(feature_dim=len(state.features), epsilon=0.0, seed=7)

    before = engine.score_actions(state, ACTION_BANK)
    reward_event = build_reward_event(state, action_id="deepen", reward=1.0)
    engine.update(reward_event)
    after = engine.score_actions(state, ACTION_BANK)

    assert before.scores["deepen"] == 0.0
    assert after.scores["deepen"] > before.scores["deepen"]
    assert after.scores["deepen"] > after.scores["no_change"]


def test_generic_mode_updates_without_user_residuals() -> None:
    state = build_state()
    engine = DecisionEngine(
        feature_dim=len(state.features),
        epsilon=0.0,
        use_personalization=False,
        seed=7,
    )

    reward_event = build_reward_event(state, action_id="worked_example", reward=0.8)
    engine.update(reward_event)
    after = engine.score_actions(state, ACTION_BANK)

    assert after.policy_info.policy_type == "generic"
    assert after.scores["worked_example"] > 0.0


def test_rejects_state_with_non_finite_features() -> None:
    state = build_state()
    bad_features = list(state.features)
    bad_features[0] = math.nan
    bad_state = State(
        timestamp=state.timestamp,
        user_id=state.user_id,
        features=bad_features,
        feature_names=state.feature_names,
        metadata=state.metadata,
    )
    engine = DecisionEngine(feature_dim=len(state.features), epsilon=0.0, seed=7)

    with pytest.raises(ValueError, match="state.features"):
        engine.score_actions(bad_state, ACTION_BANK)


def test_rejects_reward_event_with_unknown_action_id() -> None:
    state = build_state()
    engine = DecisionEngine(feature_dim=len(state.features), epsilon=0.0, seed=7)
    reward_event = build_reward_event(state, action_id="deepen", reward=1.0)
    bad_event = RewardEvent(
        timestamp=reward_event.timestamp,
        user_id=reward_event.user_id,
        state_features=reward_event.state_features,
        action_id="not_real",
        reward=reward_event.reward,
        outcome=reward_event.outcome,
    )

    with pytest.raises(ValueError, match="unknown action_id"):
        engine.update(bad_event)


def test_rejects_reward_event_with_non_finite_reward() -> None:
    state = build_state()
    engine = DecisionEngine(feature_dim=len(state.features), epsilon=0.0, seed=7)
    reward_event = build_reward_event(state, action_id="deepen", reward=math.inf)

    with pytest.raises(ValueError, match="must be finite"):
        engine.update(reward_event)


def test_persists_and_reloads_weights_with_sqlite(tmp_path) -> None:
    state = build_state(user_id="persistent_user")
    db_path = tmp_path / "decision_engine.sqlite"

    first_engine = DecisionEngine(
        feature_dim=len(state.features),
        epsilon=0.0,
        seed=7,
        db_path=str(db_path),
    )
    reward_event = build_reward_event(state, action_id="worked_example", reward=1.0)
    first_engine.update(reward_event)
    first_scores = first_engine.score_actions(state, ACTION_BANK)

    second_engine = DecisionEngine(
        feature_dim=len(state.features),
        epsilon=0.0,
        seed=7,
        db_path=str(db_path),
    )
    second_scores = second_engine.score_actions(state, ACTION_BANK)

    assert second_scores.scores["worked_example"] == pytest.approx(
        first_scores.scores["worked_example"]
    )
    assert second_scores.scores["worked_example"] > 0.0


def test_update_history_captures_decision_time_trace() -> None:
    state = build_state(user_id="trace_user")
    engine = DecisionEngine(feature_dim=len(state.features), epsilon=0.0, seed=7)

    action_scores = engine.score_actions(state, ACTION_BANK)
    reward_event = build_reward_event(state, action_id=action_scores.selected_action, reward=0.9)
    engine.update(reward_event)

    update_trace = engine.update_history[-1]
    assert update_trace.user_id == "trace_user"
    assert update_trace.action_id == action_scores.selected_action
    assert update_trace.policy_type == action_scores.policy_info.policy_type
    assert update_trace.exploration is False


def test_reward_clipping_applies_before_update_history() -> None:
    state = build_state()
    engine = DecisionEngine(
        feature_dim=len(state.features),
        epsilon=0.0,
        seed=7,
        reward_clip_abs=1.0,
    )

    reward_event = build_reward_event(state, action_id="deepen", reward=10.0)
    engine.update(reward_event)

    assert engine.update_history[-1].reward == 1.0


def test_weight_decay_shrinks_existing_weight_before_update() -> None:
    state = build_state()
    engine = DecisionEngine(
        feature_dim=len(state.features),
        epsilon=0.0,
        seed=7,
        use_personalization=False,
        l2_weight_decay=0.10,
    )

    first_event = build_reward_event(state, action_id="worked_example", reward=1.0)
    second_event = build_reward_event(state, action_id="worked_example", reward=0.0)

    engine.update(first_event)
    first_score = engine.score_actions(state, ACTION_BANK).scores["worked_example"]
    engine.update(second_event)
    second_score = engine.score_actions(state, ACTION_BANK).scores["worked_example"]

    assert second_score < first_score
