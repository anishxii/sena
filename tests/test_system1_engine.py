from systems.system1_decision.engine import DecisionEngine
from systems.system1_decision.schemas import ACTION_BANK
from systems.system2_sdk import FeatureVector, InterpretedOutcome, Outcome, RewardEvent, RewardBreakdown, State


def build_state(user_id: str = "learner_a") -> State:
    return State(
        timestamp=1,
        user_id=user_id,
        features=FeatureVector(values=[0.25, 0.50, 0.75], names=["a", "b", "c"]),
        metadata={"application": "tutor"},
    )


def build_reward_event(state: State, action_id: str, reward: float) -> RewardEvent:
    outcome = Outcome(
        timestamp=2,
        user_id=state.user_id,
        action_id=action_id,
        payload={"response_type": "continue"},
    )
    return RewardEvent(
        timestamp=3,
        user_id=state.user_id,
        state_features=state.features.values,
        action_id=action_id,
        reward=reward,
        outcome=outcome,
        interpreted_outcome=InterpretedOutcome(signals={"comprehension_score": 0.8}),
        reward_breakdown=RewardBreakdown(terms={"progress": reward}, total_reward=reward),
    )


def test_engine_scores_selects_and_updates() -> None:
    engine = DecisionEngine(feature_dim=3, epsilon=0.0, seed=7, use_personalization=True)
    state = build_state()

    action_scores = engine.score_actions(state, ACTION_BANK)

    assert set(action_scores.scores) == {action.action_id for action in ACTION_BANK}
    action = engine.select_action(action_scores)
    reward_event = build_reward_event(state, action.action_id, reward=1.0)

    engine.update(reward_event)

    assert len(engine.update_history) == 1
    trace = engine.update_history[-1]
    assert trace.user_id == "learner_a"
    assert trace.action_id == action.action_id
    assert trace.reward == 1.0


def test_engine_tracks_user_specific_residuals() -> None:
    engine = DecisionEngine(feature_dim=3, epsilon=0.0, seed=3, use_personalization=True)
    state_a = build_state("learner_a")
    state_b = build_state("learner_b")

    action_a = engine.select_action(engine.score_actions(state_a, ACTION_BANK))
    engine.update(build_reward_event(state_a, action_a.action_id, reward=1.0))

    action_b = engine.select_action(engine.score_actions(state_b, ACTION_BANK))
    engine.update(build_reward_event(state_b, action_b.action_id, reward=-0.5))

    assert "learner_a" in engine.user_residuals
    assert "learner_b" in engine.user_residuals
