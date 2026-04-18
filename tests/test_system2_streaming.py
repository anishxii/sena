from systems.system2_sdk import (
    Action,
    ActionScores,
    FeatureVector,
    InteractionEffect,
    InterpretedOutcome,
    Outcome,
    PolicyInfo,
    RawObservation,
    RewardBreakdown,
    RewardEvent,
    State,
    TurnLog,
    build_session_started_event,
    build_turn_stream_events,
    event_to_json_dict,
)


def build_turn_log() -> TurnLog:
    raw_observation = RawObservation(
        timestamp=1000,
        user_id="learner_a",
        payload={
            "eeg": {"theta_alpha_ratio": 1.82, "workload_estimate": 0.74},
            "behavior": {"followup_type": "continue", "progress_signal": 0.62},
            "context": {"topic_id": "gradient_descent", "difficulty": "medium"},
        },
    )
    state = State(
        timestamp=1001,
        user_id="learner_a",
        features=FeatureVector(
            values=[0.74, 0.58, 0.42],
            names=["overload_risk", "future_lapse_risk", "attention_stability"],
        ),
        metadata={"policy_mode": "personalized"},
    )
    action_scores = ActionScores(
        timestamp=1002,
        user_id="learner_a",
        scores={"worked_example": 0.77, "highlight_key_points": 0.49},
        selected_action="worked_example",
        policy_info=PolicyInfo(policy_type="personalized", exploration=False),
    )
    action = Action(action_id="worked_example")
    interaction_effect = InteractionEffect(
        timestamp=1003,
        user_id="learner_a",
        action_id="worked_example",
        semantic_effect={"style": "example_first"},
        rendering_info={"style_label": "worked_example"},
    )
    outcome = Outcome(
        timestamp=1004,
        user_id="learner_a",
        action_id="worked_example",
        payload={"response_type": "continue", "checkpoint_correct": 1},
    )
    interpreted = InterpretedOutcome(signals={"comprehension_score": 0.73, "confusion_score": 0.16})
    reward_event = RewardEvent(
        timestamp=1005,
        user_id="learner_a",
        state_features=state.features.values,
        action_id="worked_example",
        reward=0.42,
        outcome=outcome,
        interpreted_outcome=interpreted,
        reward_breakdown=RewardBreakdown(terms={"correctness": 0.3, "progress": 0.12}, total_reward=0.42),
        metadata={"policy_type": "personalized"},
    )
    return TurnLog(
        raw_observation=raw_observation,
        state=state,
        action_scores=action_scores,
        action=action,
        interaction_effect=interaction_effect,
        outcome=outcome,
        interpreted_outcome=interpreted,
        reward_event=reward_event,
    )


def test_build_turn_stream_events_emits_canonical_sequence() -> None:
    events = build_turn_stream_events(
        run_id="run_001",
        session_id="session_001",
        turn_index=7,
        turn_log=build_turn_log(),
    )

    assert [event.event_type for event in events] == [
        "observation.received",
        "state.updated",
        "action.scored",
        "action.selected",
        "interaction.emitted",
        "outcome.received",
        "outcome.interpreted",
        "reward.computed",
        "turn.committed",
    ]
    assert events[0].summary["node_id"] == "rawObservation"
    assert events[2].summary["primary_label"] == "top_action_score"
    assert events[-1].payload["reward_event"]["reward"] == 0.42


def test_stream_events_serialize_to_json_ready_dicts() -> None:
    event = build_session_started_event(
        run_id="run_001",
        session_id="session_001",
        user_id="learner_a",
        started_at_ms=123,
        metadata={"application": "tutor"},
    )

    payload = event_to_json_dict(event)
    assert payload["event_type"] == "session.started"
    assert payload["payload"]["metadata"]["application"] == "tutor"
