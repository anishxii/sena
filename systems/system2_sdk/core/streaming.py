from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from .types import TurnLog

STREAM_SCHEMA_VERSION = "0.1"


@dataclass(frozen=True)
class StreamEvent:
    schema_version: str
    run_id: str
    session_id: str
    trace_id: str
    turn_index: int
    event_index: int
    event_type: str
    stage: str
    ts_ms: int
    source: str
    payload: dict[str, Any]
    summary: dict[str, Any] = field(default_factory=dict)
    ui_hints: dict[str, Any] = field(default_factory=dict)


def serialize_payload(value: Any) -> Any:
    if is_dataclass(value):
        return {key: serialize_payload(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): serialize_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_payload(item) for item in value]
    return value


def event_to_json_dict(event: StreamEvent) -> dict[str, Any]:
    return serialize_payload(event)


def build_session_started_event(
    *,
    run_id: str,
    session_id: str,
    user_id: str,
    started_at_ms: int,
    metadata: dict[str, Any] | None = None,
) -> StreamEvent:
    return StreamEvent(
        schema_version=STREAM_SCHEMA_VERSION,
        run_id=run_id,
        session_id=session_id,
        trace_id=f"{session_id}:session.started",
        turn_index=-1,
        event_index=0,
        event_type="session.started",
        stage="session",
        ts_ms=started_at_ms,
        source="system2",
        payload={
            "user_id": user_id,
            "metadata": metadata or {},
        },
        summary={
            "headline": "Session started",
            "node_id": "session",
        },
    )


def build_session_ended_event(
    *,
    run_id: str,
    session_id: str,
    user_id: str,
    ended_at_ms: int,
    metadata: dict[str, Any] | None = None,
) -> StreamEvent:
    return StreamEvent(
        schema_version=STREAM_SCHEMA_VERSION,
        run_id=run_id,
        session_id=session_id,
        trace_id=f"{session_id}:session.ended",
        turn_index=-1,
        event_index=0,
        event_type="session.ended",
        stage="session",
        ts_ms=ended_at_ms,
        source="system2",
        payload={
            "user_id": user_id,
            "metadata": metadata or {},
        },
        summary={
            "headline": "Session ended",
            "node_id": "session",
        },
    )


def build_turn_stream_events(
    *,
    run_id: str,
    session_id: str,
    turn_index: int,
    turn_log: TurnLog,
) -> list[StreamEvent]:
    trace_root = f"{session_id}:turn:{turn_index:04d}"
    selected_action = turn_log.action.action_id
    policy_type = turn_log.action_scores.policy_info.policy_type
    reward_total = turn_log.reward_event.reward

    specs = [
        (
            "observation.received",
            "observation",
            "system3",
            turn_log.raw_observation.timestamp,
            serialize_payload(turn_log.raw_observation),
            {
                "node_id": "rawObservation",
                "headline": "Raw observation received",
                "subheadline": turn_log.raw_observation.user_id,
                "primary_metric": turn_log.raw_observation.timestamp,
                "primary_label": "timestamp",
            },
        ),
        (
            "state.updated",
            "state",
            "system2",
            turn_log.state.timestamp,
            serialize_payload(turn_log.state),
            {
                "node_id": "cognitiveState",
                "headline": "State updated",
                "subheadline": f"{len(turn_log.state.features.values)} dimensions",
                "primary_metric": len(turn_log.state.features.values),
                "primary_label": "feature_count",
            },
        ),
        (
            "action.scored",
            "decision",
            "system1",
            turn_log.action_scores.timestamp,
            serialize_payload(turn_log.action_scores),
            {
                "node_id": "policyDecision",
                "headline": "Actions scored",
                "subheadline": policy_type,
                "primary_metric": max(turn_log.action_scores.scores.values(), default=0.0),
                "primary_label": "top_action_score",
                "status": "exploration" if turn_log.action_scores.policy_info.exploration else "greedy",
            },
        ),
        (
            "action.selected",
            "decision",
            "system1",
            turn_log.action_scores.timestamp,
            serialize_payload(turn_log.action),
            {
                "node_id": "policyDecision",
                "headline": f"{selected_action} selected",
                "subheadline": policy_type,
                "primary_metric": turn_log.action_scores.scores.get(selected_action, 0.0),
                "primary_label": "selected_action_score",
            },
        ),
        (
            "interaction.emitted",
            "interaction",
            "system2",
            turn_log.interaction_effect.timestamp,
            serialize_payload(turn_log.interaction_effect),
            {
                "node_id": "applicationEffect",
                "headline": "Interaction effect emitted",
                "subheadline": selected_action,
                "primary_metric": len(turn_log.interaction_effect.semantic_effect),
                "primary_label": "semantic_fields",
            },
        ),
        (
            "outcome.received",
            "outcome",
            "system3",
            turn_log.outcome.timestamp,
            serialize_payload(turn_log.outcome),
            {
                "node_id": "outcome",
                "headline": "Outcome received",
                "subheadline": selected_action,
                "primary_metric": turn_log.outcome.timestamp,
                "primary_label": "timestamp",
            },
        ),
        (
            "outcome.interpreted",
            "interpretation",
            "system2",
            turn_log.outcome.timestamp,
            serialize_payload(turn_log.interpreted_outcome),
            {
                "node_id": "outcome",
                "headline": "Outcome interpreted",
                "subheadline": selected_action,
                "primary_metric": len(turn_log.interpreted_outcome.signals),
                "primary_label": "signal_count",
            },
        ),
        (
            "reward.computed",
            "reward",
            "system2",
            turn_log.reward_event.timestamp,
            serialize_payload(turn_log.reward_event),
            {
                "node_id": "reward",
                "headline": "Reward computed",
                "subheadline": selected_action,
                "primary_metric": reward_total,
                "primary_label": "reward",
            },
        ),
        (
            "turn.committed",
            "turn",
            "system2",
            turn_log.reward_event.timestamp,
            serialize_payload(turn_log),
            {
                "node_id": "turn",
                "headline": "Turn committed",
                "subheadline": selected_action,
                "primary_metric": reward_total,
                "primary_label": "reward",
            },
        ),
    ]

    events: list[StreamEvent] = []
    for event_index, (event_type, stage, source, ts_ms, payload, summary) in enumerate(specs):
        events.append(
            StreamEvent(
                schema_version=STREAM_SCHEMA_VERSION,
                run_id=run_id,
                session_id=session_id,
                trace_id=f"{trace_root}:{event_type}",
                turn_index=turn_index,
                event_index=event_index,
                event_type=event_type,
                stage=stage,
                ts_ms=ts_ms,
                source=source,
                payload=payload,
                summary=summary,
            )
        )
    return events
