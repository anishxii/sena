from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureVector:
    values: list[float]
    names: list[str]


@dataclass(frozen=True)
class RawObservation:
    timestamp: int
    user_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class State:
    timestamp: int
    user_id: str
    features: FeatureVector
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    action_id: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyInfo:
    policy_type: str
    exploration: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionScores:
    timestamp: int
    user_id: str
    scores: dict[str, float]
    selected_action: str
    policy_info: PolicyInfo


@dataclass(frozen=True)
class InteractionEffect:
    timestamp: int
    user_id: str
    action_id: str
    semantic_effect: dict[str, Any]
    rendering_info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    timestamp: int
    user_id: str
    action_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class InterpretedOutcome:
    signals: dict[str, Any]


@dataclass(frozen=True)
class RewardBreakdown:
    terms: dict[str, float]
    total_reward: float


@dataclass(frozen=True)
class RewardEvent:
    timestamp: int
    user_id: str
    state_features: list[float]
    action_id: str
    reward: float
    outcome: Outcome
    interpreted_outcome: InterpretedOutcome
    reward_breakdown: RewardBreakdown | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnContext:
    state: State
    action_scores: ActionScores
    action: Action
    interaction_effect: InteractionEffect
    outcome: Outcome
    interpreted_outcome: InterpretedOutcome
    reward_event: RewardEvent


@dataclass(frozen=True)
class TurnLog:
    raw_observation: RawObservation
    state: State
    action_scores: ActionScores
    action: Action
    interaction_effect: InteractionEffect
    outcome: Outcome
    interpreted_outcome: InterpretedOutcome
    reward_event: RewardEvent
