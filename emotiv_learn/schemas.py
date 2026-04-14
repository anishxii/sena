from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Action:
    action_id: str
    params: dict[str, Any]


@dataclass(frozen=True)
class StateMetadata:
    task_type: str
    difficulty: str
    topic_id: str


@dataclass(frozen=True)
class State:
    timestamp: int
    user_id: str
    features: list[float]
    feature_names: list[str]
    metadata: StateMetadata


@dataclass(frozen=True)
class PolicyInfo:
    policy_type: str
    exploration: bool


@dataclass(frozen=True)
class ActionScores:
    timestamp: int
    user_id: str
    scores: dict[str, float]
    selected_action: str
    policy_info: PolicyInfo


@dataclass(frozen=True)
class TaskResult:
    correct: int | None
    latency_s: float | None
    reread: int | None
    completed: int | None
    abandoned: int | None


@dataclass(frozen=True)
class SemanticSignals:
    followup_text: str | None
    followup_type: str | None
    confusion_score: float | None
    comprehension_score: float | None
    engagement_score: float | None
    pace_fast_score: float | None
    pace_slow_score: float | None


@dataclass(frozen=True)
class Outcome:
    timestamp: int
    user_id: str
    action_id: str
    task_result: TaskResult
    semantic_signals: SemanticSignals
    raw: dict[str, Any]


@dataclass(frozen=True)
class RewardEvent:
    timestamp: int
    user_id: str
    state_features: list[float]
    action_id: str
    reward: float
    outcome: Outcome


@dataclass(frozen=True)
class DecisionTrace:
    state_timestamp: int
    user_id: str
    selected_action: str
    policy_type: str
    exploration: bool
    scores: dict[str, float]


@dataclass(frozen=True)
class UpdateTrace:
    user_id: str
    action_id: str
    reward: float
    predicted_score: float
    error: float
    policy_type: str | None
    exploration: bool | None
    generic_score: float
    user_score: float


CANONICAL_ACTION_IDS = [
    "no_change",
    "simplify",
    "deepen",
    "summarize",
    "highlight_key_points",
    "worked_example",
    "analogy",
    "step_by_step",
]

ACTION_BANK = [Action(action_id=action_id, params={}) for action_id in CANONICAL_ACTION_IDS]
