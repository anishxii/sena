from __future__ import annotations

from dataclasses import dataclass

from systems.system2_sdk import (
    Action,
    ActionScores,
    InteractionEffect,
    Outcome,
    PolicyInfo,
    RewardEvent,
    State,
    TurnLog,
)


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

__all__ = [
    "ACTION_BANK",
    "CANONICAL_ACTION_IDS",
    "Action",
    "ActionScores",
    "DecisionTrace",
    "InteractionEffect",
    "Outcome",
    "PolicyInfo",
    "RewardEvent",
    "State",
    "TurnLog",
    "UpdateTrace",
]
