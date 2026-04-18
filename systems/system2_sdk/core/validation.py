from __future__ import annotations

from .types import ActionScores, FeatureVector, RewardEvent, State


def validate_feature_vector(feature_vector: FeatureVector) -> None:
    if len(feature_vector.values) != len(feature_vector.names):
        raise ValueError("feature vector values and names must align exactly")
    if not feature_vector.values:
        raise ValueError("feature vector must not be empty")


def validate_state(state: State) -> None:
    validate_feature_vector(state.features)


def validate_action_scores(action_scores: ActionScores, action_ids: list[str] | None = None) -> None:
    if action_ids is not None and sorted(action_scores.scores) != sorted(action_ids):
        raise ValueError("action scores must include exactly the application action ids")
    if action_scores.selected_action not in action_scores.scores:
        raise ValueError("selected_action must exist in action_scores.scores")


def validate_reward_event(reward_event: RewardEvent) -> None:
    if reward_event.action_id != reward_event.outcome.action_id:
        raise ValueError("reward event action_id must match outcome.action_id")
    if reward_event.user_id != reward_event.outcome.user_id:
        raise ValueError("reward event user_id must match outcome.user_id")
