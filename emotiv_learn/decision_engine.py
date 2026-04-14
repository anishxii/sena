from __future__ import annotations

import random
from .schemas import ACTION_BANK, Action, ActionScores, PolicyInfo, RewardEvent, State


class DecisionEngine:
    """Linear contextual bandit with optional user-specific residuals."""

    def __init__(
        self,
        feature_dim: int,
        epsilon: float = 0.10,
        alpha_generic: float = 0.05,
        alpha_user: float = 0.10,
        use_personalization: bool = True,
        seed: int | None = None,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0.0, 1.0]")

        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.alpha_generic = alpha_generic
        self.alpha_user = alpha_user
        self.use_personalization = use_personalization
        self.action_ids = [action.action_id for action in ACTION_BANK]
        self.rng = random.Random(seed)

        self.generic_weights: dict[str, list[float]] = {
            action_id: [0.0] * feature_dim for action_id in self.action_ids
        }
        self.user_residuals: dict[str, dict[str, list[float]]] = {}
        self._last_scored_policy_type: str = "personalized" if use_personalization else "generic"

    def score_actions(self, state: State, action_bank: list[Action]) -> ActionScores:
        self._validate_action_bank(action_bank)
        features = self._validate_state(state)
        scores = self._score_for_user(state.user_id, features)
        selected_action, exploration = self._choose_action(scores)

        return ActionScores(
            timestamp=state.timestamp,
            user_id=state.user_id,
            scores=scores,
            selected_action=selected_action,
            policy_info=PolicyInfo(
                policy_type=self._last_scored_policy_type,
                exploration=exploration,
            ),
        )

    def select_action(self, action_scores: ActionScores) -> Action:
        self._validate_selected_action(action_scores.selected_action)
        return Action(action_id=action_scores.selected_action, params={})

    def update(self, reward_event: RewardEvent) -> None:
        features = self._validate_reward_event(reward_event)
        action_id = reward_event.action_id
        user_id = reward_event.user_id

        generic_score = self._dot(self.generic_weights[action_id], features)
        total_score = generic_score

        user_weights: list[float] | None = None
        if self.use_personalization:
            user_weights = self._get_user_weights(user_id)[action_id]
            total_score += self._dot(user_weights, features)

        error = reward_event.reward - total_score
        self._apply_update(self.generic_weights[action_id], features, self.alpha_generic * error)

        if user_weights is not None:
            self._apply_update(user_weights, features, self.alpha_user * error)

    def _score_for_user(self, user_id: str, features: list[float]) -> dict[str, float]:
        scores: dict[str, float] = {}
        self._last_scored_policy_type = "personalized" if self.use_personalization else "generic"

        user_weights_by_action = self._get_user_weights(user_id) if self.use_personalization else None
        for action_id in self.action_ids:
            score = self._dot(self.generic_weights[action_id], features)
            if user_weights_by_action is not None:
                score += self._dot(user_weights_by_action[action_id], features)
            scores[action_id] = score
        return scores

    def _choose_action(self, scores: dict[str, float]) -> tuple[str, bool]:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(self.action_ids), True

        selected_action = max(self.action_ids, key=lambda action_id: (scores[action_id], -self.action_ids.index(action_id)))
        return selected_action, False

    def _get_user_weights(self, user_id: str) -> dict[str, list[float]]:
        if user_id not in self.user_residuals:
            self.user_residuals[user_id] = {
                action_id: [0.0] * self.feature_dim for action_id in self.action_ids
            }
        return self.user_residuals[user_id]

    def _validate_action_bank(self, action_bank: list[Action]) -> None:
        incoming_ids = [action.action_id for action in action_bank]
        if incoming_ids != self.action_ids:
            raise ValueError(
                f"action bank mismatch: expected {self.action_ids}, received {incoming_ids}"
            )

    def _validate_state(self, state: State) -> list[float]:
        if len(state.features) != self.feature_dim:
            raise ValueError(
                f"state feature length mismatch: expected {self.feature_dim}, received {len(state.features)}"
            )
        if len(state.feature_names) != len(state.features):
            raise ValueError("state.feature_names must align with state.features")
        return [float(value) for value in state.features]

    def _validate_selected_action(self, action_id: str) -> None:
        if action_id not in self.action_ids:
            raise ValueError(f"unknown action_id: {action_id}")

    def _validate_reward_event(self, reward_event: RewardEvent) -> list[float]:
        self._validate_selected_action(reward_event.action_id)
        if not isinstance(reward_event.reward, (float, int)):
            raise ValueError("reward_event.reward must be numeric")
        if len(reward_event.state_features) != self.feature_dim:
            raise ValueError(
                "reward_event.state_features length must match engine feature_dim"
            )
        return [float(value) for value in reward_event.state_features]

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(lval * rval for lval, rval in zip(left, right, strict=True))

    @staticmethod
    def _apply_update(weights: list[float], features: list[float], scale: float) -> None:
        for index, value in enumerate(features):
            weights[index] += scale * value
