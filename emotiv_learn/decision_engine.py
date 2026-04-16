from __future__ import annotations

import math
import random
import sqlite3
from collections import deque
from pathlib import Path

from .schemas import (
    ACTION_BANK,
    Action,
    ActionScores,
    DecisionTrace,
    PolicyInfo,
    RewardEvent,
    State,
    UpdateTrace,
)


class DecisionEngine:
    """Linear contextual bandit with optional user-specific residuals."""

    def __init__(
        self,
        feature_dim: int,
        epsilon: float = 0.10,
        alpha_generic: float = 0.05,
        alpha_user: float = 0.10,
        use_personalization: bool = True,
        action_bank: list[Action] | None = None,
        seed: int | None = None,
        db_path: str | None = None,
        reward_clip_abs: float | None = 1.5,
        l2_weight_decay: float = 0.0,
        update_clip_abs: float | None = None,
        max_update_history: int = 200,
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0.0, 1.0]")
        if reward_clip_abs is not None and reward_clip_abs <= 0.0:
            raise ValueError("reward_clip_abs must be positive when provided")
        if l2_weight_decay < 0.0:
            raise ValueError("l2_weight_decay must be non-negative")
        if update_clip_abs is not None and update_clip_abs <= 0.0:
            raise ValueError("update_clip_abs must be positive when provided")

        self.feature_dim = feature_dim
        self.epsilon = epsilon
        self.alpha_generic = alpha_generic
        self.alpha_user = alpha_user
        self.use_personalization = use_personalization
        self.reward_clip_abs = reward_clip_abs
        self.l2_weight_decay = l2_weight_decay
        self.update_clip_abs = update_clip_abs
        self.action_bank = action_bank or ACTION_BANK
        self.action_ids = [action.action_id for action in self.action_bank]
        self.rng = random.Random(seed)
        self.db_path = db_path

        self.generic_weights: dict[str, list[float]] = {
            action_id: [0.0] * feature_dim for action_id in self.action_ids
        }
        self.user_residuals: dict[str, dict[str, list[float]]] = {}
        self._last_scored_policy_type: str = "personalized" if use_personalization else "generic"
        self.pending_decision_traces: dict[str, DecisionTrace] = {}
        self.update_history: deque[UpdateTrace] = deque(maxlen=max_update_history)

        if self.db_path is not None:
            self._initialize_storage()
            self._load_persisted_weights()

    def score_actions(self, state: State, action_bank: list[Action]) -> ActionScores:
        self._validate_action_bank(action_bank)
        features = self._validate_state(state)
        scores = self._score_for_user(state.user_id, features)
        selected_action, exploration = self._choose_action(scores)

        action_scores = ActionScores(
            timestamp=state.timestamp,
            user_id=state.user_id,
            scores=scores,
            selected_action=selected_action,
            policy_info=PolicyInfo(
                policy_type=self._last_scored_policy_type,
                exploration=exploration,
            ),
        )
        self.pending_decision_traces[state.user_id] = DecisionTrace(
            state_timestamp=state.timestamp,
            user_id=state.user_id,
            selected_action=selected_action,
            policy_type=self._last_scored_policy_type,
            exploration=exploration,
            scores=dict(scores),
        )
        return action_scores

    def select_action(self, action_scores: ActionScores) -> Action:
        self._validate_selected_action(action_scores.selected_action)
        return Action(action_id=action_scores.selected_action, params={})

    def update(self, reward_event: RewardEvent) -> None:
        features = self._validate_reward_event(reward_event)
        action_id = reward_event.action_id
        user_id = reward_event.user_id
        trace = self.pending_decision_traces.pop(user_id, None)

        generic_score = self._dot(self.generic_weights[action_id], features)
        total_score = generic_score

        user_weights: list[float] | None = None
        user_score = 0.0
        if self.use_personalization:
            user_weights = self._get_user_weights(user_id)[action_id]
            user_score = self._dot(user_weights, features)
            total_score += user_score

        reward = float(reward_event.reward)
        if self.reward_clip_abs is not None:
            reward = max(-self.reward_clip_abs, min(self.reward_clip_abs, reward))

        error = reward - total_score
        generic_scale = self.alpha_generic * error
        user_scale = self.alpha_user * error
        if self.update_clip_abs is not None:
            generic_scale = max(-self.update_clip_abs, min(self.update_clip_abs, generic_scale))
            user_scale = max(-self.update_clip_abs, min(self.update_clip_abs, user_scale))

        self._apply_decay(self.generic_weights[action_id])
        self._apply_update(self.generic_weights[action_id], features, generic_scale)
        self._persist_generic_action(action_id)

        if user_weights is not None:
            self._apply_decay(user_weights)
            self._apply_update(user_weights, features, user_scale)
            self._persist_user_action(user_id, action_id)

        self.update_history.append(
            UpdateTrace(
                user_id=user_id,
                action_id=action_id,
                reward=reward,
                predicted_score=total_score,
                error=error,
                policy_type=trace.policy_type if trace is not None else None,
                exploration=trace.exploration if trace is not None else None,
                generic_score=generic_score,
                user_score=user_score,
            )
        )

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
        features = [float(value) for value in state.features]
        self._validate_numeric_vector(features, "state.features")
        return features

    def _validate_selected_action(self, action_id: str) -> None:
        if action_id not in self.action_ids:
            raise ValueError(f"unknown action_id: {action_id}")

    def _validate_reward_event(self, reward_event: RewardEvent) -> list[float]:
        self._validate_selected_action(reward_event.action_id)
        if not isinstance(reward_event.reward, (float, int)):
            raise ValueError("reward_event.reward must be numeric")
        reward = float(reward_event.reward)
        if not math.isfinite(reward):
            raise ValueError("reward_event.reward must be finite")
        if len(reward_event.state_features) != self.feature_dim:
            raise ValueError(
                "reward_event.state_features length must match engine feature_dim"
            )
        features = [float(value) for value in reward_event.state_features]
        self._validate_numeric_vector(features, "reward_event.state_features")
        return features

    def _apply_decay(self, weights: list[float]) -> None:
        if self.l2_weight_decay <= 0.0:
            return
        shrink = 1.0 - self.l2_weight_decay
        for index in range(len(weights)):
            weights[index] *= shrink

    def _initialize_storage(self) -> None:
        if self.db_path is None:
            return

        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(db_file) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS generic_weights (
                    action_id TEXT NOT NULL,
                    feature_index INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (action_id, feature_index)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_residuals (
                    user_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    feature_index INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    PRIMARY KEY (user_id, action_id, feature_index)
                )
                """
            )

    def _load_persisted_weights(self) -> None:
        if self.db_path is None:
            return

        with sqlite3.connect(self.db_path) as conn:
            for action_id, feature_index, weight in conn.execute(
                "SELECT action_id, feature_index, weight FROM generic_weights"
            ):
                if action_id in self.generic_weights and 0 <= feature_index < self.feature_dim:
                    self.generic_weights[action_id][feature_index] = float(weight)

            for user_id, action_id, feature_index, weight in conn.execute(
                "SELECT user_id, action_id, feature_index, weight FROM user_residuals"
            ):
                if action_id not in self.action_ids or not 0 <= feature_index < self.feature_dim:
                    continue
                user_weights = self._get_user_weights(user_id)
                user_weights[action_id][feature_index] = float(weight)

    def _persist_generic_action(self, action_id: str) -> None:
        if self.db_path is None:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO generic_weights(action_id, feature_index, weight)
                VALUES (?, ?, ?)
                ON CONFLICT(action_id, feature_index)
                DO UPDATE SET weight = excluded.weight
                """,
                [
                    (action_id, feature_index, weight)
                    for feature_index, weight in enumerate(self.generic_weights[action_id])
                ],
            )

    def _persist_user_action(self, user_id: str, action_id: str) -> None:
        if self.db_path is None:
            return

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT INTO user_residuals(user_id, action_id, feature_index, weight)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, action_id, feature_index)
                DO UPDATE SET weight = excluded.weight
                """,
                [
                    (user_id, action_id, feature_index, weight)
                    for feature_index, weight in enumerate(self.user_residuals[user_id][action_id])
                ],
            )

    @staticmethod
    def _validate_numeric_vector(values: list[float], label: str) -> None:
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"{label} must contain only finite numeric values")

    @staticmethod
    def _dot(left: list[float], right: list[float]) -> float:
        return sum(lval * rval for lval, rval in zip(left, right, strict=True))

    @staticmethod
    def _apply_update(weights: list[float], features: list[float], scale: float) -> None:
        for index, value in enumerate(features):
            weights[index] += scale * value
