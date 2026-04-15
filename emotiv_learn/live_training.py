from __future__ import annotations

from dataclasses import dataclass

from .eeg import EEG_SUMMARY_FEATURE_NAMES, EEGWindow
from .schemas import State, StateMetadata


LIVE_FEATURE_NAMES = [
    "student_confidence",
    "turn_index_norm",
    "difficulty_easy",
    "difficulty_medium",
    "difficulty_hard",
    "previous_reward_norm",
] + EEG_SUMMARY_FEATURE_NAMES


@dataclass(frozen=True)
class LiveStateInput:
    timestamp: int
    user_id: str
    topic_id: str
    task_type: str
    difficulty: str
    turn_index: int
    max_turns: int
    interpreted: dict | None
    student_response: dict | None
    previous_reward: float
    eeg_window: EEGWindow | None


class LiveLLMStateBuilder:
    """Builds the RL state from non-interpreted context plus EEG observations."""

    def build_state(self, state_input: LiveStateInput) -> State:
        student_response = state_input.student_response or {}
        eeg_window = state_input.eeg_window
        difficulty = state_input.difficulty

        features = [
            _score(student_response.get("self_reported_confidence"), default=0.5),
            min(state_input.turn_index / max(state_input.max_turns, 1), 1.0),
            1.0 if difficulty == "easy" else 0.0,
            1.0 if difficulty == "medium" else 0.0,
            1.0 if difficulty == "hard" else 0.0,
            _normalize_reward(state_input.previous_reward),
            *_eeg_features(eeg_window),
        ]

        return State(
            timestamp=state_input.timestamp,
            user_id=state_input.user_id,
            features=features,
            feature_names=LIVE_FEATURE_NAMES,
            metadata=StateMetadata(
                task_type=state_input.task_type,
                difficulty=difficulty,
                topic_id=state_input.topic_id,
            ),
        )


def _score(value, default: float) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


def _normalize_reward(value: float) -> float:
    # Rewards are generally clipped to [-1.5, 1.5]; map to [0, 1].
    return max(0.0, min(1.0, (float(value) + 1.5) / 3.0))


def _eeg_features(eeg_window: EEGWindow | None) -> list[float]:
    if eeg_window is None:
        return _default_eeg_features()

    feature_map = dict(zip(eeg_window.feature_names, eeg_window.features, strict=False))
    return [float(feature_map.get(name, default)) for name, default in zip(EEG_SUMMARY_FEATURE_NAMES, _default_eeg_features(), strict=False)]


def _default_eeg_features() -> list[float]:
    # Neutral EEG summary priors for first-turn state construction.
    return [0.30, 0.30, 0.25, 0.15, 0.0, 0.0, 1.0, 0.5]
