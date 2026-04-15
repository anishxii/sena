from __future__ import annotations

from dataclasses import dataclass

from .schemas import State, StateMetadata


LIVE_FEATURE_NAMES = [
    "last_confusion_score",
    "last_comprehension_score",
    "last_engagement_score",
    "last_progress_signal",
    "last_pace_fast_score",
    "last_pace_slow_score",
    "last_response_continue",
    "last_response_clarify",
    "last_response_branch",
    "last_response_other",
    "student_confidence",
    "turn_index_norm",
    "difficulty_easy",
    "difficulty_medium",
    "difficulty_hard",
    "previous_reward_norm",
]


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


class LiveLLMStateBuilder:
    """Builds a compact non-EEG state from recent LLM-interpreted learner signals."""

    def build_state(self, state_input: LiveStateInput) -> State:
        interpreted = state_input.interpreted or {}
        student_response = state_input.student_response or {}
        followup_type = interpreted.get("followup_type", "unknown")
        difficulty = state_input.difficulty

        features = [
            _score(interpreted.get("confusion_score"), default=0.5),
            _score(interpreted.get("comprehension_score"), default=0.5),
            _score(interpreted.get("engagement_score"), default=0.5),
            _score(interpreted.get("progress_signal"), default=0.0),
            _score(interpreted.get("pace_fast_score"), default=0.0),
            _score(interpreted.get("pace_slow_score"), default=0.0),
            1.0 if followup_type == "continue" else 0.0,
            1.0 if followup_type == "clarify" else 0.0,
            1.0 if followup_type == "branch" else 0.0,
            1.0 if followup_type not in {"continue", "clarify", "branch"} else 0.0,
            _score(student_response.get("self_reported_confidence"), default=0.5),
            min(state_input.turn_index / max(state_input.max_turns, 1), 1.0),
            1.0 if difficulty == "easy" else 0.0,
            1.0 if difficulty == "medium" else 0.0,
            1.0 if difficulty == "hard" else 0.0,
            _normalize_reward(state_input.previous_reward),
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
