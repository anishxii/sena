from __future__ import annotations

from dataclasses import dataclass

from .eeg_features import EEG_FEATURE_NAMES
from .schemas import State, StateMetadata
from .tutor_proxy import TUTOR_PROXY_FEATURE_NAMES, derive_tutor_facing_proxy_state

COG_PROXY_FEATURE_NAMES = [
    "proxy_workload_estimate",
    "proxy_rolling_accuracy",
    "proxy_rolling_rt_percentile",
    "proxy_lapse_rate",
]

CORE_LIVE_FEATURE_NAMES = [
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

STATE_PROFILE_BEHAVIOR_ONLY = "behavior_only"
STATE_PROFILE_CURRENT_EEG = "current_eeg"
STATE_PROFILE_TUTOR_PROXY_EEG = "tutor_proxy_eeg"
STATE_PROFILES = [
    STATE_PROFILE_BEHAVIOR_ONLY,
    STATE_PROFILE_CURRENT_EEG,
    STATE_PROFILE_TUTOR_PROXY_EEG,
]

LIVE_FEATURE_NAMES = CORE_LIVE_FEATURE_NAMES + EEG_FEATURE_NAMES + COG_PROXY_FEATURE_NAMES + TUTOR_PROXY_FEATURE_NAMES


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
    eeg_features: list[float] | None = None
    eeg_proxy_estimates: dict | None = None


class LiveLLMStateBuilder:
    """Builds a compact non-EEG state from recent interpreted learner signals."""

    def __init__(self, state_profile: str = STATE_PROFILE_CURRENT_EEG) -> None:
        if state_profile not in STATE_PROFILES:
            raise ValueError(f"unknown state_profile={state_profile}; expected one of {STATE_PROFILES}")
        self.state_profile = state_profile

    def build_state(self, state_input: LiveStateInput) -> State:
        interpreted = state_input.interpreted or {}
        student_response = state_input.student_response or {}
        eeg_features = state_input.eeg_features or [0.0] * len(EEG_FEATURE_NAMES)
        eeg_proxy_estimates = state_input.eeg_proxy_estimates or {}
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

        include_eeg = self.state_profile in {STATE_PROFILE_CURRENT_EEG, STATE_PROFILE_TUTOR_PROXY_EEG}
        include_tutor_proxy = self.state_profile == STATE_PROFILE_TUTOR_PROXY_EEG
        if include_eeg:
            features.extend([_score(value, default=0.0) for value in eeg_features])
            features.extend(
                [
                    _score(eeg_proxy_estimates.get("workload_estimate"), default=0.0),
                    _score(eeg_proxy_estimates.get("rolling_accuracy"), default=0.0),
                    _score(eeg_proxy_estimates.get("rolling_rt_percentile"), default=0.0),
                    _score(eeg_proxy_estimates.get("lapse_rate"), default=0.0),
                ]
            )
        else:
            features.extend([0.0] * len(EEG_FEATURE_NAMES))
            features.extend([0.0] * len(COG_PROXY_FEATURE_NAMES))

        if include_tutor_proxy:
            tutor_proxy = derive_tutor_facing_proxy_state(
                interpreted=interpreted,
                student_response=student_response,
                eeg_proxy_estimates=eeg_proxy_estimates,
            ).as_feature_dict()
            features.extend([_score(tutor_proxy[name], default=0.0) for name in TUTOR_PROXY_FEATURE_NAMES])
        else:
            features.extend([0.0] * len(TUTOR_PROXY_FEATURE_NAMES))

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
