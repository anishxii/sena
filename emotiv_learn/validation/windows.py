from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

from emotiv_learn.schemas import Action, State, StateMetadata


NBACK_ACTION_IDS = [
    "decrease_difficulty",
    "maintain_difficulty",
    "increase_difficulty",
]
NBACK_ACTION_BANK = [Action(action_id=action_id, params={}) for action_id in NBACK_ACTION_IDS]

NBACK_FEATURE_NAMES = [
    "workload_estimate",
    "rolling_accuracy",
    "rolling_rt_percentile",
    "lapse_rate",
    "difficulty_level_norm",
    "previous_action_decrease",
    "previous_action_maintain",
    "previous_action_increase",
    "turn_index_norm",
]


@dataclass(frozen=True)
class ExperimentWindow:
    """A real or fixture EEG/behavior window indexed for validation replay."""

    window_id: str
    subject_id: str
    session_id: str
    task: str
    difficulty_level: int
    workload_estimate: float
    rolling_accuracy: float
    rolling_rt_percentile: float
    lapse_rate: float
    eeg_features: list[float] = field(default_factory=list)
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass(frozen=True)
class NBackObservation:
    window: ExperimentWindow
    previous_action: str | None
    turn_index: int
    max_turns: int


class NBackStateBuilder:
    """Convert N-Back validation observations into DecisionEngine State."""

    def build_state(self, observation: NBackObservation) -> State:
        window = observation.window
        previous_action = observation.previous_action
        features = [
            _clip01(window.workload_estimate),
            _clip01(window.rolling_accuracy),
            _clip01(window.rolling_rt_percentile),
            _clip01(window.lapse_rate),
            _clip01(window.difficulty_level / 2.0),
            1.0 if previous_action == "decrease_difficulty" else 0.0,
            1.0 if previous_action in {None, "maintain_difficulty"} else 0.0,
            1.0 if previous_action == "increase_difficulty" else 0.0,
            _clip01(observation.turn_index / max(observation.max_turns, 1)),
        ]
        return State(
            timestamp=observation.turn_index,
            user_id=window.subject_id,
            features=features,
            feature_names=NBACK_FEATURE_NAMES,
            metadata=StateMetadata(
                task_type="validation_n_back",
                difficulty=str(window.difficulty_level),
                topic_id=window.task,
            ),
        )


def build_toy_nback_windows(seed: int = 0, subjects: int = 4, windows_per_level: int = 12) -> list[ExperimentWindow]:
    """Create a small deterministic fixture bank with N-Back-like proxy states.

    This is only a local scaffold. The same schema is intended to be populated by
    COG-BCI-derived windows once the dataset ingestion lands.
    """

    rng = random.Random(seed)
    windows: list[ExperimentWindow] = []
    for subject_index in range(subjects):
        subject_id = f"toy_sub{subject_index + 1:02d}"
        capacity = 0.35 + 0.15 * subject_index
        fatigue = 0.08 * subject_index
        for difficulty_level in range(3):
            for window_index in range(windows_per_level):
                challenge = difficulty_level / 2.0
                overload = challenge - capacity + fatigue
                workload = _clip01(0.35 + 0.38 * difficulty_level + 0.18 * fatigue + rng.gauss(0.0, 0.035))
                accuracy = _clip01(0.88 - 0.42 * max(overload, 0.0) + 0.10 * max(-overload, 0.0) + rng.gauss(0.0, 0.045))
                rt_percentile = _clip01(0.35 + 0.36 * workload + 0.22 * max(overload, 0.0) + rng.gauss(0.0, 0.04))
                lapse_rate = _clip01(0.03 + 0.22 * max(overload, 0.0) + 0.12 * fatigue + rng.gauss(0.0, 0.025))
                window_id = f"{subject_id}_d{difficulty_level}_{window_index:03d}"
                windows.append(
                    ExperimentWindow(
                        window_id=window_id,
                        subject_id=subject_id,
                        session_id="toy_ses01",
                        task="n_back",
                        difficulty_level=difficulty_level,
                        workload_estimate=round(workload, 4),
                        rolling_accuracy=round(accuracy, 4),
                        rolling_rt_percentile=round(rt_percentile, 4),
                        lapse_rate=round(lapse_rate, 4),
                        eeg_features=[
                            round(_clip01(0.25 + 0.30 * workload + rng.gauss(0.0, 0.02)), 4),
                            round(_clip01(0.45 - 0.20 * workload + rng.gauss(0.0, 0.02)), 4),
                            round(_clip01(0.25 + 0.12 * workload + rng.gauss(0.0, 0.02)), 4),
                        ],
                        metadata={
                            "source": "toy_fixture",
                            "capacity": round(capacity, 4),
                            "fatigue": round(fatigue, 4),
                        },
                    )
                )
    return windows


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
