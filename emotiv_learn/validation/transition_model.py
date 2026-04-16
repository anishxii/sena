from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .windows import ExperimentWindow, NBACK_ACTION_IDS


@dataclass(frozen=True)
class NBackTransitionTarget:
    difficulty_level: int
    workload_estimate: float
    rolling_accuracy: float
    rolling_rt_percentile: float
    lapse_rate: float
    model_info: dict[str, float | int | str]


class NBackTransitionModel:
    def predict(self, current: ExperimentWindow, action_id: str) -> NBackTransitionTarget:
        raise NotImplementedError


class HeuristicNBackTransitionModel(NBackTransitionModel):
    """Small hand-built transition fallback used when no fitted model exists."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def predict(self, current: ExperimentWindow, action_id: str) -> NBackTransitionTarget:
        target_difficulty = apply_action_to_difficulty(current.difficulty_level, action_id)
        difficulty_delta = target_difficulty - current.difficulty_level
        workload = current.workload_estimate + 0.18 * difficulty_delta + self.rng.gauss(0.0, 0.035)
        accuracy = current.rolling_accuracy - 0.12 * max(difficulty_delta, 0) + 0.08 * max(-difficulty_delta, 0)
        rt = current.rolling_rt_percentile + 0.10 * difficulty_delta + self.rng.gauss(0.0, 0.03)
        lapse = current.lapse_rate + 0.05 * max(difficulty_delta, 0) - 0.03 * max(-difficulty_delta, 0)
        if action_id == "maintain_difficulty":
            workload += self.rng.gauss(0.0, 0.02)
            accuracy += self.rng.gauss(0.0, 0.02)
        return NBackTransitionTarget(
            difficulty_level=target_difficulty,
            workload_estimate=_clip01(workload),
            rolling_accuracy=_clip01(accuracy),
            rolling_rt_percentile=_clip01(rt),
            lapse_rate=_clip01(lapse),
            model_info={"model_type": "heuristic", "difficulty_delta": difficulty_delta},
        )


class FittedNBackTransitionModel(NBackTransitionModel):
    """Data-fitted transition model over indexed N-Back window summaries.

    With condition-level COG-BCI summaries we do not yet have dense adjacent
    windows. This model estimates each subject's response to difficulty changes
    from observed difficulty-level centroids, then predicts the target proxy
    state for the requested action. It is intentionally simple and will be
    replaced by a richer regressor once window-level EEG/behavior ingestion lands.
    """

    def __init__(self, windows: list[ExperimentWindow]) -> None:
        if not windows:
            raise ValueError("windows must not be empty")
        self.global_centroids = _centroids(windows)
        self.subject_centroids = {
            subject_id: _centroids([window for window in windows if window.subject_id == subject_id])
            for subject_id in sorted({window.subject_id for window in windows})
        }

    def predict(self, current: ExperimentWindow, action_id: str) -> NBackTransitionTarget:
        if action_id not in NBACK_ACTION_IDS:
            raise ValueError(f"unknown N-Back action: {action_id}")
        target_difficulty = apply_action_to_difficulty(current.difficulty_level, action_id)
        centroids = self.subject_centroids.get(current.subject_id, {})
        source_centroid = centroids.get(current.difficulty_level) or self.global_centroids.get(current.difficulty_level)
        target_centroid = centroids.get(target_difficulty) or self.global_centroids.get(target_difficulty)
        if source_centroid is None or target_centroid is None:
            return HeuristicNBackTransitionModel(seed=0).predict(current, action_id)

        # Preserve the current window's residual from its difficulty centroid,
        # then move it by the observed subject/global difficulty shift.
        workload = current.workload_estimate + (target_centroid.workload_estimate - source_centroid.workload_estimate)
        accuracy = current.rolling_accuracy + (target_centroid.rolling_accuracy - source_centroid.rolling_accuracy)
        rt = current.rolling_rt_percentile + (target_centroid.rolling_rt_percentile - source_centroid.rolling_rt_percentile)
        lapse = current.lapse_rate + (target_centroid.lapse_rate - source_centroid.lapse_rate)
        return NBackTransitionTarget(
            difficulty_level=target_difficulty,
            workload_estimate=_clip01(workload),
            rolling_accuracy=_clip01(accuracy),
            rolling_rt_percentile=_clip01(rt),
            lapse_rate=_clip01(lapse),
            model_info={
                "model_type": "fitted_centroid_shift",
                "source_difficulty": current.difficulty_level,
                "target_difficulty": target_difficulty,
                "difficulty_delta": target_difficulty - current.difficulty_level,
            },
        )


@dataclass(frozen=True)
class _Centroid:
    workload_estimate: float
    rolling_accuracy: float
    rolling_rt_percentile: float
    lapse_rate: float


def apply_action_to_difficulty(difficulty_level: int, action_id: str) -> int:
    if action_id == "decrease_difficulty":
        return max(0, difficulty_level - 1)
    if action_id == "increase_difficulty":
        return min(2, difficulty_level + 1)
    return difficulty_level


def _centroids(windows: list[ExperimentWindow]) -> dict[int, _Centroid]:
    by_difficulty: dict[int, list[ExperimentWindow]] = {}
    for window in windows:
        by_difficulty.setdefault(window.difficulty_level, []).append(window)
    return {
        difficulty: _Centroid(
            workload_estimate=_mean(window.workload_estimate for window in rows),
            rolling_accuracy=_mean(window.rolling_accuracy for window in rows),
            rolling_rt_percentile=_mean(window.rolling_rt_percentile for window in rows),
            lapse_rate=_mean(window.lapse_rate for window in rows),
        )
        for difficulty, rows in by_difficulty.items()
    }


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows)


def _clip01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return max(0.0, min(1.0, float(value)))
