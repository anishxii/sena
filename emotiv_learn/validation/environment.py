from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random

from .reward import compute_nback_reward
from .windows import ExperimentWindow, NBACK_ACTION_IDS, NBackObservation


@dataclass(frozen=True)
class NBackStepResult:
    observation: NBackObservation
    reward: float
    done: bool
    info: dict[str, float | int | str]


class NBackRestitchingEnvironment:
    """Action-conditioned restitching environment over indexed N-Back windows."""

    def __init__(self, windows: list[ExperimentWindow], max_turns: int = 30, seed: int = 0) -> None:
        if not windows:
            raise ValueError("windows must not be empty")
        self.windows = windows
        self.max_turns = max_turns
        self.rng = random.Random(seed)
        self.windows_by_subject_and_difficulty: dict[tuple[str, int], list[ExperimentWindow]] = defaultdict(list)
        for window in windows:
            if window.task != "n_back":
                continue
            self.windows_by_subject_and_difficulty[(window.subject_id, window.difficulty_level)].append(window)

        self.subject_ids = sorted({window.subject_id for window in windows})
        self.current_window: ExperimentWindow | None = None
        self.previous_action: str | None = None
        self.turn_index = 0

    def reset(self, subject_id: str | None = None, difficulty_level: int = 1) -> NBackObservation:
        subject_id = subject_id or self.rng.choice(self.subject_ids)
        candidates = self.windows_by_subject_and_difficulty.get((subject_id, difficulty_level))
        if not candidates:
            raise ValueError(f"no windows for subject_id={subject_id}, difficulty_level={difficulty_level}")
        self.current_window = self.rng.choice(candidates)
        self.previous_action = None
        self.turn_index = 1
        return self._observation()

    def step(self, action_id: str) -> NBackStepResult:
        if self.current_window is None:
            raise RuntimeError("reset must be called before step")
        if action_id not in NBACK_ACTION_IDS:
            raise ValueError(f"unknown N-Back action: {action_id}")

        current = self.current_window
        target_difficulty = _apply_action_to_difficulty(current.difficulty_level, action_id)
        target_workload, target_accuracy, target_rt, target_lapse = self._predict_next_targets(current, action_id, target_difficulty)
        next_window = self._retrieve_window(
            subject_id=current.subject_id,
            difficulty_level=target_difficulty,
            target_workload=target_workload,
            target_accuracy=target_accuracy,
            target_rt=target_rt,
            target_lapse=target_lapse,
        )

        reward = compute_nback_reward(next_window, previous_workload=current.workload_estimate)
        self.current_window = next_window
        self.previous_action = action_id
        self.turn_index += 1
        done = self.turn_index > self.max_turns
        return NBackStepResult(
            observation=self._observation(),
            reward=reward,
            done=done,
            info={
                "target_difficulty": target_difficulty,
                "target_workload": round(target_workload, 4),
                "target_accuracy": round(target_accuracy, 4),
                "retrieved_window_id": next_window.window_id,
            },
        )

    def _observation(self) -> NBackObservation:
        if self.current_window is None:
            raise RuntimeError("environment has no current window")
        return NBackObservation(
            window=self.current_window,
            previous_action=self.previous_action,
            turn_index=self.turn_index,
            max_turns=self.max_turns,
        )

    def _predict_next_targets(
        self,
        current: ExperimentWindow,
        action_id: str,
        target_difficulty: int,
    ) -> tuple[float, float, float, float]:
        """Small transition scaffold, to be replaced by a learned transition model."""

        difficulty_delta = target_difficulty - current.difficulty_level
        workload = current.workload_estimate + 0.18 * difficulty_delta + self.rng.gauss(0.0, 0.035)
        accuracy = current.rolling_accuracy - 0.12 * max(difficulty_delta, 0) + 0.08 * max(-difficulty_delta, 0)
        rt = current.rolling_rt_percentile + 0.10 * difficulty_delta + self.rng.gauss(0.0, 0.03)
        lapse = current.lapse_rate + 0.05 * max(difficulty_delta, 0) - 0.03 * max(-difficulty_delta, 0)
        if action_id == "maintain_difficulty":
            workload += self.rng.gauss(0.0, 0.02)
            accuracy += self.rng.gauss(0.0, 0.02)
        return (_clip01(workload), _clip01(accuracy), _clip01(rt), _clip01(lapse))

    def _retrieve_window(
        self,
        *,
        subject_id: str,
        difficulty_level: int,
        target_workload: float,
        target_accuracy: float,
        target_rt: float,
        target_lapse: float,
    ) -> ExperimentWindow:
        candidates = self.windows_by_subject_and_difficulty.get((subject_id, difficulty_level), [])
        if not candidates:
            candidates = [window for window in self.windows if window.difficulty_level == difficulty_level]
        if not candidates:
            raise ValueError(f"no windows available for difficulty_level={difficulty_level}")

        scored = [
            (
                _distance(
                    (target_workload, target_accuracy, target_rt, target_lapse),
                    (
                        window.workload_estimate,
                        window.rolling_accuracy,
                        window.rolling_rt_percentile,
                        window.lapse_rate,
                    ),
                ),
                window,
            )
            for window in candidates
        ]
        scored.sort(key=lambda item: item[0])
        top_k = scored[: min(4, len(scored))]
        return self.rng.choice([window for _, window in top_k])


def _apply_action_to_difficulty(difficulty_level: int, action_id: str) -> int:
    if action_id == "decrease_difficulty":
        return max(0, difficulty_level - 1)
    if action_id == "increase_difficulty":
        return min(2, difficulty_level + 1)
    return difficulty_level


def _distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((lhs - rhs) ** 2 for lhs, rhs in zip(left, right, strict=True)))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
