from __future__ import annotations

import random

from .windows import ExperimentWindow, NBACK_ACTION_IDS


class FixedMaintainPolicy:
    def select_action(self, window: ExperimentWindow) -> str:
        return "maintain_difficulty"


class RandomNBackPolicy:
    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def select_action(self, window: ExperimentWindow) -> str:
        return self.rng.choice(NBACK_ACTION_IDS)


class ClassicWorkloadRulePolicy:
    """Prior-work-style fixed rule: high load reduces demand, low load increases it."""

    def __init__(self, low_threshold: float = 0.40, high_threshold: float = 0.72) -> None:
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def select_action(self, window: ExperimentWindow) -> str:
        if window.workload_estimate >= self.high_threshold:
            return "decrease_difficulty"
        if window.workload_estimate <= self.low_threshold:
            return "increase_difficulty"
        return "maintain_difficulty"
