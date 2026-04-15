from __future__ import annotations

from .windows import ExperimentWindow


def compute_nback_reward(window: ExperimentWindow, previous_workload: float | None = None) -> float:
    """Reward productive challenge, not raw correctness.

    The policy should avoid both easy-mode reward hacking and overload. Accuracy
    around 80% and workload around 55% are treated as the productive zone.
    """

    target_accuracy_score = _triangular_score(window.rolling_accuracy, target=0.80, width=0.45)
    target_workload_score = _triangular_score(window.workload_estimate, target=0.55, width=0.45)
    challenge_bonus = 0.18 * (window.difficulty_level / 2.0) * target_accuracy_score
    overload_penalty = max(0.0, window.workload_estimate - 0.78)
    underload_penalty = max(0.0, 0.25 - window.workload_estimate)
    lapse_penalty = window.lapse_rate
    instability_penalty = 0.0
    if previous_workload is not None:
        instability_penalty = abs(window.workload_estimate - previous_workload)

    reward = (
        0.40 * target_accuracy_score
        + 0.32 * target_workload_score
        + challenge_bonus
        - 0.35 * overload_penalty
        - 0.25 * underload_penalty
        - 0.30 * lapse_penalty
        - 0.10 * instability_penalty
    )
    return round(max(-1.0, min(1.0, reward)), 6)


def _triangular_score(value: float, target: float, width: float) -> float:
    return max(0.0, 1.0 - abs(float(value) - target) / width)
