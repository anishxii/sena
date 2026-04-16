from __future__ import annotations

from typing import Any


def compute_reward_from_interpreted(interpreted: dict[str, Any]) -> float:
    """Legacy reward used by the original live LLM loop.

    This reward is intentionally kept for backward compatibility. New policy
    comparisons should prefer `compute_observable_learning_reward`, which avoids
    heavily punishing clarification when the learner is making a reasonable
    repair move.
    """

    reward = 0.0
    if interpreted["checkpoint_correct"] is True:
        reward += 1.0
    elif interpreted["checkpoint_correct"] is False:
        reward -= 0.8

    reward += 0.6 * interpreted["progress_signal"]
    reward += 0.4 * interpreted["comprehension_score"]
    reward += 0.2 * interpreted["engagement_score"]
    reward += 0.3 * int(interpreted["followup_type"] == "branch")
    reward += 0.1 * int(interpreted["followup_type"] == "continue")
    reward -= 0.25 * int(interpreted["followup_type"] == "clarify")
    reward -= 0.6 * interpreted["confusion_score"]
    reward -= 0.3 * interpreted["pace_slow_score"]
    reward -= 0.2 * interpreted["pace_fast_score"]
    return _clip_reward(reward)


def compute_observable_learning_reward(interpreted: dict[str, Any]) -> float:
    """Reward from observable learner behavior only.

    The intent is to value learning evidence without treating every clarification
    as failure. A clarification with improving confidence and moderate
    comprehension should be close to neutral, while a failed checkpoint or high
    confusion remains negative.
    """

    reward = 0.0
    if interpreted["checkpoint_correct"] is True:
        reward += 1.0
    elif interpreted["checkpoint_correct"] is False:
        reward -= 0.9

    reward += 0.55 * interpreted["comprehension_score"]
    reward += 0.35 * interpreted["progress_signal"]
    reward += 0.20 * interpreted["engagement_score"]
    reward += 0.18 * int(interpreted["followup_type"] == "branch")
    reward += 0.12 * int(interpreted["followup_type"] == "continue")
    reward -= 0.08 * int(interpreted["followup_type"] == "clarify")
    reward -= 0.45 * interpreted["confusion_score"]
    reward -= 0.18 * interpreted["pace_slow_score"]
    reward -= 0.10 * interpreted["pace_fast_score"]
    return _clip_reward(reward)


def _clip_reward(value: float) -> float:
    return max(-1.5, min(1.5, float(value)))
