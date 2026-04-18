from __future__ import annotations

from .schema import WindowedTrial


BEHAVIOR_FEATURE_NAMES = [
    "trial_progress_norm",
    "behavior_row_progress_norm",
    "behavior_correct_rate",
    "behavior_hit_rate",
    "behavior_miss_rate",
    "behavior_error_rate",
    "behavior_mistake_rate",
    "behavior_outlier_rate",
    "behavior_rt_mean_norm",
    "behavior_rt_median_norm",
    "session_index_norm",
]


def extract_behavior_feature_dict(window: WindowedTrial) -> dict[str, float]:
    payload = window.behavior_payload
    return {
        "trial_progress_norm": _clip01(payload.get("trial_progress_norm", 0.0)),
        "behavior_row_progress_norm": _clip01(payload.get("behavior_row_progress_norm", 0.0)),
        "behavior_correct_rate": _clip01(payload.get("behavior_correct_rate", 0.0)),
        "behavior_hit_rate": _clip01(payload.get("behavior_hit_rate", 0.0)),
        "behavior_miss_rate": _clip01(payload.get("behavior_miss_rate", 0.0)),
        "behavior_error_rate": _clip01(payload.get("behavior_error_rate", 0.0)),
        "behavior_mistake_rate": _clip01(payload.get("behavior_mistake_rate", 0.0)),
        "behavior_outlier_rate": _clip01(payload.get("behavior_outlier_rate", 0.0)),
        "behavior_rt_mean_norm": _clip01(payload.get("behavior_rt_mean_norm", 0.0)),
        "behavior_rt_median_norm": _clip01(payload.get("behavior_rt_median_norm", 0.0)),
        "session_index_norm": _clip01(payload.get("session_index_norm", 0.0)),
    }


def _clip01(value: float | int | str | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
