from __future__ import annotations

from dataclasses import dataclass


TUTOR_PROXY_FEATURE_NAMES = [
    "tutor_overload_risk",
    "tutor_repair_need",
    "tutor_challenge_readiness",
    "tutor_checkpoint_readiness",
    "tutor_curiosity_headroom",
]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class TutorFacingProxyState:
    overload_risk: float
    repair_need: float
    challenge_readiness: float
    checkpoint_readiness: float
    curiosity_headroom: float

    def as_feature_dict(self) -> dict[str, float]:
        return {
            "tutor_overload_risk": self.overload_risk,
            "tutor_repair_need": self.repair_need,
            "tutor_challenge_readiness": self.challenge_readiness,
            "tutor_checkpoint_readiness": self.checkpoint_readiness,
            "tutor_curiosity_headroom": self.curiosity_headroom,
        }


def derive_tutor_facing_proxy_state(
    *,
    interpreted: dict | None,
    student_response: dict | None,
    eeg_proxy_estimates: dict | None,
) -> TutorFacingProxyState:
    interpreted = interpreted or {}
    student_response = student_response or {}
    eeg_proxy_estimates = eeg_proxy_estimates or {}

    confusion = _clip01(interpreted.get("confusion_score", 0.5))
    comprehension = _clip01(interpreted.get("comprehension_score", 0.5))
    engagement = _clip01(interpreted.get("engagement_score", 0.5))
    progress = _clip01(interpreted.get("progress_signal", 0.0))
    pace_slow = _clip01(interpreted.get("pace_slow_score", 0.0))
    checkpoint_score = _clip01(interpreted.get("checkpoint_score", 0.5) if interpreted.get("checkpoint_score") is not None else 0.5)
    confidence = _clip01(student_response.get("self_reported_confidence", 0.5))
    followup_type = interpreted.get("followup_type", "unknown")

    workload = _clip01(eeg_proxy_estimates.get("workload_estimate", 0.0))
    rolling_accuracy = _clip01(eeg_proxy_estimates.get("rolling_accuracy", 0.0))
    rt_percentile = _clip01(eeg_proxy_estimates.get("rolling_rt_percentile", 0.0))
    lapse_rate = _clip01(eeg_proxy_estimates.get("lapse_rate", 0.0))

    continue_flag = 1.0 if followup_type == "continue" else 0.0
    clarify_flag = 1.0 if followup_type == "clarify" else 0.0
    branch_flag = 1.0 if followup_type == "branch" else 0.0

    overload_risk = _clip01(
        0.34 * workload
        + 0.22 * rt_percentile
        + 0.16 * pace_slow
        + 0.16 * confusion
        + 0.12 * lapse_rate
        - 0.10 * confidence
    )
    repair_need = _clip01(
        0.34 * confusion
        + 0.20 * (1.0 - comprehension)
        + 0.16 * (1.0 - rolling_accuracy)
        + 0.12 * lapse_rate
        + 0.12 * clarify_flag
        + 0.06 * (1.0 - progress)
    )
    challenge_readiness = _clip01(
        0.26 * comprehension
        + 0.20 * confidence
        + 0.18 * engagement
        + 0.16 * progress
        + 0.10 * continue_flag
        + 0.10 * rolling_accuracy
        - 0.12 * overload_risk
        - 0.08 * repair_need
    )
    checkpoint_readiness = _clip01(
        0.30 * rolling_accuracy
        + 0.20 * checkpoint_score
        + 0.18 * comprehension
        + 0.12 * confidence
        + 0.10 * continue_flag
        + 0.10 * progress
        - 0.16 * confusion
    )
    curiosity_headroom = _clip01(
        0.34 * engagement
        + 0.24 * branch_flag
        + 0.18 * comprehension
        + 0.10 * progress
        - 0.14 * overload_risk
        - 0.08 * repair_need
    )

    return TutorFacingProxyState(
        overload_risk=overload_risk,
        repair_need=repair_need,
        challenge_readiness=challenge_readiness,
        checkpoint_readiness=checkpoint_readiness,
        curiosity_headroom=curiosity_headroom,
    )
