from __future__ import annotations

from dataclasses import dataclass


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class RepromptAnalysis:
    followup_type: str
    confusion_score: float
    comprehension_score: float
    engagement_score: float
    progress_signal: float
    pace_fast_score: float
    pace_slow_score: float
    curiosity_score: float
    confidence_score: float

    def to_interpreted(self) -> dict:
        return {
            "followup_type": self.followup_type,
            "checkpoint_correct": None,
            "checkpoint_score": None,
            "confusion_score": self.confusion_score,
            "comprehension_score": self.comprehension_score,
            "engagement_score": self.engagement_score,
            "progress_signal": self.progress_signal,
            "pace_fast_score": self.pace_fast_score,
            "pace_slow_score": self.pace_slow_score,
            "evidence": {
                "confusion_phrases": [],
                "understanding_phrases": [],
                "curiosity_phrases": [],
            },
        }


def analyze_reprompt(reprompt: str | None, response_type: str, self_reported_confidence: float, current_mastery: float) -> RepromptAnalysis:
    text = (reprompt or "").lower()
    confusion_markers = sum(text.count(token) for token in ["confused", "not sure", "don't understand", "explain", "stuck", "missing"])
    curiosity_markers = sum(text.count(token) for token in ["why", "deeper", "more", "example", "how does", "what if", "connect"])
    understanding_markers = sum(text.count(token) for token in ["i think", "i understand", "i see", "that makes sense"])

    confusion = _clip01(0.18 * confusion_markers + 0.30 * int(response_type == "clarify") + 0.30 * (1.0 - current_mastery) - 0.18 * self_reported_confidence)
    comprehension = _clip01(0.55 * current_mastery + 0.15 * understanding_markers + 0.20 * self_reported_confidence - 0.20 * confusion)
    curiosity = _clip01(0.18 * curiosity_markers + 0.30 * int(response_type == "branch") + 0.25 * self_reported_confidence)
    engagement = _clip01(0.45 + 0.25 * curiosity + 0.10 * comprehension - 0.10 * confusion)
    progress = _clip01(0.55 * comprehension + 0.15 * self_reported_confidence - 0.15 * confusion)
    pace_fast = _clip01(comprehension - 0.20 * confusion)
    pace_slow = _clip01(confusion + 0.20 * int(response_type == "clarify") - 0.10 * self_reported_confidence)

    return RepromptAnalysis(
        followup_type=response_type,
        confusion_score=confusion,
        comprehension_score=comprehension,
        engagement_score=engagement,
        progress_signal=progress,
        pace_fast_score=pace_fast,
        pace_slow_score=pace_slow,
        curiosity_score=curiosity,
        confidence_score=_clip01(self_reported_confidence),
    )
