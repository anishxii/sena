from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .eeg_features import EEG_FEATURE_NAMES


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _text_complexity(text: str) -> float:
    sentences = [
        sentence.strip()
        for sentence in text.replace("?", ".").replace("!", ".").split(".")
        if sentence.strip()
    ]
    words = text.lower().split()
    if not sentences or not words:
        return 0.5

    avg_sentence_len = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    vocab_diversity = len(set(words)) / len(words)
    return _clip01(0.6 * ((avg_sentence_len - 8.0) / 22.0) + 0.4 * ((vocab_diversity - 0.30) / 0.60))


@dataclass(frozen=True)
class TargetEEGContext:
    user_id: str
    concept_id: str
    action_id: str
    tutor_message: str
    time_on_chunk: float
    hidden_state: dict[str, Any]
    observable_signals: dict[str, Any]


class TargetEEGMapper(Protocol):
    def predict(self, context: TargetEEGContext) -> list[float]:
        ...


class HeuristicTargetEEGMapper:
    """Map hidden learner state to target EEG summary values for retrieval."""

    feature_names = EEG_FEATURE_NAMES

    def predict(self, context: TargetEEGContext) -> list[float]:
        hidden_state = context.hidden_state or {}
        observables = context.observable_signals or {}

        mastery = _clip01(hidden_state.get("concept_mastery", {}).get(context.concept_id, 0.5))
        confusion = _clip01(observables.get("confusion_score", 0.5))
        fatigue = _clip01(hidden_state.get("fatigue", observables.get("fatigue", 0.3)))
        attention = _clip01(hidden_state.get("attention", observables.get("attention", 0.6)))
        engagement = _clip01(observables.get("engagement_score", hidden_state.get("engagement", 0.6)))
        confidence = _clip01(observables.get("confidence", hidden_state.get("confidence", 0.5)))
        complexity = _text_complexity(context.tutor_message)
        time_component = _clip01((float(context.time_on_chunk) - 30.0) / 120.0)

        load = _clip01(
            0.30 * confusion
            + 0.20 * fatigue
            + 0.15 * complexity
            + 0.15 * (1.0 - mastery)
            + 0.10 * (1.0 - attention)
            + 0.10 * time_component
            - 0.08 * confidence
            - 0.07 * engagement
        )

        theta_mean = _clip01(0.28 + 0.14 * load)
        alpha_mean = _clip01(0.34 - 0.12 * load + 0.04 * confidence)
        beta_mean = _clip01(0.24 + 0.05 * attention + 0.03 * load)
        gamma_mean = _clip01(0.14 + 0.03 * complexity + 0.02 * load)
        frontal_alpha_asymmetry = max(-1.0, min(1.0, 0.18 * (engagement - confusion)))
        frontal_alpha_asymmetry_abs = max(-1.0, min(1.0, 0.12 * (confidence - fatigue)))
        frontal_theta_alpha_ratio_mean = max(0.0, 0.45 + 1.45 * load + 0.15 * (1.0 - confidence))

        return [
            round(theta_mean, 6),
            round(alpha_mean, 6),
            round(beta_mean, 6),
            round(gamma_mean, 6),
            round(frontal_alpha_asymmetry, 6),
            round(frontal_alpha_asymmetry_abs, 6),
            round(frontal_theta_alpha_ratio_mean, 6),
            round(load, 6),
        ]
