from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Protocol


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
class EEGWindow:
    timestamp: float
    user_id: str
    channels: list[list[float]] | None
    fs: int | None
    features: list[float]
    feature_names: list[str]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EEGObservationContext:
    timestamp: int
    user_id: str
    concept_id: str
    action_id: str
    tutor_message: str
    time_on_chunk: float | None
    hidden_state: dict[str, Any] | None
    observable_signals: dict[str, Any] | None


class EEGProvider(Protocol):
    def observe(self, context: EEGObservationContext) -> EEGWindow:
        """Return EEG-derived features aligned to the current learner turn."""


EEG_SUMMARY_FEATURE_NAMES = [
    "eeg_theta_mean",
    "eeg_alpha_mean",
    "eeg_beta_mean",
    "eeg_gamma_mean",
    "eeg_frontal_alpha_asymmetry",
    "eeg_frontal_alpha_asymmetry_abs",
    "eeg_frontal_theta_alpha_ratio_mean",
    "eeg_load_score",
]


def estimate_time_on_chunk(content_text: str) -> float:
    """Estimate reading/study time in seconds from content complexity and length."""
    words = content_text.split()
    word_count = len(words)
    complexity = _text_complexity(content_text)

    base_seconds = 18.0
    length_seconds = 0.42 * word_count
    complexity_multiplier = 1.0 + 0.85 * complexity
    estimated = (base_seconds + length_seconds) * complexity_multiplier
    return max(20.0, min(180.0, round(estimated, 1)))


class SyntheticEEGProvider:
    """Emit EEG-like summary features from simulated learner state."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)

    def observe(self, context: EEGObservationContext) -> EEGWindow:
        hidden_state = context.hidden_state or {}
        observables = context.observable_signals or {}

        mastery = _clip01(hidden_state.get("concept_mastery", {}).get(context.concept_id, 0.5))
        confusion = _clip01(observables.get("confusion_score", 0.5))
        fatigue = _clip01(hidden_state.get("fatigue", observables.get("fatigue", 0.3)))
        attention = _clip01(hidden_state.get("attention", observables.get("attention", 0.6)))
        engagement = _clip01(observables.get("engagement_score", hidden_state.get("engagement", 0.6)))
        confidence = _clip01(observables.get("confidence", hidden_state.get("confidence", 0.5)))
        complexity = _text_complexity(context.tutor_message)

        time_on_chunk = context.time_on_chunk
        if time_on_chunk is None:
            time_on_chunk = estimate_time_on_chunk(context.tutor_message)
        time_component = _clip01((float(time_on_chunk) - 30.0) / 120.0)

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

        theta_mean = _clip01(0.28 + 0.14 * load + self.rng.gauss(0.0, 0.01))
        alpha_mean = _clip01(0.34 - 0.12 * load + 0.04 * confidence + self.rng.gauss(0.0, 0.01))
        beta_mean = _clip01(0.24 + 0.05 * attention + 0.03 * load + self.rng.gauss(0.0, 0.01))
        gamma_mean = _clip01(0.14 + 0.03 * complexity + 0.02 * load + self.rng.gauss(0.0, 0.008))
        frontal_alpha_asymmetry = max(-1.0, min(1.0, 0.18 * (engagement - confusion) + self.rng.gauss(0.0, 0.02)))
        frontal_alpha_asymmetry_abs = max(-1.0, min(1.0, 0.12 * (confidence - fatigue) + self.rng.gauss(0.0, 0.02)))
        frontal_theta_alpha_ratio_mean = max(
            0.0,
            0.45 + 1.45 * load + 0.15 * (1.0 - confidence) + self.rng.gauss(0.0, 0.05),
        )

        features = [
            round(theta_mean, 4),
            round(alpha_mean, 4),
            round(beta_mean, 4),
            round(gamma_mean, 4),
            round(frontal_alpha_asymmetry, 4),
            round(frontal_alpha_asymmetry_abs, 4),
            round(frontal_theta_alpha_ratio_mean, 4),
            round(load, 4),
        ]

        return EEGWindow(
            timestamp=float(context.timestamp),
            user_id=context.user_id,
            channels=None,
            fs=None,
            features=features,
            feature_names=EEG_SUMMARY_FEATURE_NAMES,
            metadata={
                "source": "synthetic",
                "concept_id": context.concept_id,
                "action_id": context.action_id,
                "message_complexity": round(complexity, 4),
                "time_on_chunk": time_on_chunk,
                "confidence": round(confidence, 4),
                "attention": round(attention, 4),
                "engagement": round(engagement, 4),
            },
        )
