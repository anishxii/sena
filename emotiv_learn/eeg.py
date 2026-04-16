from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Protocol

from .eeg_features import EEG_FEATURE_NAMES
from .eeg_retrieval import NearestNeighborEEGRetriever
from .stew_index import STEWFeatureIndex, STEWSubjectLoader, build_stew_feature_index, load_feature_index


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _hidden_bucket(hidden_state: dict[str, Any], key: str) -> dict[str, Any]:
    value = hidden_state.get(key)
    return value if isinstance(value, dict) else {}


def _hidden_value(hidden_state: dict[str, Any], *, bucket: str, key: str, default: float) -> float:
    bucket_payload = _hidden_bucket(hidden_state, bucket)
    if key in bucket_payload:
        return _clip01(bucket_payload[key])
    return _clip01(hidden_state.get(key, default))


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


EEG_SUMMARY_FEATURE_NAMES = EEG_FEATURE_NAMES


@dataclass(frozen=True)
class EEGProxyState:
    workload: float
    fatigue: float
    attention: float
    engagement: float
    confidence: float
    semantic_friction: float


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
    def predict_proxy_state(self, context: TargetEEGContext) -> EEGProxyState:
        ...

    def predict_features(self, context: TargetEEGContext) -> list[float]:
        ...


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

    def __init__(self, seed: int = 0, mapper: TargetEEGMapper | None = None) -> None:
        self.rng = random.Random(seed)
        self.mapper = mapper or HeuristicTargetEEGMapper()

    def observe(self, context: EEGObservationContext) -> EEGWindow:
        time_on_chunk = context.time_on_chunk
        if time_on_chunk is None:
            time_on_chunk = estimate_time_on_chunk(context.tutor_message)
        target_context = TargetEEGContext(
            user_id=context.user_id,
            concept_id=context.concept_id,
            action_id=context.action_id,
            tutor_message=context.tutor_message,
            time_on_chunk=time_on_chunk,
            hidden_state=context.hidden_state or {},
            observable_signals=context.observable_signals or {},
        )
        proxy_state = self.mapper.predict_proxy_state(target_context)
        target_features = self.mapper.predict_features(target_context)
        features = []
        for index, value in enumerate(target_features):
            if index == 4:
                noisy = max(-1.0, min(1.0, value + self.rng.gauss(0.0, 0.02)))
            elif index == 6:
                noisy = max(0.0, value + self.rng.gauss(0.0, 0.05))
            elif index == 5:
                noisy = value + self.rng.gauss(0.0, 0.02)
            else:
                noisy = _clip01(value + self.rng.gauss(0.0, 0.01))
            features.append(round(float(noisy), 4))

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
                "time_on_chunk": time_on_chunk,
                "proxy_state": proxy_state.__dict__,
                "target_features": target_features,
            },
        )


class RetrievedEEGProvider:
    """Retrieve the nearest real STEW window for the target EEG state."""

    def __init__(
        self,
        feature_index: STEWFeatureIndex,
        stew_dir: str | Path,
        mapper: TargetEEGMapper | None = None,
        user_to_subject: dict[str, str] | None = None,
        seed: int = 0,
        epoch_sec: int = 30,
        stride_sec: int = 10,
    ) -> None:
        self.feature_index = feature_index
        self.subject_loader = STEWSubjectLoader(stew_dir, epoch_sec=epoch_sec, stride_sec=stride_sec)
        self.mapper = mapper or HeuristicTargetEEGMapper()
        self.retriever = NearestNeighborEEGRetriever(feature_index=feature_index, seed=seed)
        self.user_to_subject = user_to_subject or {}

    def observe(self, context: EEGObservationContext) -> EEGWindow:
        time_on_chunk = context.time_on_chunk
        if time_on_chunk is None:
            time_on_chunk = estimate_time_on_chunk(context.tutor_message)
        target_context = TargetEEGContext(
            user_id=context.user_id,
            concept_id=context.concept_id,
            action_id=context.action_id,
            tutor_message=context.tutor_message,
            time_on_chunk=time_on_chunk,
            hidden_state=context.hidden_state or {},
            observable_signals=context.observable_signals or {},
        )
        subject_id = self.user_to_subject.get(context.user_id) or self._default_subject_id()
        proxy_state = self.mapper.predict_proxy_state(target_context)
        target_features = self.mapper.predict_features(target_context)
        match = self.retriever.sample_match(subject_id=subject_id, target_features=target_features, k=5)
        raw_window = self.subject_loader.load_epoch(match.window.subject_id, match.window.condition, match.window.epoch_index)

        return EEGWindow(
            timestamp=float(context.timestamp),
            user_id=context.user_id,
            channels=raw_window.tolist(),
            fs=128,
            features=[round(float(value), 4) for value in match.window.features],
            feature_names=EEG_SUMMARY_FEATURE_NAMES,
            metadata={
                "source": "retrieved_real",
                "subject_id": match.window.subject_id,
                "condition": match.window.condition,
                "window_id": match.window.window_id,
                "distance": round(match.distance, 6),
                "proxy_state": proxy_state.__dict__,
                "target_features": [round(float(value), 4) for value in target_features],
                "matched_features": [round(float(value), 4) for value in match.window.features],
            },
        )

    def _default_subject_id(self) -> str:
        if not self.feature_index.windows_by_subject:
            raise ValueError("feature index has no subjects")
        return sorted(self.feature_index.windows_by_subject)[0]


class HeuristicTargetEEGMapper:
    """Default mapper from learner state into proxy state and EEG summary targets.

    This is a baseline for infrastructure wiring. The intended upgrade path is a
    trained mapper from hidden/observable cognitive state to EEG proxy state.
    """

    def predict_proxy_state(self, context: TargetEEGContext) -> EEGProxyState:
        hidden_state = context.hidden_state or {}
        observables = context.observable_signals or {}

        knowledge_state = _hidden_bucket(hidden_state, "knowledge_state")
        mastery = _clip01(knowledge_state.get("concept_mastery", {}).get(context.concept_id, hidden_state.get("concept_mastery", {}).get(context.concept_id, 0.5)))
        confusion = _clip01(observables.get("confusion_score", 0.5))
        fatigue = _hidden_value(hidden_state, bucket="neuro_state", key="fatigue", default=observables.get("fatigue", 0.3))
        attention = _hidden_value(hidden_state, bucket="neuro_state", key="attention", default=observables.get("attention", 0.6))
        engagement = _clip01(observables.get("engagement_score", _hidden_value(hidden_state, bucket="neuro_state", key="engagement", default=0.6)))
        confidence = _clip01(observables.get("confidence", _hidden_value(hidden_state, bucket="knowledge_state", key="confidence", default=0.5)))
        workload = _hidden_value(hidden_state, bucket="neuro_state", key="workload", default=0.35)
        vigilance = _hidden_value(hidden_state, bucket="neuro_state", key="vigilance", default=0.6)
        stress = _hidden_value(hidden_state, bucket="neuro_state", key="stress", default=0.25)
        semantic_friction = _text_complexity(context.tutor_message)
        workload = _clip01(
            0.42 * workload
            + 0.18 * confusion
            + 0.14 * fatigue
            + 0.18 * semantic_friction
            + 0.06 * (1.0 - mastery)
            + 0.06 * stress
            + 0.05 * (1.0 - attention)
            - 0.04 * confidence
            - 0.03 * vigilance
        )
        return EEGProxyState(
            workload=workload,
            fatigue=fatigue,
            attention=attention,
            engagement=engagement,
            confidence=confidence,
            semantic_friction=semantic_friction,
        )

    def predict_features(self, context: TargetEEGContext) -> list[float]:
        proxy = self.predict_proxy_state(context)
        theta_mean = _clip01(0.26 + 0.16 * proxy.workload + 0.03 * proxy.fatigue)
        alpha_mean = _clip01(0.36 - 0.13 * proxy.workload + 0.04 * proxy.confidence)
        beta_mean = _clip01(0.22 + 0.08 * proxy.attention + 0.03 * proxy.workload)
        gamma_mean = _clip01(0.13 + 0.04 * proxy.semantic_friction + 0.02 * proxy.workload)
        frontal_alpha_asymmetry = max(-1.0, min(1.0, 0.18 * (proxy.engagement - proxy.workload)))
        frontal_alpha_asymmetry_abs = max(-1.0, min(1.0, 0.14 * (proxy.confidence - proxy.fatigue)))
        frontal_theta_alpha_ratio_mean = max(0.0, 0.45 + 1.45 * proxy.workload + 0.15 * (1.0 - proxy.confidence))
        return [
            round(theta_mean, 6),
            round(alpha_mean, 6),
            round(beta_mean, 6),
            round(gamma_mean, 6),
            round(frontal_alpha_asymmetry, 6),
            round(frontal_alpha_asymmetry_abs, 6),
            round(frontal_theta_alpha_ratio_mean, 6),
            round(proxy.workload, 6),
        ]


def build_eeg_provider(
    *,
    eeg_mode: str,
    seed: int,
    stew_dir: str | None = None,
    index_path: str | None = None,
    mapper_path: str | None = None,
    epoch_sec: int = 30,
    stride_sec: int = 10,
    user_to_subject: dict[str, str] | None = None,
) -> EEGProvider:
    mapper = None
    if mapper_path:
        from .eeg_mapper import load_stew_workload_feature_mapper

        mapper = load_stew_workload_feature_mapper(mapper_path)
    if eeg_mode == "synthetic":
        return SyntheticEEGProvider(seed=seed, mapper=mapper)
    if eeg_mode != "retrieved_real":
        raise ValueError(f"unknown eeg_mode: {eeg_mode}")
    if stew_dir is None:
        raise ValueError("stew_dir is required for retrieved_real mode")
    feature_index = load_feature_index(index_path) if index_path else build_stew_feature_index(
        stew_dir=stew_dir,
        feature_names=EEG_SUMMARY_FEATURE_NAMES,
        epoch_sec=epoch_sec,
        stride_sec=stride_sec,
    )
    return RetrievedEEGProvider(
        feature_index=feature_index,
        stew_dir=stew_dir,
        mapper=mapper,
        user_to_subject=user_to_subject,
        seed=seed,
        epoch_sec=epoch_sec,
        stride_sec=stride_sec,
    )
