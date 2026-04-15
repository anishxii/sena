from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Protocol

from .eeg_features import EEG_FEATURE_NAMES
from .eeg_retrieval import EEGMatchResult, NearestNeighborPatientEEGRetriever
from .stew_index import STEWFeatureIndex, STEWSubjectLoader, build_stew_feature_index, load_feature_index
from .target_eeg_mapper import HeuristicTargetEEGMapper, TargetEEGContext


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


EEG_SUMMARY_FEATURE_NAMES = EEG_FEATURE_NAMES

DEFAULT_USER_TO_STEW_SUBJECT = {
    "advanced_concise": "sub01",
    "example_builder": "sub02",
    "visual_scanner": "sub03",
}


def estimate_time_on_chunk(content_text: str) -> float:
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
        self.mapper = HeuristicTargetEEGMapper()
        self.rng = random.Random(seed)

    def observe(self, context: EEGObservationContext) -> EEGWindow:
        time_on_chunk = context.time_on_chunk if context.time_on_chunk is not None else estimate_time_on_chunk(context.tutor_message)
        target_features = self.mapper.predict(
            TargetEEGContext(
                user_id=context.user_id,
                concept_id=context.concept_id,
                action_id=context.action_id,
                tutor_message=context.tutor_message,
                time_on_chunk=time_on_chunk,
                hidden_state=context.hidden_state or {},
                observable_signals=context.observable_signals or {},
            )
        )

        noisy_features = []
        for index, value in enumerate(target_features):
            if index in {4, 5}:
                noisy = max(-1.0, min(1.0, value + self.rng.gauss(0.0, 0.02)))
            elif index == 6:
                noisy = max(0.0, value + self.rng.gauss(0.0, 0.05))
            else:
                noisy = _clip01(value + self.rng.gauss(0.0, 0.01))
            noisy_features.append(round(float(noisy), 4))

        return EEGWindow(
            timestamp=float(context.timestamp),
            user_id=context.user_id,
            channels=None,
            fs=None,
            features=noisy_features,
            feature_names=EEG_SUMMARY_FEATURE_NAMES,
            metadata={
                "source": "synthetic",
                "concept_id": context.concept_id,
                "action_id": context.action_id,
                "time_on_chunk": time_on_chunk,
                "target_features": target_features,
            },
        )


class MatchedRealEEGProvider:
    """Retrieve the closest real STEW EEG window for the mapped target state."""

    def __init__(
        self,
        feature_index: STEWFeatureIndex,
        stew_dir: str | Path,
        user_to_subject: dict[str, str] | None = None,
        seed: int = 0,
        epoch_sec: int = 30,
        stride_sec: int = 10,
    ) -> None:
        self.feature_index = feature_index
        self.subject_loader = STEWSubjectLoader(stew_dir, epoch_sec=epoch_sec, stride_sec=stride_sec)
        self.user_to_subject = dict(DEFAULT_USER_TO_STEW_SUBJECT)
        if user_to_subject:
            self.user_to_subject.update(user_to_subject)
        self.mapper = HeuristicTargetEEGMapper()
        self.retriever = NearestNeighborPatientEEGRetriever(feature_index=feature_index, seed=seed)

    def observe(self, context: EEGObservationContext) -> EEGWindow:
        subject_id = self.user_to_subject.get(context.user_id, context.user_id)
        time_on_chunk = context.time_on_chunk if context.time_on_chunk is not None else estimate_time_on_chunk(context.tutor_message)
        target_features = self.mapper.predict(
            TargetEEGContext(
                user_id=context.user_id,
                concept_id=context.concept_id,
                action_id=context.action_id,
                tutor_message=context.tutor_message,
                time_on_chunk=time_on_chunk,
                hidden_state=context.hidden_state or {},
                observable_signals=context.observable_signals or {},
            )
        )
        match = self.retriever.sample_match(subject_id=subject_id, target_features=target_features, k=5)
        raw_window = self.subject_loader.load_epoch(subject_id, match.window.epoch_index)

        return EEGWindow(
            timestamp=float(context.timestamp),
            user_id=context.user_id,
            channels=raw_window.tolist(),
            fs=128,
            features=[round(float(value), 4) for value in match.window.features],
            feature_names=EEG_SUMMARY_FEATURE_NAMES,
            metadata=_match_metadata(
                match=match,
                subject_id=subject_id,
                concept_id=context.concept_id,
                action_id=context.action_id,
                time_on_chunk=time_on_chunk,
            ),
        )


def build_eeg_provider(
    *,
    eeg_mode: str,
    seed: int,
    stew_dir: str | Path,
    index_path: str | Path | None = None,
    user_to_subject: dict[str, str] | None = None,
    epoch_sec: int = 30,
    stride_sec: int = 10,
) -> EEGProvider:
    if eeg_mode == "synthetic":
        return SyntheticEEGProvider(seed=seed)
    if eeg_mode != "matched_real":
        raise ValueError(f"unknown eeg_mode: {eeg_mode}")

    feature_index = _load_or_build_feature_index(
        index_path=index_path,
        stew_dir=stew_dir,
        epoch_sec=epoch_sec,
        stride_sec=stride_sec,
    )
    return MatchedRealEEGProvider(
        feature_index=feature_index,
        stew_dir=stew_dir,
        user_to_subject=user_to_subject,
        seed=seed,
        epoch_sec=epoch_sec,
        stride_sec=stride_sec,
    )


def _load_or_build_feature_index(
    index_path: str | Path | None,
    stew_dir: str | Path,
    epoch_sec: int,
    stride_sec: int,
) -> STEWFeatureIndex:
    if index_path is not None and Path(index_path).exists():
        return load_feature_index(index_path)
    return build_stew_feature_index(
        stew_dir=stew_dir,
        feature_names=EEG_SUMMARY_FEATURE_NAMES,
        epoch_sec=epoch_sec,
        stride_sec=stride_sec,
    )


def _match_metadata(
    *,
    match: EEGMatchResult,
    subject_id: str,
    concept_id: str,
    action_id: str,
    time_on_chunk: float,
) -> dict[str, Any]:
    return {
        "source": "matched_real",
        "subject_id": subject_id,
        "window_id": match.window.window_id,
        "epoch_index": match.window.epoch_index,
        "start_offset_s": match.window.start_offset_s,
        "distance": round(match.distance, 6),
        "target_features": [round(float(value), 4) for value in match.target_features],
        "matched_features": [round(float(value), 4) for value in match.window.features],
        "concept_id": concept_id,
        "action_id": action_id,
        "time_on_chunk": time_on_chunk,
    }
