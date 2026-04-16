from __future__ import annotations

from dataclasses import dataclass
import math
import random

import numpy as np

from .stew_index import IndexedEEGWindow, STEWFeatureIndex


@dataclass(frozen=True)
class EEGMatchResult:
    window: IndexedEEGWindow
    distance: float
    target_features: list[float]
    target_normalized_features: list[float]


class NearestNeighborEEGRetriever:
    def __init__(
        self,
        feature_index: STEWFeatureIndex,
        feature_weights: list[float] | None = None,
        seed: int = 0,
    ) -> None:
        self.feature_index = feature_index
        self.feature_weights = feature_weights or [1.0, 1.0, 0.7, 0.5, 0.4, 0.4, 1.5, 1.6]
        self.rng = random.Random(seed)

    def retrieve(self, subject_id: str, target_features: list[float], k: int = 5) -> list[EEGMatchResult]:
        candidates = self.feature_index.windows_by_subject.get(subject_id)
        if not candidates:
            raise ValueError(f"no indexed STEW windows found for subject_id={subject_id}")

        target_normalized = self._normalize(target_features)
        scored = [
            EEGMatchResult(
                window=candidate,
                distance=self._distance(target_normalized, candidate.normalized_features),
                target_features=[float(value) for value in target_features],
                target_normalized_features=[float(value) for value in target_normalized],
            )
            for candidate in candidates
        ]
        scored.sort(key=lambda item: item.distance)
        return scored[: max(1, min(k, len(scored)))]

    def sample_match(self, subject_id: str, target_features: list[float], k: int = 5, temperature: float = 0.05) -> EEGMatchResult:
        matches = self.retrieve(subject_id=subject_id, target_features=target_features, k=k)
        if len(matches) == 1:
            return matches[0]

        min_distance = min(match.distance for match in matches)
        scaled = [math.exp(-(match.distance - min_distance) / max(temperature, 1e-6)) for match in matches]
        total = sum(scaled)
        if total <= 0.0:
            return matches[0]
        threshold = self.rng.random()
        cumulative = 0.0
        for match, weight in zip(matches, scaled, strict=False):
            cumulative += weight / total
            if threshold <= cumulative:
                return match
        return matches[-1]

    def _normalize(self, target_features: list[float]) -> list[float]:
        mean = np.array(self.feature_index.feature_mean, dtype=np.float64)
        std = np.array(self.feature_index.feature_std, dtype=np.float64)
        target = np.array(target_features, dtype=np.float64)
        return ((target - mean) / std).tolist()

    def _distance(self, left: list[float], right: list[float]) -> float:
        return float(sum(weight * (lhs - rhs) ** 2 for lhs, rhs, weight in zip(left, right, self.feature_weights, strict=False)))
