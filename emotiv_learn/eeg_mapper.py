from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .eeg import EEGProxyState, TargetEEGContext
from .eeg_features import EEG_FEATURE_NAMES
from .stew_index import STEWFeatureIndex


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class STEWWorkloadFeatureMapper:
    """Polynomial ridge mapper from workload to EEG summary features.

    This is the first learned bridge for the emitter stack. It is intentionally
    narrow: STEW provides subjective workload labels, so we fit the relationship
    between workload and EEG summary targets from real windows without inventing
    extra latent supervision.
    """

    feature_names: list[str]
    basis_names: list[str]
    coefficients: list[list[float]]
    feature_min: list[float]
    feature_max: list[float]

    def predict_proxy_state(self, context: TargetEEGContext) -> EEGProxyState:
        hidden_state = context.hidden_state or {}
        observables = context.observable_signals or {}

        mastery = _clip01(hidden_state.get("concept_mastery", {}).get(context.concept_id, 0.5))
        confusion = _clip01(observables.get("confusion_score", 0.5))
        fatigue = _clip01(hidden_state.get("fatigue", observables.get("fatigue", 0.3)))
        attention = _clip01(hidden_state.get("attention", observables.get("attention", 0.6)))
        engagement = _clip01(observables.get("engagement_score", hidden_state.get("engagement", 0.6)))
        confidence = _clip01(observables.get("confidence", hidden_state.get("confidence", 0.5)))

        semantic_friction = _clip01(observables.get("semantic_friction", observables.get("comprehension_difficulty", 0.5)))
        workload = _clip01(
            0.34 * confusion
            + 0.18 * fatigue
            + 0.16 * semantic_friction
            + 0.14 * (1.0 - mastery)
            + 0.10 * (1.0 - attention)
            + 0.08 * (1.0 - confidence)
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
        design = np.array(
            [
                1.0,
                proxy.workload,
                proxy.workload ** 2,
                proxy.workload * proxy.confidence,
                proxy.workload * proxy.fatigue,
            ],
            dtype=np.float64,
        )
        coefficients = np.array(self.coefficients, dtype=np.float64)
        raw_prediction = design @ coefficients
        clipped = []
        for index, value in enumerate(raw_prediction.tolist()):
            lower = self.feature_min[index]
            upper = self.feature_max[index]
            bounded = min(max(float(value), lower), upper)
            if index == 4:
                clipped.append(round(max(-1.0, min(1.0, bounded)), 6))
            elif index == 6:
                clipped.append(round(max(0.0, bounded), 6))
            else:
                clipped.append(round(bounded if index == 5 else _clip01(bounded), 6))
        return clipped


def fit_stew_workload_feature_mapper(
    feature_index: STEWFeatureIndex,
    ridge_alpha: float = 1e-3,
) -> STEWWorkloadFeatureMapper:
    rows: list[list[float]] = []
    targets: list[list[float]] = []
    for windows in feature_index.windows_by_subject.values():
        for window in windows:
            if window.workload_rating is None:
                continue
            workload = _clip01((window.workload_rating - 1.0) / 8.0)
            rows.append([1.0, workload, workload ** 2, workload * 0.5, workload * 0.5])
            targets.append(window.features)

    if not rows:
        raise ValueError("cannot fit workload mapper: no windows with workload_rating")

    x = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    xtx = x.T @ x
    ridge = ridge_alpha * np.eye(xtx.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(xtx + ridge, x.T @ y)
    feature_min = y.min(axis=0).tolist()
    feature_max = y.max(axis=0).tolist()

    return STEWWorkloadFeatureMapper(
        feature_names=list(feature_index.feature_names),
        basis_names=["bias", "workload", "workload_sq", "workload_x_confidence", "workload_x_fatigue"],
        coefficients=coefficients.tolist(),
        feature_min=[float(value) for value in feature_min],
        feature_max=[float(value) for value in feature_max],
    )


def save_stew_workload_feature_mapper(
    mapper: STEWWorkloadFeatureMapper,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(mapper), indent=2), encoding="utf-8")


def load_stew_workload_feature_mapper(path: str | Path) -> STEWWorkloadFeatureMapper:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = payload.get("feature_names")
    if expected != EEG_FEATURE_NAMES:
        raise ValueError(f"unexpected EEG feature names in mapper file: {expected}")
    return STEWWorkloadFeatureMapper(
        feature_names=list(payload["feature_names"]),
        basis_names=list(payload["basis_names"]),
        coefficients=[[float(value) for value in row] for row in payload["coefficients"]],
        feature_min=[float(value) for value in payload["feature_min"]],
        feature_max=[float(value) for value in payload["feature_max"]],
    )
