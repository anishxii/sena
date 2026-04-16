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


def _hidden_bucket(hidden_state: dict, key: str) -> dict:
    value = hidden_state.get(key)
    return value if isinstance(value, dict) else {}


def _hidden_value(hidden_state: dict, *, bucket: str, key: str, default: float) -> float:
    bucket_payload = _hidden_bucket(hidden_state, bucket)
    if key in bucket_payload:
        return _clip01(bucket_payload[key])
    return _clip01(hidden_state.get(key, default))


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
    feature_lower: list[float]
    feature_upper: list[float]

    def predict_proxy_state(self, context: TargetEEGContext) -> EEGProxyState:
        hidden_state = context.hidden_state or {}
        observables = context.observable_signals or {}

        knowledge_state = _hidden_bucket(hidden_state, "knowledge_state")
        mastery = _clip01(
            knowledge_state.get("concept_mastery", {}).get(
                context.concept_id,
                hidden_state.get("concept_mastery", {}).get(context.concept_id, 0.5),
            )
        )
        confusion = _clip01(observables.get("confusion_score", 0.5))
        fatigue = _hidden_value(hidden_state, bucket="neuro_state", key="fatigue", default=observables.get("fatigue", 0.3))
        attention = _hidden_value(hidden_state, bucket="neuro_state", key="attention", default=observables.get("attention", 0.6))
        engagement = _clip01(observables.get("engagement_score", _hidden_value(hidden_state, bucket="neuro_state", key="engagement", default=0.6)))
        confidence = _clip01(observables.get("confidence", _hidden_value(hidden_state, bucket="knowledge_state", key="confidence", default=0.5)))
        workload = _hidden_value(hidden_state, bucket="neuro_state", key="workload", default=0.35)
        vigilance = _hidden_value(hidden_state, bucket="neuro_state", key="vigilance", default=0.6)
        stress = _hidden_value(hidden_state, bucket="neuro_state", key="stress", default=0.25)

        semantic_friction = _clip01(observables.get("semantic_friction", observables.get("comprehension_difficulty", 0.5)))
        workload = _clip01(
            0.42 * workload
            + 0.18 * confusion
            + 0.14 * fatigue
            + 0.16 * semantic_friction
            + 0.06 * (1.0 - mastery)
            + 0.08 * stress
            + 0.06 * (1.0 - attention)
            + 0.05 * (1.0 - confidence)
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
        design = np.array(
            [
                1.0,
                proxy.workload,
                proxy.workload ** 2,
            ],
            dtype=np.float64,
        )
        coefficients = np.array(self.coefficients, dtype=np.float64)
        raw_prediction = design @ coefficients
        clipped = []
        for index, value in enumerate(raw_prediction.tolist()):
            lower = self.feature_lower[index]
            upper = self.feature_upper[index]
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
            rows.append([1.0, workload, workload ** 2])
            targets.append(window.features)

    if not rows:
        raise ValueError("cannot fit workload mapper: no windows with workload_rating")

    x = np.array(rows, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    xtx = x.T @ x
    ridge = ridge_alpha * np.eye(xtx.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(xtx + ridge, x.T @ y)
    # The absolute alpha asymmetry feature has a very large raw scale and is
    # not stable enough to use as a forward retrieval target in the current
    # STEW-only workload mapper, so keep it at its empirical center.
    coefficients[:, 5] = 0.0
    feature_lower = np.quantile(y, 0.01, axis=0).tolist()
    feature_upper = np.quantile(y, 0.99, axis=0).tolist()

    return STEWWorkloadFeatureMapper(
        feature_names=list(feature_index.feature_names),
        basis_names=["bias", "workload", "workload_sq"],
        coefficients=coefficients.tolist(),
        feature_lower=[float(value) for value in feature_lower],
        feature_upper=[float(value) for value in feature_upper],
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
        feature_lower=[float(value) for value in payload["feature_lower"]],
        feature_upper=[float(value) for value in payload["feature_upper"]],
    )
