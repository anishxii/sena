from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .cog_bci_ingest import COGBCINBackWindow


@dataclass(frozen=True)
class COGBCIProxyRegressor:
    feature_names: list[str]
    target_names: list[str]
    feature_mean: list[float]
    feature_std: list[float]
    target_mean: list[float]
    coefficients: list[list[float]]
    intercept: list[float]

    def predict(self, eeg_features: list[float]) -> dict[str, float]:
        x = (np.array(eeg_features, dtype=np.float64) - np.array(self.feature_mean)) / np.array(self.feature_std)
        weights = np.array(self.coefficients, dtype=np.float64)
        intercept = np.array(self.intercept, dtype=np.float64)
        target_mean = np.array(self.target_mean, dtype=np.float64)
        prediction = x @ weights + intercept + target_mean
        clipped = prediction.copy()
        clipped[0] = np.clip(clipped[0], 0.0, 1.0)
        clipped[1] = np.clip(clipped[1], 0.0, 1.0)
        clipped[2] = np.clip(clipped[2], 0.0, 1.0)
        clipped[3] = np.clip(clipped[3], 0.0, 1.0)
        return {
            target_name: float(value)
            for target_name, value in zip(self.target_names, clipped.tolist(), strict=False)
        }


def fit_cog_bci_proxy_regressor(
    windows: list[COGBCINBackWindow],
    ridge_alpha: float = 1e-3,
) -> COGBCIProxyRegressor:
    if not windows:
        raise ValueError("cannot fit COG-BCI proxy regressor: no windows provided")

    x = np.array([window.eeg_features for window in windows], dtype=np.float64)
    y = np.array(
        [
            [
                window.workload_estimate,
                window.rolling_accuracy,
                window.rolling_rt_percentile,
                window.lapse_rate,
            ]
            for window in windows
        ],
        dtype=np.float64,
    )

    feature_mean = x.mean(axis=0)
    feature_std = np.maximum(x.std(axis=0), 1e-6)
    xz = (x - feature_mean) / feature_std

    target_mean = y.mean(axis=0)
    yz = y - target_mean

    xtx = xz.T @ xz
    ridge = ridge_alpha * np.eye(xtx.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(xtx + ridge, xz.T @ yz)
    intercept = np.zeros(y.shape[1], dtype=np.float64)

    return COGBCIProxyRegressor(
        feature_names=[
            "eeg_theta_mean",
            "eeg_alpha_mean",
            "eeg_beta_mean",
            "eeg_gamma_mean",
            "eeg_frontal_alpha_asymmetry",
            "eeg_frontal_alpha_asymmetry_abs",
            "eeg_frontal_theta_alpha_ratio_mean",
            "eeg_load_score",
        ],
        target_names=[
            "workload_estimate",
            "rolling_accuracy",
            "rolling_rt_percentile",
            "lapse_rate",
        ],
        feature_mean=[float(value) for value in feature_mean],
        feature_std=[float(value) for value in feature_std],
        target_mean=[float(value) for value in target_mean],
        coefficients=coefficients.tolist(),
        intercept=intercept.tolist(),
    )


def save_cog_bci_proxy_regressor(model: COGBCIProxyRegressor, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")


def load_cog_bci_proxy_regressor(path: str | Path) -> COGBCIProxyRegressor:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return COGBCIProxyRegressor(
        feature_names=list(payload["feature_names"]),
        target_names=list(payload["target_names"]),
        feature_mean=[float(value) for value in payload["feature_mean"]],
        feature_std=[float(value) for value in payload["feature_std"]],
        target_mean=[float(value) for value in payload["target_mean"]],
        coefficients=[[float(value) for value in row] for row in payload["coefficients"]],
        intercept=[float(value) for value in payload["intercept"]],
    )
