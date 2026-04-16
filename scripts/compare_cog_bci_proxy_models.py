from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.cog_bci_ingest import COGBCINBackWindow  # noqa: E402
from emotiv_learn.cog_bci_proxy_model import fit_cog_bci_proxy_regressor  # noqa: E402


TARGET_NAMES = [
    "workload_estimate",
    "rolling_accuracy",
    "rolling_rt_percentile",
    "lapse_rate",
]


def _load_windows(path: str | Path) -> list[COGBCINBackWindow]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [COGBCINBackWindow(**row) for row in payload]


def _dataset(windows: list[COGBCINBackWindow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    groups = np.array([window.subject_id for window in windows], dtype=object)
    return x, y, groups


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    per_target = {}
    maes = []
    rmses = []
    for index, target_name in enumerate(TARGET_NAMES):
        mae = float(mean_absolute_error(y_true[:, index], y_pred[:, index]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, index], y_pred[:, index])))
        maes.append(mae)
        rmses.append(rmse)
        per_target[target_name] = {
            "mae": round(mae, 6),
            "rmse": round(rmse, 6),
        }
    return {
        "mae_macro": round(float(np.mean(maes)), 6),
        "rmse_macro": round(float(np.mean(rmses)), 6),
        "per_target": per_target,
    }


def _fit_predict_ridge(
    train_windows: list[COGBCINBackWindow],
    test_x: np.ndarray,
) -> np.ndarray:
    model = fit_cog_bci_proxy_regressor(train_windows)
    x = (test_x - np.array(model.feature_mean)) / np.array(model.feature_std)
    w = np.array(model.coefficients)
    b = np.array(model.intercept)
    mu = np.array(model.target_mean)
    pred = x @ w + b + mu
    return np.clip(pred, 0.0, 1.0)


def _fit_predict_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    seed: int,
) -> np.ndarray:
    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6)
    x_train = (train_x - feature_mean) / feature_std
    x_test = (test_x - feature_mean) / feature_std

    model = MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=128,
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=seed,
    )
    model.fit(x_train, train_y)
    pred = model.predict(x_test)
    return np.clip(pred, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ridge vs small MLP on cached COG-BCI proxy-state prediction.")
    parser.add_argument("--windows-json", default="artifacts/cog_bci_cache/cog_bci_nback_windows_full.json")
    parser.add_argument("--output", default="artifacts/cog_bci_cache/proxy_model_comparison.json")
    parser.add_argument("--test-subjects", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    windows = _load_windows(args.windows_json)
    x, y, groups = _dataset(windows)

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_subjects, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x, y, groups=groups))

    train_windows = [windows[index] for index in train_idx]
    train_x = x[train_idx]
    train_y = y[train_idx]
    test_x = x[test_idx]
    test_y = y[test_idx]

    ridge_pred = _fit_predict_ridge(train_windows, test_x)
    mlp_pred = _fit_predict_mlp(train_x, train_y, test_x, seed=args.seed)

    test_subject_ids = sorted(set(groups[test_idx].tolist()))
    payload = {
        "train_subjects": sorted(set(groups[train_idx].tolist())),
        "test_subjects": test_subject_ids,
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "ridge": _metrics(test_y, ridge_pred),
        "mlp": _metrics(test_y, mlp_pred),
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
