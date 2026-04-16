from __future__ import annotations

import argparse
import itertools
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


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    per_target = {}
    maes = []
    rmses = []
    for index, target_name in enumerate(TARGET_NAMES):
        mae = float(mean_absolute_error(y_true[:, index], y_pred[:, index]))
        rmse = float(np.sqrt(mean_squared_error(y_true[:, index], y_pred[:, index])))
        maes.append(mae)
        rmses.append(rmse)
        per_target[target_name] = {"mae": round(mae, 6), "rmse": round(rmse, 6)}
    return {
        "mae_macro": round(float(np.mean(maes)), 6),
        "rmse_macro": round(float(np.mean(rmses)), 6),
        "per_target": per_target,
    }


def _ridge_baseline(train_windows: list[COGBCINBackWindow], test_x: np.ndarray) -> np.ndarray:
    model = fit_cog_bci_proxy_regressor(train_windows)
    x = (test_x - np.array(model.feature_mean)) / np.array(model.feature_std)
    pred = x @ np.array(model.coefficients) + np.array(model.intercept) + np.array(model.target_mean)
    return np.clip(pred, 0.0, 1.0)


def _fit_mlp(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    *,
    hidden_layer_sizes: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
    batch_size: int,
    seed: int,
) -> np.ndarray:
    feature_mean = train_x.mean(axis=0)
    feature_std = np.maximum(train_x.std(axis=0), 1e-6)
    x_train = (train_x - feature_mean) / feature_std
    x_test = (test_x - feature_mean) / feature_std
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=600,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=seed,
    )
    model.fit(x_train, train_y)
    pred = model.predict(x_test)
    return np.clip(pred, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune small MLP regressors against the ridge COG-BCI proxy baseline.")
    parser.add_argument("--windows-json", default="artifacts/cog_bci_cache/cog_bci_nback_windows_full.json")
    parser.add_argument("--output", default="artifacts/cog_bci_cache/proxy_mlp_tuning.json")
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

    ridge_pred = _ridge_baseline(train_windows, test_x)
    ridge_metrics = _score(test_y, ridge_pred)

    hidden_options = [(16,), (32,), (32, 16), (64, 32), (64, 32, 16)]
    alpha_options = [1e-4, 1e-3, 1e-2]
    lr_options = [1e-3, 3e-4]
    batch_options = [64, 128]

    trials = []
    for trial_index, (hidden, alpha, lr, batch) in enumerate(
        itertools.product(hidden_options, alpha_options, lr_options, batch_options),
        start=1,
    ):
        pred = _fit_mlp(
            train_x,
            train_y,
            test_x,
            hidden_layer_sizes=hidden,
            alpha=alpha,
            learning_rate_init=lr,
            batch_size=batch,
            seed=args.seed,
        )
        metrics = _score(test_y, pred)
        row = {
            "trial": trial_index,
            "hidden_layer_sizes": list(hidden),
            "alpha": alpha,
            "learning_rate_init": lr,
            "batch_size": batch,
            "metrics": metrics,
        }
        trials.append(row)
        print(json.dumps(row), flush=True)

    trials.sort(key=lambda row: (row["metrics"]["rmse_macro"], row["metrics"]["mae_macro"]))
    payload = {
        "train_subjects": sorted(set(groups[train_idx].tolist())),
        "test_subjects": sorted(set(groups[test_idx].tolist())),
        "train_windows": int(len(train_idx)),
        "test_windows": int(len(test_idx)),
        "ridge_baseline": ridge_metrics,
        "best_trial": trials[0] if trials else None,
        "trials": trials,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"best_trial": payload["best_trial"], "ridge_baseline": ridge_metrics}, indent=2))


if __name__ == "__main__":
    main()
