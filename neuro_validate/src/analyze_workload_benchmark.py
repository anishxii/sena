from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

from .align_nback_windows import build_nback_windows
from .config import load_benchmark_config
from .datasets import build_benchmark_samples, build_feature_matrices
from .evaluate import _leave_one_group_out_splits
from .models import build_classifier


def analyze_workload_benchmark(config_path: str | Path) -> dict[str, object]:
    config = load_benchmark_config(config_path)
    windows = build_nback_windows(config)
    samples = build_benchmark_samples(windows, config)
    matrices = build_feature_matrices(samples, config)

    analyses = {
        "behavior_random_forest": _analyze_family(
            X=np.asarray(matrices["behavior_only"]["X"], dtype=float),
            y=np.asarray(matrices["behavior_only"]["y"], dtype=int),
            groups=matrices["behavior_only"]["groups"],
            feature_names=matrices["behavior_only"]["feature_names"],
            model_family="random_forest",
        ),
        "eeg_random_forest": _analyze_family(
            X=np.asarray(matrices["eeg_only"]["X"], dtype=float),
            y=np.asarray(matrices["eeg_only"]["y"], dtype=int),
            groups=matrices["eeg_only"]["groups"],
            feature_names=matrices["eeg_only"]["feature_names"],
            model_family="random_forest",
        ),
        "fused_random_forest": _analyze_family(
            X=np.asarray(matrices["fused"]["X"], dtype=float),
            y=np.asarray(matrices["fused"]["y"], dtype=int),
            groups=matrices["fused"]["groups"],
            feature_names=matrices["fused"]["feature_names"],
            model_family="random_forest",
        ),
        "fused_random_forest_no_signal_mean": _analyze_family(
            X=np.asarray(matrices["fused"]["X"], dtype=float),
            y=np.asarray(matrices["fused"]["y"], dtype=int),
            groups=matrices["fused"]["groups"],
            feature_names=matrices["fused"]["feature_names"],
            model_family="random_forest",
            drop_features={"signal_mean"},
        ),
    }

    output_path = config.output_dir / "analysis_report.json"
    output_path.write_text(json.dumps(analyses, indent=2), encoding="utf-8")
    return {"output_path": str(output_path), "analysis_count": len(analyses)}


def _analyze_family(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str],
    feature_names: list[str],
    model_family: str,
    drop_features: set[str] | None = None,
) -> dict[str, object]:
    if drop_features:
        keep_indices = [idx for idx, name in enumerate(feature_names) if name not in drop_features]
        X = X[:, keep_indices]
        feature_names = [feature_names[idx] for idx in keep_indices]

    folds = []
    importances = []
    for train_indices, test_indices in _leave_one_group_out_splits(groups):
        held_out = groups[test_indices[0]]
        model = build_classifier(model_family)
        model.fit(X[train_indices], y[train_indices])
        preds = model.predict(X[test_indices])
        truth = y[test_indices]
        folds.append(
            {
                "held_out_subject": held_out,
                "confusion_matrix": confusion_matrix(truth, preds, labels=[0, 1, 2]).tolist(),
                "truth_counts": {
                    "0": int(np.sum(truth == 0)),
                    "1": int(np.sum(truth == 1)),
                    "2": int(np.sum(truth == 2)),
                },
                "pred_counts": {
                    "0": int(np.sum(preds == 0)),
                    "1": int(np.sum(preds == 1)),
                    "2": int(np.sum(preds == 2)),
                },
            }
        )
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)

    mean_importance = np.mean(np.vstack(importances), axis=0) if importances else np.zeros(len(feature_names))
    ranked = [
        {"feature": name, "importance": float(importance)}
        for name, importance in sorted(zip(feature_names, mean_importance), key=lambda item: item[1], reverse=True)
    ]
    eeg_importance = sum(
        item["importance"]
        for item in ranked
        if item["feature"].startswith(("theta", "alpha", "beta", "gamma")) or item["feature"] in {"theta_alpha_ratio", "signal_variance_proxy"}
    )
    behavior_importance = 1.0 - eeg_importance if ranked else 0.0

    return {
        "feature_names": feature_names,
        "folds": folds,
        "feature_importance_ranked": ranked,
        "eeg_importance_sum": float(eeg_importance),
        "behavior_importance_sum": float(behavior_importance),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze held-out subject performance for the COG-BCI workload benchmark.")
    parser.add_argument(
        "--config",
        default="/Users/anish/PERSONAL/emotiv_learn/neuro_validate/configs/cog_bci_nback_workload.yaml",
    )
    args = parser.parse_args()
    print(analyze_workload_benchmark(args.config))


if __name__ == "__main__":
    main()
