from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_benchmark_config
from .align_nback_windows import build_nback_windows
from .datasets import build_benchmark_samples, build_feature_matrices
from .evaluate import evaluate_feature_family


ABLATIONS = (
    {"name": "behavior_logreg", "family": "behavior_only", "model_family": "logistic_regression"},
    {"name": "eeg_logreg", "family": "eeg_only", "model_family": "logistic_regression"},
    {"name": "fused_logreg", "family": "fused", "model_family": "logistic_regression"},
    {"name": "fused_logreg_no_signal_mean", "family": "fused", "model_family": "logistic_regression", "drop": {"signal_mean"}},
    {"name": "fused_logreg_no_eeg_moments", "family": "fused", "model_family": "logistic_regression", "drop": {"signal_mean", "signal_variance_proxy"}},
    {"name": "fused_random_forest", "family": "fused", "model_family": "random_forest"},
    {"name": "fused_random_forest_no_signal_mean", "family": "fused", "model_family": "random_forest", "drop": {"signal_mean"}},
    {"name": "fused_mlp", "family": "fused", "model_family": "mlp"},
    {"name": "eeg_random_forest", "family": "eeg_only", "model_family": "random_forest"},
    {"name": "behavior_random_forest", "family": "behavior_only", "model_family": "random_forest"},
)


def run_fusion_ablation(config_path: str | Path) -> dict[str, object]:
    config = load_benchmark_config(config_path)
    windows = build_nback_windows(config)
    samples = build_benchmark_samples(windows, config)
    matrices = build_feature_matrices(samples, config)

    results = {}
    for spec in ABLATIONS:
        matrix = matrices[spec["family"]]
        X, feature_names = _apply_drop(
            X=matrix["X"],
            feature_names=matrix["feature_names"],
            drop=set(spec.get("drop", set())),
        )
        evaluation = evaluate_feature_family(
            feature_family=spec["name"],
            X=X,
            y=matrix["y"],
            groups=matrix["groups"],
            model_family=spec["model_family"],
        )
        results[spec["name"]] = {
            "source_family": spec["family"],
            "model_family": spec["model_family"],
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "mean_balanced_accuracy": evaluation.mean_balanced_accuracy,
            "mean_macro_f1": evaluation.mean_macro_f1,
            "per_fold_balanced_accuracy": list(evaluation.per_fold_balanced_accuracy),
            "per_fold_macro_f1": list(evaluation.per_fold_macro_f1),
        }

    output_path = config.output_dir / "fusion_ablation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return {"output_path": str(output_path), "ablation_count": len(results)}


def _apply_drop(
    *,
    X: list[list[float]],
    feature_names: list[str],
    drop: set[str],
) -> tuple[list[list[float]], list[str]]:
    if not drop:
        return X, list(feature_names)
    keep_indices = [idx for idx, name in enumerate(feature_names) if name not in drop]
    keep_names = [feature_names[idx] for idx in keep_indices]
    reduced = np.asarray(X, dtype=float)[:, keep_indices].tolist()
    return reduced, keep_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fusion ablations for the COG-BCI workload benchmark.")
    parser.add_argument(
        "--config",
        default="/Users/anish/PERSONAL/emotiv_learn/neuro_validate/configs/cog_bci_nback_workload.yaml",
    )
    args = parser.parse_args()
    print(run_fusion_ablation(args.config))


if __name__ == "__main__":
    main()
