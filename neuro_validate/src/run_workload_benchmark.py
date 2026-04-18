from __future__ import annotations

import argparse
import json
from pathlib import Path

from .align_nback_windows import build_nback_windows
from .config import load_benchmark_config
from .datasets import build_benchmark_samples, build_feature_matrices
from .evaluate import evaluate_feature_family
from .plots import write_metrics_json, write_placeholder_plot, write_summary_csv


def run_workload_benchmark(config_path: str | Path) -> dict[str, object]:
    config = load_benchmark_config(config_path)
    windows = build_nback_windows(config)
    samples = build_benchmark_samples(windows, config)
    matrices = build_feature_matrices(samples, config)

    evaluations = []
    for feature_family, payload in matrices.items():
        evaluations.append(
            evaluate_feature_family(
                feature_family=feature_family,
                X=payload["X"],
                y=payload["y"],
                groups=payload["groups"],
                model_family=config.model_family,
            )
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_metrics_json(evaluations, config.output_dir / "metrics.json")
    write_summary_csv(evaluations, config.output_dir / "summary.csv")
    write_placeholder_plot(evaluations, config.output_dir / "benchmark_bar.txt")
    _write_sample_preview(samples, config.output_dir / "sample_windows_preview.json")

    return {
        "config_path": str(config_path),
        "sample_count": len(samples),
        "feature_families": [evaluation.feature_family for evaluation in evaluations],
        "output_dir": str(config.output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-data EEG workload benchmark scaffold.")
    parser.add_argument(
        "--config",
        default="/Users/anish/PERSONAL/emotiv_learn/neuro_validate/configs/cog_bci_nback_workload.yaml",
    )
    args = parser.parse_args()
    result = run_workload_benchmark(args.config)
    print(result)


def _write_sample_preview(samples, output_path: Path, limit: int = 12) -> None:
    payload = []
    for sample in samples[:limit]:
        payload.append(
            {
                "subject_id": sample.subject_id,
                "session_id": sample.session_id,
                "task_name": sample.task_name,
                "workload_label": sample.workload_label,
                "window_start_s": sample.window_start_s,
                "window_end_s": sample.window_end_s,
                "behavior_features": sample.behavior_features,
                "eeg_feature_keys": list(sample.eeg_features.keys()),
            }
        )
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
