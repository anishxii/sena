from __future__ import annotations

import json
from pathlib import Path

from .schema import ModelEvaluation


def write_summary_csv(evaluations: list[ModelEvaluation], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["feature_family,mean_balanced_accuracy,mean_macro_f1,sample_count,subject_count"]
    for evaluation in evaluations:
        lines.append(
            ",".join(
                [
                    evaluation.feature_family,
                    f"{evaluation.mean_balanced_accuracy:.6f}",
                    f"{evaluation.mean_macro_f1:.6f}",
                    str(evaluation.sample_count),
                    str(evaluation.subject_count),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_json(evaluations: list[ModelEvaluation], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        evaluation.feature_family: {
            "mean_balanced_accuracy": evaluation.mean_balanced_accuracy,
            "mean_macro_f1": evaluation.mean_macro_f1,
            "per_fold_balanced_accuracy": list(evaluation.per_fold_balanced_accuracy),
            "per_fold_macro_f1": list(evaluation.per_fold_macro_f1),
            "sample_count": evaluation.sample_count,
            "subject_count": evaluation.subject_count,
        }
        for evaluation in evaluations
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_placeholder_plot(evaluations: list[ModelEvaluation], output_path: str | Path) -> None:
    """Write a text placeholder until the matplotlib figure is filled in."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Feature family performance summary"]
    for evaluation in evaluations:
        lines.append(
            f"- {evaluation.feature_family}: "
            f"balanced_accuracy={evaluation.mean_balanced_accuracy:.3f}, "
            f"macro_f1={evaluation.mean_macro_f1:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

