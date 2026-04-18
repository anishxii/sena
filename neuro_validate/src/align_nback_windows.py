from __future__ import annotations

from math import ceil
from pathlib import Path

import pandas as pd

from .ingest_cog_bci import (
    NBACK_WORKLOAD_LABELS,
    discover_nback_task_records,
    extract_event_rows,
    load_eeglab_header,
    load_rsme_scores,
)
from .schema import BenchmarkConfig, WindowedTrial


def build_nback_windows(config: BenchmarkConfig) -> list[WindowedTrial]:
    """Create labeled workload windows from COG-BCI N-Back recordings.

    The implementation here is intentionally a scaffold. The full version should:

    1. iterate subject/session/task files from the BIDS layout
    2. align EEG samples to N-Back task blocks or trials
    3. assign workload labels from task condition (0/1/2-back)
    4. join available non-neural behavioral/task outputs
    """

    rsme = load_rsme_scores(config.dataset_root)
    task_records = discover_nback_task_records(config.dataset_root)
    windows: list[WindowedTrial] = []
    for task_record in task_records:
        header = load_eeglab_header(task_record.eeg_set_path)
        srate = int(header["srate"])
        window_length_samples = int(round(config.window_length_s * srate))
        step_samples = int(round(config.window_step_s * srate))
        if window_length_samples <= 0 or step_samples <= 0:
            raise ValueError("window length and step must be positive")
        total_points = int(header["pnts"])
        if total_points < window_length_samples:
            continue
        total_windows = ceil((total_points - window_length_samples) / step_samples) + 1
        event_summary = _build_nback_event_summary(task_record.eeg_set_path, srate)
        behavior_table = _load_behavior_table_csv(
            dataset_root=config.dataset_root,
            subject_id=task_record.subject_id,
            session_id=task_record.session_id,
            task_condition=task_record.task_condition,
        )
        rsme_score = _lookup_rsme_score(
            rsme=rsme,
            subject_id=task_record.subject_id,
            session_id=task_record.session_id,
            condition_name=task_record.task_condition,
        )
        workload_label = NBACK_WORKLOAD_LABELS[task_record.task_condition]
        for window_index, start_sample in enumerate(range(0, total_points - window_length_samples + 1, step_samples)):
            end_sample = start_sample + window_length_samples
            window_start_s = start_sample / srate
            window_end_s = end_sample / srate
            progress_norm = (
                window_index / max(total_windows - 1, 1)
                if total_windows > 1
                else 0.0
            )
            behavior_rollup = _behavior_rollup_for_progress(
                behavior_table=behavior_table,
                progress_norm=progress_norm,
            )
            windows.append(
                WindowedTrial(
                    subject_id=task_record.subject_id,
                    session_id=task_record.session_id,
                    task_name=task_record.task_condition,
                    workload_label=workload_label,
                    window_start_s=window_start_s,
                    window_end_s=window_end_s,
                    eeg_samples=None,
                    behavior_payload={
                        "trial_progress_norm": progress_norm,
                        "rolling_accuracy": _rolling_accuracy(event_summary, window_end_s),
                        "rolling_rt_percentile": _rolling_rt_percentile(event_summary, window_end_s),
                        "lapse_rate": _rolling_lapse_rate(event_summary, window_end_s),
                        "behavior_row_progress_norm": behavior_rollup["behavior_row_progress_norm"],
                        "behavior_correct_rate": behavior_rollup["behavior_correct_rate"],
                        "behavior_hit_rate": behavior_rollup["behavior_hit_rate"],
                        "behavior_miss_rate": behavior_rollup["behavior_miss_rate"],
                        "behavior_error_rate": behavior_rollup["behavior_error_rate"],
                        "behavior_mistake_rate": behavior_rollup["behavior_mistake_rate"],
                        "behavior_outlier_rate": behavior_rollup["behavior_outlier_rate"],
                        "behavior_rt_mean_norm": behavior_rollup["behavior_rt_mean_norm"],
                        "behavior_rt_median_norm": behavior_rollup["behavior_rt_median_norm"],
                        "session_index_norm": (_session_index(task_record.session_id) - 1) / 2.0,
                        "subjective_workload_rsme": rsme_score,
                        "behavioral_mat_path": str(task_record.behavioral_mat_path) if task_record.behavioral_mat_path else None,
                        "eeg_set_path": str(task_record.eeg_set_path),
                        "eeg_fdt_path": str(task_record.eeg_fdt_path),
                        "nbchan": int(header["nbchan"]),
                        "sample_start": int(start_sample),
                        "sample_end": int(end_sample),
                        "total_points": total_points,
                        "srate": srate,
                    },
                )
            )
    return windows


def _lookup_rsme_score(*, rsme, subject_id: str, session_id: str, condition_name: str) -> float:
    subject_numeric = int(subject_id)
    session_numeric = _session_index(session_id)
    rows = rsme[
        (rsme["sbj"] == subject_numeric)
        & (rsme["Session"] == session_numeric)
        & (rsme["condition"] == condition_name)
    ]
    if rows.empty:
        return 0.0
    score = float(rows.iloc[0]["Score"])
    return max(0.0, min(score / 100.0, 1.5))


def _session_index(session_id: str) -> int:
    return int(session_id.replace("ses-", ""))


def _build_nback_event_summary(set_path: str, srate: int) -> list[dict[str, float | bool]]:
    rows = extract_event_rows(set_path)
    summary = []
    for index, row in enumerate(rows):
        event_type = str(row["type"])
        if not event_type.endswith("22"):
            continue
        latency_s = float(row["latency"]) / srate
        responded = False
        correct = False
        rt_s = None
        for look_ahead in rows[index + 1:index + 6]:
            next_type = str(look_ahead["type"])
            if next_type.endswith("32"):
                responded = True
                correct = True
                rt_s = (float(look_ahead["latency"]) - float(row["latency"])) / srate
                break
            if next_type.endswith("31") or next_type.endswith("33"):
                responded = True
                correct = False
                rt_s = (float(look_ahead["latency"]) - float(row["latency"])) / srate
                break
            if next_type.endswith("21") or next_type.endswith("22"):
                break
        summary.append(
            {
                "target_time_s": latency_s,
                "responded": responded,
                "correct": correct,
                "missed": not responded,
                "rt_s": float(rt_s) if rt_s is not None else None,
            }
        )
    return summary


def _rolling_accuracy(summary: list[dict[str, float | bool]], window_end_s: float) -> float:
    eligible = [item for item in summary if float(item["target_time_s"]) <= window_end_s]
    if not eligible:
        return 0.0
    correct = sum(1 for item in eligible if bool(item["correct"]))
    return correct / len(eligible)


def _rolling_lapse_rate(summary: list[dict[str, float | bool]], window_end_s: float) -> float:
    eligible = [item for item in summary if float(item["target_time_s"]) <= window_end_s]
    if not eligible:
        return 0.0
    missed = sum(1 for item in eligible if bool(item["missed"]))
    return missed / len(eligible)


def _rolling_rt_percentile(summary: list[dict[str, float | bool]], window_end_s: float) -> float:
    eligible = [float(item["rt_s"]) for item in summary if float(item["target_time_s"]) <= window_end_s and item["rt_s"] is not None]
    if not eligible:
        return 0.0
    current = eligible[-1]
    rank = sum(1 for value in eligible if value <= current)
    return rank / len(eligible)


def _load_behavior_table_csv(
    *,
    dataset_root: str | Path,
    subject_id: str,
    session_id: str,
    task_condition: str,
) -> pd.DataFrame | None:
    csv_root = Path(dataset_root) / "behavior_csv_flat"
    if not csv_root.exists():
        return None
    task_stem = {
        "ZeroBack": "0-Back",
        "OneBack": "1-Back",
        "TwoBack": "2-Back",
    }.get(task_condition)
    if task_stem is None:
        return None
    session_suffix = int(session_id.replace("ses-", ""))
    filename = f"sub-{subject_id}__ses-S{session_suffix}__{task_stem}.csv"
    path = csv_root / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


def _behavior_rollup_for_progress(
    *,
    behavior_table: pd.DataFrame | None,
    progress_norm: float,
) -> dict[str, float]:
    if behavior_table is None or behavior_table.empty:
        return {
            "behavior_row_progress_norm": 0.0,
            "behavior_correct_rate": 0.0,
            "behavior_hit_rate": 0.0,
            "behavior_miss_rate": 0.0,
            "behavior_error_rate": 0.0,
            "behavior_mistake_rate": 0.0,
            "behavior_outlier_rate": 0.0,
            "behavior_rt_mean_norm": 0.0,
            "behavior_rt_median_norm": 0.0,
        }

    cutoff = max(1, min(len(behavior_table), int(round(progress_norm * (len(behavior_table) - 1))) + 1))
    rows = behavior_table.iloc[:cutoff]
    rt_series = rows["rt"].dropna() if "rt" in rows else pd.Series(dtype=float)
    rt_mean = float(rt_series.mean()) if not rt_series.empty else 0.0
    rt_median = float(rt_series.median()) if not rt_series.empty else 0.0

    return {
        "behavior_row_progress_norm": cutoff / max(len(behavior_table), 1),
        "behavior_correct_rate": float(rows["correct"].mean()) if "correct" in rows else 0.0,
        "behavior_hit_rate": float(rows["hittrials"].mean()) if "hittrials" in rows else 0.0,
        "behavior_miss_rate": float(rows["miss"].mean()) if "miss" in rows else 0.0,
        "behavior_error_rate": float(rows["error"].mean()) if "error" in rows else 0.0,
        "behavior_mistake_rate": float(rows["mistake"].mean()) if "mistake" in rows else 0.0,
        "behavior_outlier_rate": float(rows["outlier"].mean()) if "outlier" in rows else 0.0,
        "behavior_rt_mean_norm": min(rt_mean / 1500.0, 1.0),
        "behavior_rt_median_norm": min(rt_median / 1500.0, 1.0),
    }
