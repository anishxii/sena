from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from pathlib import Path
import zipfile

import scipy.io

from .windows import ExperimentWindow


NBACK_CONDITION_TO_DIFFICULTY = {
    "ZeroBack": 0,
    "OneBack": 1,
    "TwoBack": 2,
}

NBACK_CONDITION_TO_SET_FILE = {
    "ZeroBack": "zeroBACK.set",
    "OneBack": "oneBACK.set",
    "TwoBack": "twoBACK.set",
}

NBACK_TRIGGER_FAMILIES = {
    "ZeroBack": {
        "stimulus": {"6021", "6022", "6023"},
        "correct": {"6032"},
        "incorrect": {"6031", "6033"},
    },
    "OneBack": {
        "stimulus": {"6121", "6122", "6123"},
        "correct": {"6132"},
        "incorrect": {"6131", "6133"},
    },
    "TwoBack": {
        "stimulus": {"6221", "6222", "6223"},
        "correct": {"6232"},
        "incorrect": {"6231", "6233"},
    },
}


@dataclass(frozen=True)
class RsmeScore:
    subject_id: str
    session_id: str
    condition: str
    score: float
    normalized_score: float


@dataclass(frozen=True)
class KssScore:
    subject_id: str
    session_id: str
    condition: str
    score: float
    normalized_score: float


@dataclass(frozen=True)
class TriggerCode:
    code: str
    content: str


@dataclass(frozen=True)
class NBackConditionLabel:
    subject_id: str
    session_id: str
    condition: str
    difficulty_level: int
    workload_estimate: float
    kss_beginning: float | None
    kss_end: float | None


@dataclass(frozen=True)
class NBackRecordingSummary:
    subject_id: str
    session_id: str
    condition: str
    difficulty_level: int
    workload_estimate: float
    response_accuracy: float
    mean_response_time_s: float | None
    response_count: int
    stimulus_count: int
    kss_beginning: float | None
    kss_end: float | None


def load_rsme_scores(path: str | Path) -> list[RsmeScore]:
    rows = _read_csv_rows(path)
    scores: list[RsmeScore] = []
    for row in rows:
        condition = row["condition"]
        score = float(row["Score"])
        scores.append(
            RsmeScore(
                subject_id=_subject_id(row["sbj"]),
                session_id=str(row["Session"]),
                condition=condition,
                score=score,
                normalized_score=_normalize_rsme(score),
            )
        )
    return scores


def load_kss_scores(path: str | Path) -> list[KssScore]:
    rows = _read_csv_rows(path)
    scores: list[KssScore] = []
    for row in rows:
        score = float(row["score"])
        scores.append(
            KssScore(
                subject_id=_subject_id(row["sbj"]),
                session_id=str(row["sess"]),
                condition=row["Condition"],
                score=score,
                normalized_score=_normalize_kss(score),
            )
        )
    return scores


def load_trigger_codes(path: str | Path) -> list[TriggerCode]:
    rows = _read_csv_rows(path)
    return [TriggerCode(code=str(row["code"]), content=row["content"]) for row in rows]


def build_nback_condition_labels(
    rsme_scores: list[RsmeScore],
    kss_scores: list[KssScore],
) -> list[NBackConditionLabel]:
    kss_by_subject_session: dict[tuple[str, str], dict[str, float]] = {}
    for score in kss_scores:
        kss_by_subject_session.setdefault((score.subject_id, score.session_id), {})[score.condition] = score.normalized_score

    labels: list[NBackConditionLabel] = []
    for score in rsme_scores:
        if score.condition not in NBACK_CONDITION_TO_DIFFICULTY:
            continue
        kss = kss_by_subject_session.get((score.subject_id, score.session_id), {})
        labels.append(
            NBackConditionLabel(
                subject_id=score.subject_id,
                session_id=score.session_id,
                condition=score.condition,
                difficulty_level=NBACK_CONDITION_TO_DIFFICULTY[score.condition],
                workload_estimate=score.normalized_score,
                kss_beginning=kss.get("beginning"),
                kss_end=kss.get("end"),
            )
        )
    return labels


def load_nback_condition_labels(cog_bci_dir: str | Path) -> list[NBackConditionLabel]:
    root = Path(cog_bci_dir)
    rsme_scores = load_rsme_scores(root / "RSME.txt")
    kss_scores = load_kss_scores(root / "KSS.txt")
    return build_nback_condition_labels(rsme_scores=rsme_scores, kss_scores=kss_scores)


def build_subject_nback_recording_summaries(
    cog_bci_dir: str | Path,
    subject_id: str,
) -> list[NBackRecordingSummary]:
    root = Path(cog_bci_dir)
    labels = {
        (label.subject_id, label.session_id, label.condition): label
        for label in load_nback_condition_labels(root)
        if label.subject_id == subject_id
    }
    zip_path = root / f"{subject_id}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"missing COG-BCI subject archive: {zip_path}")

    summaries: list[NBackRecordingSummary] = []
    with zipfile.ZipFile(zip_path) as archive:
        for session_id in ["1", "2", "3"]:
            for condition, set_file in NBACK_CONDITION_TO_SET_FILE.items():
                label = labels.get((subject_id, session_id, condition))
                if label is None:
                    continue
                archive_name = f"{subject_id}/ses-S{session_id}/eeg/{set_file}"
                if archive_name not in archive.namelist():
                    continue
                set_payload = archive.read(archive_name)
                event_summary = summarize_nback_set_events(set_payload, condition=condition)
                summaries.append(
                    NBackRecordingSummary(
                        subject_id=subject_id,
                        session_id=session_id,
                        condition=condition,
                        difficulty_level=label.difficulty_level,
                        workload_estimate=label.workload_estimate,
                        response_accuracy=event_summary["response_accuracy"],
                        mean_response_time_s=event_summary["mean_response_time_s"],
                        response_count=int(event_summary["response_count"]),
                        stimulus_count=int(event_summary["stimulus_count"]),
                        kss_beginning=label.kss_beginning,
                        kss_end=label.kss_end,
                    )
                )
    return summaries


def build_subject_nback_windows(cog_bci_dir: str | Path, subject_id: str) -> list[ExperimentWindow]:
    summaries = build_subject_nback_recording_summaries(cog_bci_dir=cog_bci_dir, subject_id=subject_id)
    rt_values = [summary.mean_response_time_s for summary in summaries if summary.mean_response_time_s is not None]
    rt_min = min(rt_values) if rt_values else 0.0
    rt_max = max(rt_values) if rt_values else 1.0
    rt_span = max(rt_max - rt_min, 1e-6)

    windows: list[ExperimentWindow] = []
    for summary in summaries:
        if summary.mean_response_time_s is None:
            rt_percentile = 0.5
        else:
            rt_percentile = (summary.mean_response_time_s - rt_min) / rt_span
        windows.append(
            ExperimentWindow(
                window_id=f"{summary.subject_id}_ses{summary.session_id}_{summary.condition}",
                subject_id=summary.subject_id,
                session_id=summary.session_id,
                task="n_back",
                difficulty_level=summary.difficulty_level,
                workload_estimate=summary.workload_estimate,
                rolling_accuracy=summary.response_accuracy,
                rolling_rt_percentile=_clip01(rt_percentile),
                lapse_rate=_clip01(1.0 - summary.response_accuracy),
                eeg_features=[],
                metadata={
                    "source": "cog_bci_condition_summary",
                    "condition": summary.condition,
                    "response_count": summary.response_count,
                    "stimulus_count": summary.stimulus_count,
                    "mean_response_time_s": -1.0
                    if summary.mean_response_time_s is None
                    else round(summary.mean_response_time_s, 6),
                    "kss_beginning": -1.0 if summary.kss_beginning is None else summary.kss_beginning,
                    "kss_end": -1.0 if summary.kss_end is None else summary.kss_end,
                },
            )
        )
    return windows


def summarize_nback_set_events(set_payload: bytes, condition: str) -> dict[str, float | int | None]:
    mat = scipy.io.loadmat(io.BytesIO(set_payload), squeeze_me=True, struct_as_record=False)
    srate = float(mat["srate"])
    events = [
        {"type": str(event.type), "latency": float(event.latency)}
        for event in mat["event"]
        if hasattr(event, "type") and str(event.type) != "boundary"
    ]
    return summarize_nback_events(events=events, condition=condition, srate=srate)


def summarize_nback_events(
    events: list[dict[str, float | str]],
    condition: str,
    srate: float,
) -> dict[str, float | int | None]:
    triggers = NBACK_TRIGGER_FAMILIES[condition]
    stimulus_codes = triggers["stimulus"]
    correct_codes = triggers["correct"]
    incorrect_codes = triggers["incorrect"]

    stimulus_latencies: list[float] = []
    response_times: list[float] = []
    correct_count = 0
    incorrect_count = 0
    last_stimulus_latency: float | None = None

    for event in sorted(events, key=lambda item: float(item["latency"])):
        event_type = str(event["type"])
        latency = float(event["latency"])
        if event_type in stimulus_codes:
            stimulus_latencies.append(latency)
            last_stimulus_latency = latency
            continue
        if event_type in correct_codes or event_type in incorrect_codes:
            if event_type in correct_codes:
                correct_count += 1
            else:
                incorrect_count += 1
            if last_stimulus_latency is not None:
                response_times.append((latency - last_stimulus_latency) / srate)
                last_stimulus_latency = None

    response_count = correct_count + incorrect_count
    response_accuracy = correct_count / response_count if response_count else 0.0
    mean_response_time_s = sum(response_times) / len(response_times) if response_times else None
    return {
        "stimulus_count": len(stimulus_latencies),
        "response_count": response_count,
        "response_accuracy": response_accuracy,
        "mean_response_time_s": mean_response_time_s,
    }


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _subject_id(value: str) -> str:
    return f"sub-{int(value):02d}"


def _normalize_rsme(value: float) -> float:
    # The RSME scale is commonly treated as 0-150; COG-BCI values in practice
    # occupy a smaller range but keeping the published scale avoids leakage.
    return _clip01(value / 150.0)


def _normalize_kss(value: float) -> float:
    # KSS is a 1-9 sleepiness scale. Some rows contain sentinel negatives for
    # missing values, so clamp after normalizing.
    return _clip01((value - 1.0) / 8.0)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
