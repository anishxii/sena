from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


NBACK_CONDITION_TO_DIFFICULTY = {
    "ZeroBack": 0,
    "OneBack": 1,
    "TwoBack": 2,
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
