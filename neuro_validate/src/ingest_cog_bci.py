from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from functools import lru_cache

import numpy as np
import pandas as pd
import scipy.io


@dataclass(frozen=True)
class CogBCISubjectRecord:
    subject_id: str
    root: Path


@dataclass(frozen=True)
class EEGTaskRecord:
    subject_id: str
    session_id: str
    session_root: Path
    task_condition: str
    eeg_set_path: Path
    eeg_fdt_path: Path
    behavioral_mat_path: Path | None


def discover_cog_bci_subjects(dataset_root: str | Path) -> list[CogBCISubjectRecord]:
    root = Path(dataset_root)
    if not root.exists():
        return []
    subjects = []
    for path in sorted(root.glob("sub-*")):
        if not path.is_dir():
            continue
        subject_root = _normalize_subject_root(path)
        subjects.append(CogBCISubjectRecord(subject_id=subject_root.name.replace("sub-", ""), root=subject_root))
    return subjects


def describe_dataset_layout(dataset_root: str | Path) -> dict[str, int]:
    subjects = discover_cog_bci_subjects(dataset_root)
    return {
        "subject_count": len(subjects),
    }


TASK_CODE_TO_CONDITION = {
    1: "MATB_easy",
    2: "MATB_med",
    3: "MATB_diff",
    4: "ZeroBack",
    5: "OneBack",
    6: "TwoBack",
    7: "PVT",
    8: "Flanker",
}

NBACK_WORKLOAD_LABELS = {
    "ZeroBack": 0,
    "OneBack": 1,
    "TwoBack": 2,
}


def load_rsme_scores(dataset_root: str | Path) -> pd.DataFrame:
    path = Path(dataset_root) / "RSME.txt"
    frame = pd.read_csv(path)
    frame["subject_id"] = frame["sbj"].map(_format_subject_id)
    frame["session_id"] = frame["Session"].map(_format_session_id)
    return frame


def load_kss_scores(dataset_root: str | Path) -> pd.DataFrame:
    path = Path(dataset_root) / "KSS.txt"
    frame = pd.read_csv(path)
    frame["subject_id"] = frame["sbj"].map(_format_subject_id)
    frame["session_id"] = frame["sess"].map(_format_session_id)
    return frame


def load_session_orders(dataset_root: str | Path) -> dict[tuple[str, str], list[str]]:
    path = Path(dataset_root) / "notebook.mat"
    notebook = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)["notebook"]
    session_orders: dict[tuple[str, str], list[str]] = {}
    for field_name in notebook._fieldnames:
        subject_struct = getattr(notebook, field_name)
        subject_id = _format_subject_id(int(re.sub(r"^SBJ_", "", field_name)))
        for session_field in subject_struct._fieldnames:
            session_struct = getattr(subject_struct, session_field)
            session_id = _format_session_id(int(re.sub(r"^SESS_", "", session_field)))
            raw_order = session_struct.Order.tolist() if hasattr(session_struct.Order, "tolist") else list(session_struct.Order)
            session_orders[(subject_id, session_id)] = [
                TASK_CODE_TO_CONDITION[int(code)]
                for code in raw_order
                if int(code) in TASK_CODE_TO_CONDITION
            ]
    return session_orders


def discover_nback_task_records(dataset_root: str | Path) -> list[EEGTaskRecord]:
    root = Path(dataset_root)
    task_records: list[EEGTaskRecord] = []
    eeg_name_map = {
        "ZeroBack": "zeroBACK",
        "OneBack": "oneBACK",
        "TwoBack": "twoBACK",
    }
    behavior_name_map = {
        "ZeroBack": "0-Back.mat",
        "OneBack": "1-Back.mat",
        "TwoBack": "2-Back.mat",
    }
    for subject_root in sorted(root.glob("sub-*")):
        if not subject_root.is_dir():
            continue
        subject_root = _normalize_subject_root(subject_root)
        subject_id = _format_subject_id(_extract_numeric_suffix(subject_root.name))
        for session_root in sorted(subject_root.glob("ses-*")):
            session_id = _format_session_id(_extract_numeric_suffix(session_root.name))
            for condition_name, eeg_stem in eeg_name_map.items():
                eeg_set_path = session_root / "eeg" / f"{eeg_stem}.set"
                eeg_fdt_path = session_root / "eeg" / f"{eeg_stem}.fdt"
                behavioral_mat_path = session_root / "behavioral" / behavior_name_map[condition_name]
                if not eeg_set_path.exists() or not eeg_fdt_path.exists():
                    continue
                task_records.append(
                    EEGTaskRecord(
                        subject_id=subject_id,
                        session_id=session_id,
                        session_root=session_root,
                        task_condition=condition_name,
                        eeg_set_path=eeg_set_path,
                        eeg_fdt_path=eeg_fdt_path,
                        behavioral_mat_path=behavioral_mat_path if behavioral_mat_path.exists() else None,
                    )
                )
    return task_records


def load_eeglab_header(set_path: str | Path) -> dict[str, object]:
    payload = _load_set_payload(set_path)
    return {
        "nbchan": int(payload["nbchan"]),
        "pnts": int(payload["pnts"]),
        "trials": int(payload["trials"]),
        "srate": int(payload["srate"]),
        "xmin": float(payload["xmin"]),
        "xmax": float(payload["xmax"]),
        "times": np.asarray(payload["times"], dtype=np.float32),
        "datfile": str(payload["datfile"]),
        "events": payload.get("event"),
        "chanlocs": payload.get("chanlocs"),
        "chan_labels": _extract_chan_labels(payload.get("chanlocs")),
    }


def load_eeglab_samples(record: EEGTaskRecord) -> np.ndarray:
    header = load_eeglab_header(record.eeg_set_path)
    raw = np.fromfile(record.eeg_fdt_path, dtype=np.float32)
    expected = int(header["nbchan"]) * int(header["pnts"]) * int(header["trials"])
    if raw.size != expected:
        raise ValueError(
            f"unexpected sample count for {record.eeg_fdt_path}: got {raw.size}, expected {expected}"
        )
    if int(header["trials"]) != 1:
        data = raw.reshape((int(header["nbchan"]), int(header["pnts"]), int(header["trials"])), order="F")
        return data[:, :, 0]
    return raw.reshape((int(header["nbchan"]), int(header["pnts"])), order="F")


def extract_event_rows(set_path: str | Path) -> list[dict[str, float | str]]:
    payload = _load_set_payload(set_path)
    event_payload = payload.get("event")
    if event_payload is None:
        return []
    if hasattr(event_payload, "__len__") and not isinstance(event_payload, (str, bytes)):
        events = event_payload
    else:
        events = [event_payload]
    rows = []
    for event in events:
        event_type = str(getattr(event, "type", ""))
        rows.append(
            {
                "type": event_type,
                "latency": float(getattr(event, "latency", 0.0)),
                "duration": float(getattr(event, "duration", 0.0)),
            }
        )
    return rows


def _format_subject_id(value: int) -> str:
    return f"{int(value):02d}"


def _format_session_id(value: int) -> str:
    return f"ses-{int(value):02d}"


def _extract_numeric_suffix(text: str) -> int:
    match = re.search(r"(\d+)$", text)
    if not match:
        raise ValueError(f"unable to extract numeric suffix from {text!r}")
    return int(match.group(1))


def _normalize_subject_root(path: Path) -> Path:
    nested = list(path.glob("sub-*"))
    if len(nested) == 1 and nested[0].is_dir():
        return nested[0]
    return path


@lru_cache(maxsize=64)
def _load_set_payload(set_path: str | Path):
    return scipy.io.loadmat(Path(set_path), squeeze_me=True, struct_as_record=False)


def _extract_chan_labels(chanlocs) -> list[str]:
    if chanlocs is None:
        return []
    if hasattr(chanlocs, "__len__") and not isinstance(chanlocs, (str, bytes)):
        records = chanlocs
    else:
        records = [chanlocs]
    labels = []
    for record in records:
        label = getattr(record, "labels", "")
        labels.append(str(label))
    return labels
