from __future__ import annotations

from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import scipy.io

from .eeg_features import compute_eeg_summary_for_channels
from .cog_bci_metadata import (
    NBACK_CONDITION_TO_DIFFICULTY,
    NBACK_CONDITION_TO_SET_FILE,
    build_subject_nback_recording_summaries,
)


@dataclass(frozen=True)
class COGBCINBackWindow:
    window_id: str
    subject_id: str
    session_id: str
    condition: str
    difficulty_level: int
    start_offset_s: float
    duration_s: float
    workload_estimate: float
    rolling_accuracy: float
    rolling_rt_percentile: float
    lapse_rate: float
    eeg_features: list[float]
    metadata: dict[str, float | int | str]


def list_available_subject_archives(cog_bci_dir: str | Path) -> list[str]:
    root = Path(cog_bci_dir)
    available: list[str] = []
    for path in sorted(root.glob("sub-*.zip")):
        try:
            with zipfile.ZipFile(path) as archive:
                archive.testzip()
        except zipfile.BadZipFile:
            continue
        available.append(path.stem)
    return available


def build_cog_bci_nback_windows(
    cog_bci_dir: str | Path,
    subject_ids: list[str] | None = None,
    window_sec: int = 30,
    stride_sec: int = 10,
) -> list[COGBCINBackWindow]:
    available = list_available_subject_archives(cog_bci_dir)
    subjects = subject_ids or available
    all_windows: list[COGBCINBackWindow] = []
    for subject_id in subjects:
        if subject_id not in available:
            continue
        all_windows.extend(
            build_subject_cog_bci_nback_windows(
                cog_bci_dir=cog_bci_dir,
                subject_id=subject_id,
                window_sec=window_sec,
                stride_sec=stride_sec,
            )
        )
    return all_windows


def build_subject_cog_bci_nback_windows(
    cog_bci_dir: str | Path,
    subject_id: str,
    window_sec: int = 30,
    stride_sec: int = 10,
) -> list[COGBCINBackWindow]:
    root = Path(cog_bci_dir)
    zip_path = root / f"{subject_id}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"missing COG-BCI subject archive: {zip_path}")

    summaries = {
        (summary.session_id, summary.condition): summary
        for summary in build_subject_nback_recording_summaries(cog_bci_dir=root, subject_id=subject_id)
    }

    windows: list[COGBCINBackWindow] = []
    rt_values = [summary.mean_response_time_s for summary in summaries.values() if summary.mean_response_time_s is not None]
    rt_min = min(rt_values) if rt_values else 0.0
    rt_span = max((max(rt_values) - rt_min), 1e-6) if rt_values else 1.0

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        for session_id in ["1", "2", "3"]:
            for condition, set_name in NBACK_CONDITION_TO_SET_FILE.items():
                summary = summaries.get((session_id, condition))
                if summary is None:
                    continue
                set_path = _resolve_archive_member(names=names, suffix=f"/ses-S{session_id}/eeg/{set_name}")
                if set_path is None:
                    continue
                fdt_path = set_path.replace(".set", ".fdt")
                if fdt_path not in names:
                    continue
                signal, channel_names, sampling_rate = _load_eeglab_signal_from_archive(
                    archive=archive,
                    set_path=set_path,
                    fdt_path=fdt_path,
                )
                windows.extend(
                    _window_recording(
                        signal=signal,
                        channel_names=channel_names,
                        sampling_rate=sampling_rate,
                        subject_id=subject_id,
                        session_id=session_id,
                        condition=condition,
                        summary=summary,
                        window_sec=window_sec,
                        stride_sec=stride_sec,
                        rt_min=rt_min,
                        rt_span=rt_span,
                    )
                )
    return windows


def save_cog_bci_nback_windows(windows: list[COGBCINBackWindow], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(window) for window in windows]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _window_recording(
    *,
    signal: np.ndarray,
    channel_names: list[str],
    sampling_rate: int,
    subject_id: str,
    session_id: str,
    condition: str,
    summary,
    window_sec: int,
    stride_sec: int,
    rt_min: float,
    rt_span: float,
) -> list[COGBCINBackWindow]:
    window_samples = sampling_rate * window_sec
    stride_samples = sampling_rate * stride_sec
    last_start = signal.shape[0] - window_samples
    if last_start < 0:
        return []

    if summary.mean_response_time_s is None:
        rt_percentile = 0.5
    else:
        rt_percentile = (summary.mean_response_time_s - rt_min) / rt_span

    windows: list[COGBCINBackWindow] = []
    epoch_index = 0
    for start in range(0, last_start + 1, stride_samples):
        window = signal[start : start + window_samples]
        features = compute_eeg_summary_for_channels(window=window, channel_names=channel_names, fs=sampling_rate)
        windows.append(
            COGBCINBackWindow(
                window_id=f"{subject_id}_ses{session_id}_{condition}_epoch_{epoch_index:04d}",
                subject_id=subject_id,
                session_id=session_id,
                condition=condition,
                difficulty_level=NBACK_CONDITION_TO_DIFFICULTY[condition],
                start_offset_s=round(start / sampling_rate, 3),
                duration_s=float(window_sec),
                workload_estimate=float(summary.workload_estimate),
                rolling_accuracy=float(summary.response_accuracy),
                rolling_rt_percentile=float(max(0.0, min(1.0, rt_percentile))),
                lapse_rate=float(max(0.0, min(1.0, 1.0 - summary.response_accuracy))),
                eeg_features=[float(value) for value in features],
                metadata={
                    "source": "cog_bci_nback_window",
                    "sampling_rate": sampling_rate,
                    "nbchan": len(channel_names),
                },
            )
        )
        epoch_index += 1
    return windows


def _load_eeglab_signal_from_archive(
    *,
    archive: zipfile.ZipFile,
    set_path: str,
    fdt_path: str,
) -> tuple[np.ndarray, list[str], int]:
    set_payload = archive.read(set_path)
    mat = scipy.io.loadmat(io.BytesIO(set_payload), squeeze_me=True, struct_as_record=False)
    data_file = str(mat["data"])
    if not data_file.lower().endswith(".fdt"):
        raise ValueError(f"{set_path}: expected FDT-backed recording, got data={data_file}")
    sampling_rate = int(mat["srate"])
    points = int(mat["pnts"])
    channels = int(mat["nbchan"])
    trials = int(mat["trials"])
    if trials != 1:
        raise ValueError(f"{set_path}: expected continuous recording with trials=1, got {trials}")

    channel_names = [str(chan.labels) for chan in mat["chanlocs"]]
    raw = np.frombuffer(archive.read(fdt_path), dtype="<f4")
    expected = channels * points * trials
    if raw.size != expected:
        raise ValueError(f"{fdt_path}: expected {expected} float32 values, got {raw.size}")
    signal = raw.reshape((channels, points * trials), order="F").T
    return signal, channel_names, sampling_rate


def _resolve_archive_member(names: list[str], suffix: str) -> str | None:
    for name in names:
        if name.endswith(suffix):
            return name
    return None
