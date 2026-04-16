from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.cog_bci_ingest import (  # noqa: E402
    COGBCINBackWindow,
    build_subject_cog_bci_nback_windows,
    save_cog_bci_nback_windows,
)
from emotiv_learn.cog_bci_proxy_model import (  # noqa: E402
    fit_cog_bci_proxy_regressor,
    save_cog_bci_proxy_regressor,
)


def _subject_ids(limit: int | None) -> list[str]:
    subjects = [f"sub-{index:02d}" for index in range(1, 17)]
    return subjects if limit is None else subjects[:limit]


def _load_windows(path: Path) -> list[COGBCINBackWindow]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [COGBCINBackWindow(**row) for row in payload]


def _write_progress(
    path: Path,
    *,
    status: str,
    completed_subjects: list[str],
    failed_subjects: list[dict[str, str]],
    total_subjects: int,
    total_windows: int,
    started_at: float,
    last_subject: str | None = None,
    last_subject_windows: int | None = None,
    elapsed_s: float | None = None,
) -> None:
    payload = {
        "status": status,
        "completed_subjects": completed_subjects,
        "completed_count": len(completed_subjects),
        "failed_subjects": failed_subjects,
        "failed_count": len(failed_subjects),
        "total_subjects": total_subjects,
        "total_windows": total_windows,
        "last_subject": last_subject,
        "last_subject_windows": last_subject_windows,
        "started_at_epoch_s": round(started_at, 3),
        "elapsed_s": round(elapsed_s if elapsed_s is not None else (time.time() - started_at), 3),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache COG-BCI window features per subject and train a proxy regressor from the cached artifact."
    )
    parser.add_argument("--cog-bci-dir", default="data/cog_bci")
    parser.add_argument("--cache-dir", default="artifacts/cog_bci_cache")
    parser.add_argument("--window-sec", type=int, default=30)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    started_at = time.time()
    cache_dir = Path(args.cache_dir)
    subject_cache_dir = cache_dir / "subjects"
    subject_cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cache_dir / "progress.json"
    combined_windows_path = cache_dir / "cog_bci_nback_windows_full.json"
    model_path = cache_dir / "cog_bci_proxy_model_full.json"

    subjects = _subject_ids(args.limit_subjects)
    completed_subjects: list[str] = []
    failed_subjects: list[dict[str, str]] = []
    total_windows = 0

    _write_progress(
        progress_path,
        status="running",
        completed_subjects=completed_subjects,
        failed_subjects=failed_subjects,
        total_subjects=len(subjects),
        total_windows=total_windows,
        started_at=started_at,
    )

    all_windows: list[COGBCINBackWindow] = []
    for index, subject_id in enumerate(subjects, start=1):
        subject_cache_path = subject_cache_dir / f"{subject_id}.json"
        if subject_cache_path.exists() and not args.force:
            subject_windows = _load_windows(subject_cache_path)
            source = "cache"
        else:
            try:
                subject_windows = build_subject_cog_bci_nback_windows(
                    cog_bci_dir=args.cog_bci_dir,
                    subject_id=subject_id,
                    window_sec=args.window_sec,
                    stride_sec=args.stride_sec,
                )
            except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
                failed_subjects.append({"subject_id": subject_id, "error": exc.__class__.__name__})
                _write_progress(
                    progress_path,
                    status="running",
                    completed_subjects=completed_subjects,
                    failed_subjects=failed_subjects,
                    total_subjects=len(subjects),
                    total_windows=total_windows,
                    started_at=started_at,
                    last_subject=subject_id,
                    last_subject_windows=None,
                    elapsed_s=time.time() - started_at,
                )
                print(f"[{index}/{len(subjects)}] {subject_id}: skipped ({exc.__class__.__name__})", flush=True)
                continue
            save_cog_bci_nback_windows(subject_windows, subject_cache_path)
            source = "fresh"

        all_windows.extend(subject_windows)
        completed_subjects.append(subject_id)
        total_windows += len(subject_windows)
        elapsed_s = time.time() - started_at

        _write_progress(
            progress_path,
            status="running",
            completed_subjects=completed_subjects,
            failed_subjects=failed_subjects,
            total_subjects=len(subjects),
            total_windows=total_windows,
            started_at=started_at,
            last_subject=subject_id,
            last_subject_windows=len(subject_windows),
            elapsed_s=elapsed_s,
        )
        print(
            f"[{index}/{len(subjects)}] {subject_id}: {len(subject_windows)} windows ({source}) "
            f"total={total_windows} elapsed={elapsed_s:.1f}s",
            flush=True,
        )

    save_cog_bci_nback_windows(all_windows, combined_windows_path)
    model = fit_cog_bci_proxy_regressor(windows=all_windows, ridge_alpha=args.ridge_alpha)
    save_cog_bci_proxy_regressor(model, model_path)

    _write_progress(
        progress_path,
        status="complete",
        completed_subjects=completed_subjects,
        failed_subjects=failed_subjects,
        total_subjects=len(subjects),
        total_windows=total_windows,
        started_at=started_at,
        last_subject=completed_subjects[-1] if completed_subjects else None,
        last_subject_windows=None,
        elapsed_s=time.time() - started_at,
    )
    summary = {
        "combined_windows_path": str(combined_windows_path),
        "model_path": str(model_path),
        "subjects": len(subjects),
        "successful_subjects": len(completed_subjects),
        "failed_subjects": failed_subjects,
        "windows": len(all_windows),
    }
    (cache_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
