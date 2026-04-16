from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.cog_bci_ingest import build_cog_bci_nback_windows, list_available_subject_archives, save_cog_bci_nback_windows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build windowed N-Back EEG summaries from local COG-BCI subject archives.")
    parser.add_argument("--cog-bci-dir", default="data/cog_bci")
    parser.add_argument("--output", default="artifacts/cog_bci_nback_windows.json")
    parser.add_argument("--window-sec", type=int, default=30)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--limit-subjects", type=int, default=None)
    args = parser.parse_args()

    subjects = list_available_subject_archives(args.cog_bci_dir)
    if args.limit_subjects is not None:
        subjects = subjects[: args.limit_subjects]
    windows = build_cog_bci_nback_windows(
        cog_bci_dir=args.cog_bci_dir,
        subject_ids=subjects,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
    )
    save_cog_bci_nback_windows(windows, args.output)
    print(f"subjects={len(subjects)} windows={len(windows)} output={args.output}")


if __name__ == "__main__":
    main()
