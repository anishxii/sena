from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.eeg import EEG_SUMMARY_FEATURE_NAMES  # noqa: E402
from emotiv_learn.stew_index import build_stew_feature_index, save_feature_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a STEW EEG summary feature index for retrieval-based emission.")
    parser.add_argument("--stew-dir", default="stew_dataset")
    parser.add_argument("--output", default="artifacts/stew_feature_index.json")
    parser.add_argument("--epoch-sec", type=int, default=30)
    parser.add_argument("--stride-sec", type=int, default=10)
    args = parser.parse_args()

    index = build_stew_feature_index(
        stew_dir=args.stew_dir,
        feature_names=EEG_SUMMARY_FEATURE_NAMES,
        epoch_sec=args.epoch_sec,
        stride_sec=args.stride_sec,
    )
    save_feature_index(index, args.output)
    total_windows = sum(len(windows) for windows in index.windows_by_subject.values())
    print(f"subjects={len(index.windows_by_subject)} windows={total_windows} output={args.output}")


if __name__ == "__main__":
    main()
