from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.eeg_mapper import fit_stew_workload_feature_mapper, save_stew_workload_feature_mapper  # noqa: E402
from emotiv_learn.stew_index import build_stew_feature_index, load_feature_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a workload-conditioned EEG target mapper from STEW.")
    parser.add_argument("--stew-dir", default="stew_dataset")
    parser.add_argument("--index-path", default="artifacts/stew_feature_index.json")
    parser.add_argument("--output", default="artifacts/stew_workload_mapper.json")
    parser.add_argument("--epoch-sec", type=int, default=30)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    args = parser.parse_args()

    index_path = Path(args.index_path)
    if index_path.exists():
        feature_index = load_feature_index(index_path)
    else:
        feature_index = build_stew_feature_index(
            stew_dir=args.stew_dir,
            feature_names=[
                "eeg_theta_mean",
                "eeg_alpha_mean",
                "eeg_beta_mean",
                "eeg_gamma_mean",
                "eeg_frontal_alpha_asymmetry",
                "eeg_frontal_alpha_asymmetry_abs",
                "eeg_frontal_theta_alpha_ratio_mean",
                "eeg_load_score",
            ],
            epoch_sec=args.epoch_sec,
            stride_sec=args.stride_sec,
        )

    mapper = fit_stew_workload_feature_mapper(feature_index=feature_index, ridge_alpha=args.ridge_alpha)
    save_stew_workload_feature_mapper(mapper, args.output)
    print(f"saved mapper to {args.output}")


if __name__ == "__main__":
    main()
