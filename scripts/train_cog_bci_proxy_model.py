from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.cog_bci_ingest import COGBCINBackWindow, build_cog_bci_nback_windows  # noqa: E402
from emotiv_learn.cog_bci_proxy_model import fit_cog_bci_proxy_regressor, save_cog_bci_proxy_regressor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a first COG-BCI EEG proxy-state regressor.")
    parser.add_argument("--cog-bci-dir", default="data/cog_bci")
    parser.add_argument("--windows-json", default=None)
    parser.add_argument("--output", default="artifacts/cog_bci_proxy_model.json")
    parser.add_argument("--window-sec", type=int, default=30)
    parser.add_argument("--stride-sec", type=int, default=10)
    parser.add_argument("--limit-subjects", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1e-3)
    args = parser.parse_args()

    if args.windows_json:
        payload = json.loads(Path(args.windows_json).read_text(encoding="utf-8"))
        windows = [COGBCINBackWindow(**row) for row in payload]
    else:
        subjects = None
        if args.limit_subjects is not None:
            subjects = sorted(path.stem for path in Path(args.cog_bci_dir).glob("sub-*.zip"))[: args.limit_subjects]
        windows = build_cog_bci_nback_windows(
            cog_bci_dir=args.cog_bci_dir,
            subject_ids=subjects,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
        )

    model = fit_cog_bci_proxy_regressor(windows=windows, ridge_alpha=args.ridge_alpha)
    save_cog_bci_proxy_regressor(model, args.output)
    print(f"trained proxy model on {len(windows)} windows -> {args.output}")


if __name__ == "__main__":
    main()
