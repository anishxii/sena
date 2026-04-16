from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.validation.runner import (
    run_cog_bci_nback_validation,
    run_toy_nback_validation,
    summarize_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the N-Back policy validation scaffold.")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--source", choices=["toy", "cog_bci"], default="toy")
    parser.add_argument("--subject-id", default=None)
    parser.add_argument("--cog-bci-dir", default="data/cog_bci")
    parser.add_argument("--output", default="artifacts/nback_policy_validation.json")
    args = parser.parse_args()

    subject_id = args.subject_id or ("toy_sub02" if args.source == "toy" else "sub-01")
    if args.source == "toy":
        results = run_toy_nback_validation(turns=args.turns, seed=args.seed, subject_id=subject_id)
    else:
        results = run_cog_bci_nback_validation(
            cog_bci_dir=args.cog_bci_dir,
            turns=args.turns,
            seed=args.seed,
            subject_id=subject_id,
        )
    summary = summarize_validation(results)
    output = {
        "turns": args.turns,
        "seed": args.seed,
        "source": args.source,
        "subject_id": subject_id,
        "summary": summary,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    for policy_name, row in summary.items():
        print(
            f"{policy_name}: avg_reward={row['average_reward']:.3f} "
            f"avg_workload={row['average_workload']:.3f} avg_accuracy={row['average_accuracy']:.3f} "
            f"overload_rate={row['overload_rate']:.3f} actions={row['action_counts']}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
