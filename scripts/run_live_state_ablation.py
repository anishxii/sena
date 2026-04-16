from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.live_training import (  # noqa: E402
    STATE_PROFILE_BEHAVIOR_ONLY,
    STATE_PROFILE_CURRENT_EEG,
    STATE_PROFILE_TUTOR_PROXY_EEG,
)
from scripts.live_policy_comparison import run_live_policy_comparison  # noqa: E402


ABLATION_PROFILES = [
    STATE_PROFILE_BEHAVIOR_ONLY,
    STATE_PROFILE_CURRENT_EEG,
    STATE_PROFILE_TUTOR_PROXY_EEG,
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the live policy comparison across state-ablation profiles.")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default=None)
    parser.add_argument("--modes", default="personalized,generic,fixed_no_change")
    parser.add_argument("--users", default=None)
    parser.add_argument("--eeg-mode", default="synthetic", choices=["synthetic", "retrieved_real"])
    parser.add_argument("--stew-dir", default="stew_dataset")
    parser.add_argument("--eeg-mapper-path", default="artifacts/stew_workload_mapper.json")
    parser.add_argument("--cog-proxy-model-path", default=None)
    parser.add_argument("--output-dir", default="artifacts/state_ablation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for profile in ABLATION_PROFILES:
        output_path = output_dir / f"{profile}.json"
        events_path = output_dir / f"{profile}.jsonl"
        run_live_policy_comparison(
            turns=args.turns,
            seed=args.seed,
            model=args.model,
            output_path=output_path,
            events_output_path=events_path,
            policy_modes=args.modes.split(",") if args.modes else None,
            user_ids=args.users.split(",") if args.users else None,
            eeg_mode=args.eeg_mode,
            stew_dir=args.stew_dir,
            eeg_mapper_path=args.eeg_mapper_path,
            cog_proxy_model_path=args.cog_proxy_model_path,
            state_profile=profile,
        )
        print(f"completed state profile {profile} -> {output_path}")


if __name__ == "__main__":
    main()
