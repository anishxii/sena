from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.validation.cog_bci import (
    build_subject_nback_recording_summaries,
    load_nback_condition_labels,
    load_trigger_codes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect parsed COG-BCI questionnaire and trigger metadata.")
    parser.add_argument("--cog-bci-dir", default="data/cog_bci")
    parser.add_argument("--subject-id", default=None)
    args = parser.parse_args()

    labels = load_nback_condition_labels(args.cog_bci_dir)
    by_condition = defaultdict(list)
    for label in labels:
        by_condition[label.condition].append(label.workload_estimate)

    print(f"N-Back condition labels: {len(labels)}")
    for condition, values in sorted(by_condition.items()):
        print(
            f"{condition}: n={len(values)} "
            f"min={min(values):.3f} mean={sum(values) / len(values):.3f} max={max(values):.3f}"
        )

    trigger_path = Path(args.cog_bci_dir) / "triggerlist.txt"
    if trigger_path.exists():
        nback_triggers = [trigger for trigger in load_trigger_codes(trigger_path) if "BACK" in trigger.content.upper()]
        print(f"N-Back trigger codes: {len(nback_triggers)}")
        for trigger in nback_triggers[:12]:
            print(f"  {trigger.code}: {trigger.content}")

    if args.subject_id:
        summaries = build_subject_nback_recording_summaries(args.cog_bci_dir, args.subject_id)
        print(f"{args.subject_id} N-Back recording summaries: {len(summaries)}")
        for summary in summaries:
            print(
                f"  ses={summary.session_id} condition={summary.condition} "
                f"difficulty={summary.difficulty_level} workload={summary.workload_estimate:.3f} "
                f"accuracy={summary.response_accuracy:.3f} rt={summary.mean_response_time_s}"
            )


if __name__ == "__main__":
    main()
