"""
Training Log Analyzer
======================
Analyze training simulation logs to visualize and understand the RL training data.

Usage:
    python analyze_training_logs.py --log-file ./training_logs/session_20260414_*.json
    python analyze_training_logs.py --log-dir ./training_logs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_training_log(log_file: Path) -> Dict:
    """Load a single training log JSON file."""
    with open(log_file, 'r') as f:
        return json.load(f)


def analyze_session(data: Dict) -> None:
    """Print detailed analysis of a training session."""
    print("\n" + "="*80)
    print(f"SESSION: {data['session_id']}")
    print("="*80)
    print(f"Topic:         {data['topic']}")
    print(f"Subject:       {data['subject_id']} (workload rating: {data['subject_rating']:.1f}/9)")
    print(f"Total steps:   {len(data['steps'])}")
    print()

    # Extract metrics
    steps = data['steps']
    cognitive_loads = [s['cognitive_load'] for s in steps]
    times_on_chunk = [s['behavioral_cues']['time_on_chunk'] for s in steps]
    scroll_rates = [s['behavioral_cues']['scroll_rate'] for s in steps]
    reread_counts = [s['behavioral_cues']['reread_count'] for s in steps]
    epochs_consumed = [s['epochs_consumed'] for s in steps]

    # Summary statistics
    print("Cognitive Load Statistics:")
    print(f"  Mean:   {np.mean(cognitive_loads):.3f}")
    print(f"  Std:    {np.std(cognitive_loads):.3f}")
    print(f"  Min:    {np.min(cognitive_loads):.3f}")
    print(f"  Max:    {np.max(cognitive_loads):.3f}")
    print()

    print("Time on Chunk Statistics:")
    print(f"  Mean:   {np.mean(times_on_chunk):.1f}s")
    print(f"  Std:    {np.std(times_on_chunk):.1f}s")
    print(f"  Min:    {np.min(times_on_chunk):.1f}s")
    print(f"  Max:    {np.max(times_on_chunk):.1f}s")
    print(f"  Total:  {np.sum(times_on_chunk)/60:.1f} minutes")
    print()

    print("Behavioral Metrics:")
    print(f"  Avg scroll rate:   {np.mean(scroll_rates):.3f}")
    print(f"  Avg reread count:  {np.mean(reread_counts):.1f}")
    print(f"  Total epochs used: {np.sum(epochs_consumed)}")
    print()

    # Action distribution
    user_actions = [s['user_action'] for s in steps]
    rl_actions = [s['rl_action'] for s in steps]

    print("User Action Distribution:")
    for action in ['continue', 'clarify', 'branch']:
        count = user_actions.count(action)
        pct = 100 * count / len(user_actions)
        print(f"  {action:10s}: {count:2d} ({pct:5.1f}%)")
    print()

    print("RL Action Distribution:")
    action_counts = {}
    for action in rl_actions:
        action_counts[action] = action_counts.get(action, 0) + 1

    for action in sorted(action_counts.keys()):
        count = action_counts[action]
        pct = 100 * count / len(rl_actions)
        print(f"  {action:15s}: {count:2d} ({pct:5.1f}%)")
    print()

    # Step-by-step breakdown
    print("Step-by-Step Breakdown:")
    print(f"{'Step':>4} | {'Load':>5} | {'Time':>5} | {'Scroll':>6} | {'Reread':>6} | {'User':>8} | {'RL Action':>15}")
    print("-"*80)

    for s in steps:
        print(
            f"{s['step_number']:>4} | "
            f"{s['cognitive_load']:>5.3f} | "
            f"{s['behavioral_cues']['time_on_chunk']:>5.1f} | "
            f"{s['behavioral_cues']['scroll_rate']:>6.3f} | "
            f"{s['behavioral_cues']['reread_count']:>6} | "
            f"{s['user_action']:>8} | "
            f"{s['rl_action']:>15}"
        )

    print()

    # Identify interesting moments
    print("Interesting Moments:")

    # High cognitive load steps
    high_load_threshold = np.mean(cognitive_loads) + np.std(cognitive_loads)
    high_load_steps = [s for s in steps if s['cognitive_load'] > high_load_threshold]
    if high_load_steps:
        print(f"\n  High cognitive load (>{high_load_threshold:.3f}):")
        for s in high_load_steps:
            print(f"    Step {s['step_number']}: {s['step_title']}")
            print(f"      Load={s['cognitive_load']:.3f}, Action={s['user_action']}")
            if s['user_prompt']:
                print(f"      User: \"{s['user_prompt'][:60]}...\"")

    # Clarify requests
    clarify_steps = [s for s in steps if s['user_action'] == 'clarify']
    if clarify_steps:
        print(f"\n  Clarification requests ({len(clarify_steps)}):")
        for s in clarify_steps:
            print(f"    Step {s['step_number']}: {s['step_title']}")
            if s['user_prompt']:
                print(f"      \"{s['user_prompt'][:70]}...\"")

    # Branch explorations
    branch_steps = [s for s in steps if s['user_action'] == 'branch']
    if branch_steps:
        print(f"\n  Branch explorations ({len(branch_steps)}):")
        for s in branch_steps:
            print(f"    Step {s['step_number']}: {s['step_title']}")
            if s['user_prompt']:
                print(f"      \"{s['user_prompt'][:70]}...\"")

    print("\n" + "="*80 + "\n")


def compare_sessions(log_files: List[Path]) -> None:
    """Compare metrics across multiple training sessions."""
    if len(log_files) < 2:
        print("Need at least 2 log files for comparison")
        return

    print("\n" + "="*80)
    print("CROSS-SESSION COMPARISON")
    print("="*80)

    all_data = [load_training_log(f) for f in log_files]

    print(f"\n{'Session ID':^25} | {'Subject':^8} | {'Steps':^5} | {'Avg Load':^9} | {'Avg Time':^9}")
    print("-"*80)

    for data in all_data:
        steps = data['steps']
        avg_load = np.mean([s['cognitive_load'] for s in steps])
        avg_time = np.mean([s['behavioral_cues']['time_on_chunk'] for s in steps])

        print(
            f"{data['session_id']:^25} | "
            f"{data['subject_id']:^8} | "
            f"{len(steps):^5} | "
            f"{avg_load:^9.3f} | "
            f"{avg_time:^9.1f}"
        )

    print()


def export_rl_training_data(data: Dict, output_file: Path) -> None:
    """
    Export training data in format suitable for RL model training.

    Format: CSV with columns:
        state_eeg_1, ..., state_eeg_62, state_time, state_scroll, state_reread,
        action, reward, next_state_eeg_1, ..., next_state_eeg_62, next_state_time,
        next_state_scroll, next_state_reread
    """
    import csv

    steps = data['steps']
    if len(steps) < 2:
        print("Need at least 2 steps for transition data")
        return

    with open(output_file, 'w', newline='') as f:
        # Build header
        eeg_cols = [f"state_eeg_{i}" for i in range(62)]
        behavioral_cols = ["state_time", "state_scroll", "state_reread"]
        next_eeg_cols = [f"next_state_eeg_{i}" for i in range(62)]
        next_behavioral_cols = ["next_state_time", "next_state_scroll", "next_state_reread"]

        header = eeg_cols + behavioral_cols + ["action", "reward"] + next_eeg_cols + next_behavioral_cols
        writer = csv.writer(f)
        writer.writerow(header)

        # Write state transitions
        for i in range(len(steps) - 1):
            curr_step = steps[i]
            next_step = steps[i + 1]

            # Current state
            curr_eeg = curr_step['eeg_features']
            curr_bc = curr_step['behavioral_cues']
            curr_state = (
                curr_eeg +
                [curr_bc['time_on_chunk'], curr_bc['scroll_rate'], curr_bc['reread_count']]
            )

            # Action (encode as index)
            from train_simulation import CONTENT_ACTIONS
            action_idx = CONTENT_ACTIONS.index(curr_step['rl_action'])

            # Reward (placeholder — would normally come from prompt similarity + engagement)
            # For now, use inverse of cognitive load (lower load = better)
            reward = 1.0 - curr_step['cognitive_load']

            # Next state
            next_eeg = next_step['eeg_features']
            next_bc = next_step['behavioral_cues']
            next_state = (
                next_eeg +
                [next_bc['time_on_chunk'], next_bc['scroll_rate'], next_bc['reread_count']]
            )

            row = curr_state + [action_idx, reward] + next_state
            writer.writerow(row)

    print(f"Exported {len(steps)-1} transitions to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Emotiv Learn training logs")
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to a single training log JSON file",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        help="Path to directory containing multiple training logs",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare metrics across multiple sessions",
    )
    parser.add_argument(
        "--export-rl",
        type=str,
        help="Export RL training data to CSV file",
    )

    args = parser.parse_args()

    log_files = []

    if args.log_file:
        log_files = [Path(args.log_file)]
    elif args.log_dir:
        log_dir = Path(args.log_dir)
        log_files = sorted(log_dir.glob("session_*.json"))

    if not log_files:
        print("No log files found. Specify --log-file or --log-dir")
        return 1

    print(f"Found {len(log_files)} log file(s)")

    # Analyze each session
    for log_file in log_files:
        data = load_training_log(log_file)
        analyze_session(data)

    # Compare if requested
    if args.compare and len(log_files) > 1:
        compare_sessions(log_files)

    # Export RL data if requested
    if args.export_rl:
        data = load_training_log(log_files[0])
        export_rl_training_data(data, Path(args.export_rl))

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
