from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

_DEFAULT_MPLCONFIGDIR = Path("/Users/anish/PERSONAL/emotiv_learn/artifacts/mplconfig")
_DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_DEFAULT_MPLCONFIGDIR))

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROFILE_LABELS = {
    "behavior_only": "Behavior Only",
    "current_eeg": "Behavior + EEG Proxy",
    "tutor_proxy_eeg": "Behavior + EEG + Tutor Proxy",
}

POLICY_LABELS = {
    "personalized": "Personalized",
    "generic": "Generic",
    "fixed_no_change": "Fixed Baseline",
    "random": "Random",
}

PROFILE_ORDER = ["behavior_only", "current_eeg", "tutor_proxy_eeg"]
POLICY_ORDER = ["personalized", "generic", "fixed_no_change", "random"]

COLORS = {
    "ink": "#1F1F1C",
    "muted": "#6F6A64",
    "grid": "#D7D2CB",
    "paper": "#F6F2EB",
    "panel": "#FBF8F3",
    "personalized": "#375E97",
    "generic": "#D97B29",
    "fixed_no_change": "#9B9B9B",
    "random": "#A9546D",
    "highlight": "#1E7F5A",
}


def _configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": COLORS["paper"],
            "axes.facecolor": COLORS["panel"],
            "savefig.facecolor": COLORS["paper"],
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["ink"],
            "axes.titlecolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "grid.alpha": 0.5,
        }
    )


def _load_ablation_runs(input_dir: Path) -> dict[str, dict]:
    runs: dict[str, dict] = {}
    for profile in PROFILE_ORDER:
        path = input_dir / f"{profile}.json"
        if not path.exists():
            continue
        runs[profile] = json.loads(path.read_text(encoding="utf-8"))
    if not runs:
        raise FileNotFoundError(f"no ablation JSON files found in {input_dir}")
    return runs


def _policy_summary_matrix(runs: dict[str, dict]) -> dict[str, dict[str, dict]]:
    matrix: dict[str, dict[str, dict]] = {}
    for profile, payload in runs.items():
        matrix[profile] = payload.get("summary", {})
    return matrix


def _average_cumulative_turn_metric(turn_logs: list[dict], metric_key: str) -> tuple[list[int], list[float]]:
    if not turn_logs:
        return [], []
    grouped: dict[str, list[dict]] = {}
    for row in turn_logs:
        grouped.setdefault(row["user_id"], []).append(row)

    per_user_curves: list[list[float]] = []
    max_turn = 0
    for rows in grouped.values():
        rows = sorted(rows, key=lambda item: item["turn_index"])
        running = 0.0
        curve: list[float] = []
        for row in rows:
            if metric_key == "reward":
                value = float(row.get("reward", 0.0))
            elif metric_key == "oracle_mastery_gain":
                value = float(row.get("student_transition", {}).get("oracle_mastery_gain", 0.0))
            else:
                raise ValueError(f"unsupported metric_key={metric_key}")
            running += value
            curve.append(running)
        per_user_curves.append(curve)
        max_turn = max(max_turn, len(curve))

    averaged: list[float] = []
    for turn_idx in range(max_turn):
        values = [curve[turn_idx] for curve in per_user_curves if turn_idx < len(curve)]
        averaged.append(float(sum(values) / len(values)))
    return list(range(1, max_turn + 1)), averaged


def _format_policy_name(policy: str) -> str:
    return POLICY_LABELS.get(policy, policy.replace("_", " ").title())


def _format_profile_name(profile: str) -> str:
    return PROFILE_LABELS.get(profile, profile.replace("_", " ").title())


def _draw_metric_panel(ax, matrix: dict[str, dict[str, dict]], metric_key: str, title: str, subtitle: str) -> None:
    profiles = [profile for profile in PROFILE_ORDER if profile in matrix]
    policies = [policy for policy in POLICY_ORDER if any(policy in matrix[profile] for profile in profiles)]
    x = np.arange(len(profiles))

    for policy in policies:
        y = [matrix[profile].get(policy, {}).get(metric_key, np.nan) for profile in profiles]
        color = COLORS.get(policy, COLORS["ink"])
        ax.plot(x, y, color=color, linewidth=2.2, marker="o", markersize=7, label=_format_policy_name(policy))
        for xi, yi in zip(x, y, strict=False):
            if np.isnan(yi):
                continue
            ax.text(
                xi,
                yi,
                f"{yi:.3f}",
                color=color,
                fontsize=9,
                ha="center",
                va="bottom",
            )

    ax.set_title(title, loc="left", fontsize=16, fontweight="semibold", pad=16)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=10, color=COLORS["muted"], ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([_format_profile_name(profile) for profile in profiles], fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.legend(frameon=False, fontsize=10, loc="upper left")


def _draw_turn_panel(ax, runs: dict[str, dict], policy_mode: str, metric_key: str, title: str) -> None:
    for profile in PROFILE_ORDER:
        payload = runs.get(profile)
        if payload is None:
            continue
        policy_blob = payload.get("results", {}).get(policy_mode)
        if not policy_blob:
            continue
        turns, values = _average_cumulative_turn_metric(policy_blob.get("turn_logs", []), metric_key)
        if not turns:
            continue
        color = {
            "behavior_only": "#8D99AE",
            "current_eeg": "#4C78A8",
            "tutor_proxy_eeg": COLORS["highlight"],
        }.get(profile, COLORS["ink"])
        ax.plot(turns, values, linewidth=2.5, color=color, label=_format_profile_name(profile))
        ax.scatter(turns[-1], values[-1], color=color, s=34, zorder=5)
        ax.text(turns[-1], values[-1], f"  {values[-1]:.3f}", color=color, fontsize=9, va="center")

    ax.set_title(title, loc="left", fontsize=16, fontweight="semibold", pad=16)
    ax.text(
        0.0,
        1.02,
        f"Cumulative {metric_key.replace('_', ' ')} for the {_format_policy_name(policy_mode).lower()} policy.",
        transform=ax.transAxes,
        fontsize=10,
        color=COLORS["muted"],
        ha="left",
    )
    ax.set_xlabel("Turn", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines["left"].set_color(COLORS["grid"])
    ax.spines["bottom"].set_color(COLORS["grid"])
    ax.legend(frameon=False, fontsize=10, loc="upper left")


def _best_profile_callout(matrix: dict[str, dict[str, dict]], metric_key: str, policy_mode: str) -> str:
    best_profile = None
    best_value = None
    for profile in PROFILE_ORDER:
        value = matrix.get(profile, {}).get(policy_mode, {}).get(metric_key)
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_profile = profile
    if best_profile is None or best_value is None:
        return "Awaiting completed runs."
    return (
        f"Best {metric_key.replace('_', ' ')} so far: {_format_profile_name(best_profile)} "
        f"at {best_value:.3f} for {_format_policy_name(policy_mode).lower()}."
    )


def build_ablation_figure(input_dir: Path, output_path: Path) -> Path:
    _configure_style()
    runs = _load_ablation_runs(input_dir)
    matrix = _policy_summary_matrix(runs)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.26, wspace=0.16)
    ax_reward = fig.add_subplot(gs[0, 0])
    ax_oracle = fig.add_subplot(gs[0, 1])
    ax_turns = fig.add_subplot(gs[1, :])

    _draw_metric_panel(
        ax_reward,
        matrix,
        metric_key="average_reward",
        title="Reward Lift Across Observable State Designs",
        subtitle="Each line holds policy fixed and changes only the state construction used by the decision engine.",
    )
    _draw_metric_panel(
        ax_oracle,
        matrix,
        metric_key="total_oracle_mastery_gain",
        title="Oracle Mastery Gain Across State Designs",
        subtitle="This is the cleaner backward-pass proxy: how much hidden mastery improved over the run.",
    )
    _draw_turn_panel(
        ax_turns,
        runs,
        policy_mode="personalized",
        metric_key="reward",
        title="Personalized Policy Reward Trajectory",
    )

    fig.suptitle(
        "Emotiv Learn Ablation Study",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=24,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.935,
        "A restrained matplotlib view of how richer observable state changes policy performance.",
        fontsize=12,
        color=COLORS["muted"],
    )
    fig.text(
        0.055,
        0.045,
        _best_profile_callout(matrix, metric_key="average_reward", policy_mode="personalized"),
        fontsize=11,
        color=COLORS["highlight"],
    )
    fig.text(
        0.945,
        0.045,
        f"Source: {input_dir}",
        fontsize=10,
        color=COLORS["muted"],
        ha="right",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a polished matplotlib summary of the state ablation study.")
    parser.add_argument(
        "--input-dir",
        default="/Users/anish/PERSONAL/emotiv_learn/artifacts/state_ablation_10turn",
        help="Directory containing behavior_only.json/current_eeg.json/tutor_proxy_eeg.json",
    )
    parser.add_argument(
        "--output",
        default="/Users/anish/PERSONAL/emotiv_learn/artifacts/state_ablation_10turn/ablation_summary.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    output_path = build_ablation_figure(Path(args.input_dir), Path(args.output))
    print(f"wrote ablation figure to {output_path}")


if __name__ == "__main__":
    main()
