from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import ACTION_BANK, DecisionEngine
from emotiv_learn.schemas import Outcome, RewardEvent, SemanticSignals, State, StateMetadata, TaskResult


FEATURE_NAMES = [
    "eeg_theta",
    "eeg_alpha",
    "eeg_beta",
    "eeg_beta_alpha_ratio",
    "eeg_load_proxy",
    "turn_index_norm",
    "difficulty_easy",
    "difficulty_medium",
    "difficulty_hard",
    "prev_correct",
    "prev_latency_norm",
    "prev_reread",
    "recent_clarify_count_norm",
    "recent_progress_norm",
    "checkpoint_mode_flag",
]


@dataclass(frozen=True)
class UserArchetype:
    user_id: str
    preferred_action: str
    difficulty: str
    eeg_load_proxy: float
    clarify_baseline: float
    progress_baseline: float


ARCHETYPES = [
    UserArchetype(
        user_id="user_overloaded",
        preferred_action="simplify",
        difficulty="hard",
        eeg_load_proxy=0.85,
        clarify_baseline=0.70,
        progress_baseline=0.30,
    ),
    UserArchetype(
        user_id="user_advanced",
        preferred_action="deepen",
        difficulty="medium",
        eeg_load_proxy=0.35,
        clarify_baseline=0.10,
        progress_baseline=0.75,
    ),
    UserArchetype(
        user_id="user_examples",
        preferred_action="worked_example",
        difficulty="medium",
        eeg_load_proxy=0.55,
        clarify_baseline=0.40,
        progress_baseline=0.45,
    ),
    UserArchetype(
        user_id="user_scanner",
        preferred_action="highlight_key_points",
        difficulty="easy",
        eeg_load_proxy=0.25,
        clarify_baseline=0.15,
        progress_baseline=0.70,
    ),
]


def build_state(archetype: UserArchetype, turn_index: int) -> State:
    difficulty_easy = 1.0 if archetype.difficulty == "easy" else 0.0
    difficulty_medium = 1.0 if archetype.difficulty == "medium" else 0.0
    difficulty_hard = 1.0 if archetype.difficulty == "hard" else 0.0
    previous_correct = 1.0 if turn_index > 1 and archetype.progress_baseline >= 0.5 else 0.0

    features = [
        0.40 + (0.05 * difficulty_hard),  # eeg_theta
        0.55 - (0.05 * archetype.eeg_load_proxy),  # eeg_alpha
        0.50 + (0.10 * archetype.progress_baseline),  # eeg_beta
        0.90 + (0.40 * archetype.eeg_load_proxy),  # eeg_beta_alpha_ratio
        archetype.eeg_load_proxy,  # eeg_load_proxy
        min(turn_index / 12.0, 1.0),  # turn_index_norm
        difficulty_easy,
        difficulty_medium,
        difficulty_hard,
        previous_correct,
        min(0.20 + archetype.clarify_baseline, 1.0),  # prev_latency_norm
        1.0 if archetype.clarify_baseline > 0.5 else 0.0,  # prev_reread
        archetype.clarify_baseline,  # recent_clarify_count_norm
        archetype.progress_baseline,  # recent_progress_norm
        1.0 if turn_index % 4 == 0 else 0.0,  # checkpoint_mode_flag
    ]

    return State(
        timestamp=turn_index,
        user_id=archetype.user_id,
        features=features,
        feature_names=FEATURE_NAMES,
        metadata=StateMetadata(
            task_type="learn",
            difficulty=archetype.difficulty,
            topic_id="algebra",
        ),
    )


def reward_for_action(archetype: UserArchetype, action_id: str, turn_index: int) -> float:
    reward = 0.0
    progress_signal = archetype.progress_baseline
    clarify_penalty = archetype.clarify_baseline

    if action_id == archetype.preferred_action:
        reward += 1.0
        progress_signal = min(progress_signal + 0.35, 1.0)
        clarify_penalty = max(clarify_penalty - 0.35, 0.0)
    elif action_id == "no_change":
        reward += 0.15 if archetype.progress_baseline > 0.65 else -0.20
    else:
        reward -= 0.15

    if turn_index % 4 == 0:
        reward += 0.80 if action_id == archetype.preferred_action else -0.30

    reward += 0.50 * progress_signal
    reward -= 0.60 * clarify_penalty
    return max(-1.5, min(1.5, reward))


def make_reward_event(state: State, action_id: str, reward: float) -> RewardEvent:
    outcome = Outcome(
        timestamp=state.timestamp,
        user_id=state.user_id,
        action_id=action_id,
        task_result=TaskResult(
            correct=1 if reward > 0.75 else 0,
            latency_s=max(1.0, 6.0 - reward),
            reread=1 if reward < 0 else 0,
            completed=1,
            abandoned=0,
        ),
        semantic_signals=SemanticSignals(
            followup_text="continue" if reward > 0 else "clarify",
            followup_type="continue" if reward > 0 else "clarify",
            confusion_score=max(0.0, -reward),
            comprehension_score=max(0.0, reward),
            engagement_score=max(0.0, min(1.0, 0.5 + reward / 2.0)),
            pace_fast_score=0.2,
            pace_slow_score=0.2,
        ),
        raw={"simulated_reward": reward},
    )

    return RewardEvent(
        timestamp=state.timestamp,
        user_id=state.user_id,
        state_features=state.features,
        action_id=action_id,
        reward=reward,
        outcome=outcome,
    )


def run_policy(engine: DecisionEngine, num_rounds: int = 20) -> dict:
    action_counts: dict[str, dict[str, int]] = {
        archetype.user_id: {action.action_id: 0 for action in ACTION_BANK}
        for archetype in ARCHETYPES
    }
    total_reward = 0.0
    total_turns = 0
    per_user_reward = {archetype.user_id: 0.0 for archetype in ARCHETYPES}

    for turn_index in range(1, num_rounds + 1):
        for archetype in ARCHETYPES:
            state = build_state(archetype, turn_index)
            action_scores = engine.score_actions(state, ACTION_BANK)
            action = engine.select_action(action_scores)
            reward = reward_for_action(archetype, action.action_id, turn_index)
            reward_event = make_reward_event(state, action.action_id, reward)
            engine.update(reward_event)

            action_counts[archetype.user_id][action.action_id] += 1
            total_reward += reward
            total_turns += 1
            per_user_reward[archetype.user_id] += reward

    preferred_action_hits = {
        archetype.user_id: action_counts[archetype.user_id][archetype.preferred_action]
        for archetype in ARCHETYPES
    }
    return {
        "average_reward": total_reward / total_turns,
        "per_user_reward": {
            user_id: reward / num_rounds for user_id, reward in per_user_reward.items()
        },
        "preferred_action_hits": preferred_action_hits,
        "action_counts": action_counts,
    }


def print_summary(label: str, results: dict) -> None:
    print(f"=== {label} ===")
    print(f"average_reward: {results['average_reward']:.3f}")
    print("per_user_reward:")
    for user_id, reward in results["per_user_reward"].items():
        print(f"  {user_id}: {reward:.3f}")
    print("preferred_action_hits:")
    for user_id, count in results["preferred_action_hits"].items():
        print(f"  {user_id}: {count}")
    print("top_actions_by_user:")
    for user_id, counts in results["action_counts"].items():
        top_action = max(counts, key=counts.get)
        print(f"  {user_id}: {top_action} ({counts[top_action]}/{sum(counts.values())})")
    print()


def main() -> None:
    generic_engine = DecisionEngine(
        feature_dim=len(FEATURE_NAMES),
        epsilon=0.10,
        use_personalization=False,
        seed=7,
    )
    personalized_engine = DecisionEngine(
        feature_dim=len(FEATURE_NAMES),
        epsilon=0.10,
        use_personalization=True,
        seed=7,
    )

    generic_results = run_policy(generic_engine)
    personalized_results = run_policy(personalized_engine)

    print_summary("Generic Policy", generic_results)
    print_summary("Personalized Policy", personalized_results)
    print("=== Lift ===")
    print(
        "average_reward_delta: "
        f"{personalized_results['average_reward'] - generic_results['average_reward']:.3f}"
    )
    for archetype in ARCHETYPES:
        delta = (
            personalized_results["per_user_reward"][archetype.user_id]
            - generic_results["per_user_reward"][archetype.user_id]
        )
        print(f"  {archetype.user_id}: {delta:.3f}")


if __name__ == "__main__":
    main()
