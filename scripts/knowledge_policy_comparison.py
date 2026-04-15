from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import ACTION_BANK, DecisionEngine
from emotiv_learn.live_training import LIVE_FEATURE_NAMES, LiveLLMStateBuilder, LiveStateInput
from emotiv_learn.llm_contracts import compute_reward_from_interpreted
from emotiv_learn.schemas import Outcome, RewardEvent, SemanticSignals, TaskResult
from emotiv_learn.student_model import HiddenKnowledgeState, HiddenKnowledgeStudent, default_hidden_knowledge_state


CONTENT_STEPS = [
    ("gradient", False),
    ("learning_rate", False),
    ("gradient_descent_update", True),
    ("overshooting", False),
    ("convergence", True),
]


USER_PROFILES = {
    "advanced_concise": {
        "mastery": {
            "gradient": 0.80,
            "learning_rate": 0.68,
            "gradient_descent_update": 0.62,
            "overshooting": 0.60,
            "convergence": 0.58,
        },
        "confidence": 0.82,
        "curiosity": 0.68,
        "fatigue": 0.12,
        "preferred_style": {
            "concise": 0.95,
            "technical_depth": 0.85,
            "worked_examples": 0.35,
            "accessible": 0.35,
        },
    },
    "example_builder": {
        "mastery": {
            "gradient": 0.42,
            "learning_rate": 0.32,
            "gradient_descent_update": 0.24,
            "overshooting": 0.28,
            "convergence": 0.25,
        },
        "confidence": 0.45,
        "curiosity": 0.52,
        "fatigue": 0.20,
        "preferred_style": {
            "worked_examples": 0.96,
            "step_by_step": 0.88,
            "accessible": 0.70,
            "technical_depth": 0.25,
        },
    },
    "visual_scanner": {
        "mastery": {
            "gradient": 0.55,
            "learning_rate": 0.48,
            "gradient_descent_update": 0.40,
            "overshooting": 0.43,
            "convergence": 0.38,
        },
        "confidence": 0.58,
        "curiosity": 0.50,
        "fatigue": 0.34,
        "preferred_style": {
            "structured": 0.97,
            "concise": 0.82,
            "worked_examples": 0.42,
            "technical_depth": 0.45,
        },
    },
}


ACTION_TEMPLATES = {
    "no_change": "{concept} is an important idea in gradient descent. Keep connecting it to the loss and the update rule.",
    "simplify": "{concept} means the simple version of the idea. Think of it as one small piece of how the model learns.",
    "deepen": "{concept} connects to the derivative, parameter update geometry, optimization stability, and loss-surface behavior.",
    "summarize": "{concept}: the key takeaway is how this part changes the next gradient descent step.",
    "highlight_key_points": "- {concept} is the focus.\n- Watch how it affects the update.\n- Keep the loss direction in mind.",
    "worked_example": "Example: suppose the current parameter is 4. Step 1: compute the gradient. Step 2: apply {concept}. Step 3: update the parameter.",
    "analogy": "Imagine hiking downhill. {concept} is like choosing how you read the slope before taking the next step.",
    "step_by_step": "Step 1: identify {concept}. Step 2: connect it to the gradient. Step 3: use it in the parameter update.",
}


def run_comparison(turns: int, seed: int) -> dict[str, list[dict]]:
    return {
        "personalized": run_policy_mode(policy_mode="personalized", turns=turns, seed=seed),
        "generic": run_policy_mode(policy_mode="generic", turns=turns, seed=seed),
        "fixed_no_change": run_policy_mode(policy_mode="fixed_no_change", turns=turns, seed=seed),
        "random": run_policy_mode(policy_mode="random", turns=turns, seed=seed),
    }


def run_policy_mode(policy_mode: str, turns: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    state_builder = LiveLLMStateBuilder()
    students = {
        user_id: HiddenKnowledgeStudent(_initial_state_for_user(user_id), seed=seed + index * 101)
        for index, user_id in enumerate(USER_PROFILES)
    }
    engine = None
    if policy_mode in {"personalized", "generic"}:
        engine = DecisionEngine(
            feature_dim=len(LIVE_FEATURE_NAMES),
            epsilon=0.10,
            use_personalization=policy_mode == "personalized",
            seed=seed,
            reward_clip_abs=1.5,
            update_clip_abs=0.2,
            l2_weight_decay=0.001,
        )

    trackers = {
        user_id: {
            "previous_interpreted": None,
            "previous_student_response": None,
            "previous_reward": 0.0,
            "total_reward": 0.0,
            "total_oracle_gain": 0.0,
            "checkpoint_correct": 0,
            "checkpoint_count": 0,
            "followups": Counter(),
            "action_counts": Counter(),
        }
        for user_id in USER_PROFILES
    }

    for turn_index in range(1, turns + 1):
        for user_id, student in students.items():
            tracker = trackers[user_id]
            concept_id, checkpoint_expected = CONTENT_STEPS[(turn_index - 1) % len(CONTENT_STEPS)]
            state = state_builder.build_state(
                LiveStateInput(
                    timestamp=turn_index,
                    user_id=user_id,
                    topic_id="gradient_descent",
                    task_type="learn",
                    difficulty="medium",
                    turn_index=turn_index,
                    max_turns=turns,
                    interpreted=tracker["previous_interpreted"],
                    student_response=tracker["previous_student_response"],
                    previous_reward=tracker["previous_reward"],
                )
            )
            action_id = _select_action(policy_mode, engine, state, rng)
            tutor_message = _tutor_message(action_id, concept_id, checkpoint_expected)
            transition = student.step(
                concept_id=concept_id,
                action_id=action_id,
                tutor_message=tutor_message,
                checkpoint_expected=checkpoint_expected,
            )
            interpreted = _transition_to_interpreted(transition)
            reward = compute_reward_from_interpreted(interpreted)

            if engine is not None:
                outcome = _make_outcome(turn_index, user_id, action_id, interpreted)
                engine.update(
                    RewardEvent(
                        timestamp=turn_index,
                        user_id=user_id,
                        state_features=state.features,
                        action_id=action_id,
                        reward=reward,
                        outcome=outcome,
                    )
                )

            tracker["total_reward"] += reward
            tracker["total_oracle_gain"] += transition.oracle_mastery_gain
            tracker["action_counts"][action_id] += 1
            tracker["followups"][transition.sampled_response_type] += 1
            if transition.checkpoint_correct is not None:
                tracker["checkpoint_count"] += 1
                tracker["checkpoint_correct"] += int(transition.checkpoint_correct)

            tracker["previous_interpreted"] = interpreted
            tracker["previous_student_response"] = {
                "response_type": transition.sampled_response_type,
                "self_reported_confidence": transition.observable_signals["confidence"],
            }
            tracker["previous_reward"] = reward

    return [_summarize_user(user_id, trackers[user_id], turns) for user_id in USER_PROFILES]


def _summarize_user(user_id: str, tracker: dict, turns: int) -> dict:
    return {
        "user_id": user_id,
        "average_reward": tracker["total_reward"] / turns,
        "total_oracle_mastery_gain": tracker["total_oracle_gain"],
        "average_oracle_mastery_gain": tracker["total_oracle_gain"] / turns,
        "checkpoint_accuracy": (
            tracker["checkpoint_correct"] / tracker["checkpoint_count"] if tracker["checkpoint_count"] else 0.0
        ),
        "followups": dict(tracker["followups"]),
        "action_counts": {action.action_id: tracker["action_counts"][action.action_id] for action in ACTION_BANK},
    }


def _select_action(policy_mode: str, engine: DecisionEngine | None, state, rng: random.Random) -> str:
    if policy_mode == "fixed_no_change":
        return "no_change"
    if policy_mode == "random":
        return rng.choice([action.action_id for action in ACTION_BANK])
    if engine is None:
        raise ValueError(f"policy mode requires engine: {policy_mode}")
    action_scores = engine.score_actions(state, ACTION_BANK)
    return engine.select_action(action_scores).action_id


def _initial_state_for_user(user_id: str) -> HiddenKnowledgeState:
    base = default_hidden_knowledge_state()
    profile = USER_PROFILES[user_id]
    preferred_style = dict(base.preferred_style)
    preferred_style.update(profile["preferred_style"])
    return replace(
        base,
        concept_mastery=profile["mastery"],
        confidence=profile["confidence"],
        curiosity=profile["curiosity"],
        fatigue=profile["fatigue"],
        preferred_style=preferred_style,
    )


def _tutor_message(action_id: str, concept_id: str, checkpoint_expected: bool) -> str:
    message = ACTION_TEMPLATES[action_id].format(concept=concept_id.replace("_", " "))
    if checkpoint_expected:
        message += " What is the main role of this concept in gradient descent?"
    return message


def _transition_to_interpreted(transition) -> dict:
    observables = transition.observable_signals
    return {
        "followup_type": transition.sampled_response_type,
        "checkpoint_correct": transition.checkpoint_correct,
        "checkpoint_score": observables["checkpoint_score"],
        "confusion_score": observables["confusion_score"],
        "comprehension_score": observables["comprehension_score"],
        "engagement_score": observables["engagement_score"],
        "progress_signal": observables["progress_signal"],
        "pace_fast_score": observables["pace_fast_score"],
        "pace_slow_score": observables["pace_slow_score"],
        "evidence": {
            "confusion_phrases": [],
            "understanding_phrases": [],
            "curiosity_phrases": [],
        },
    }


def _make_outcome(turn_index: int, user_id: str, action_id: str, interpreted: dict) -> Outcome:
    checkpoint_correct = interpreted["checkpoint_correct"]
    return Outcome(
        timestamp=turn_index,
        user_id=user_id,
        action_id=action_id,
        task_result=TaskResult(
            correct=None if checkpoint_correct is None else int(checkpoint_correct),
            latency_s=None,
            reread=None,
            completed=1,
            abandoned=0,
        ),
        semantic_signals=SemanticSignals(
            followup_text=interpreted["followup_type"],
            followup_type=interpreted["followup_type"],
            confusion_score=interpreted["confusion_score"],
            comprehension_score=interpreted["comprehension_score"],
            engagement_score=interpreted["engagement_score"],
            pace_fast_score=interpreted["pace_fast_score"],
            pace_slow_score=interpreted["pace_slow_score"],
        ),
        raw={"interpreted": interpreted},
    )


def summarize(rows: list[dict]) -> dict:
    return {
        "average_reward": sum(row["average_reward"] for row in rows) / len(rows),
        "total_oracle_mastery_gain": sum(row["total_oracle_mastery_gain"] for row in rows) / len(rows),
        "average_oracle_mastery_gain": sum(row["average_oracle_mastery_gain"] for row in rows) / len(rows),
        "checkpoint_accuracy": sum(row["checkpoint_accuracy"] for row in rows) / len(rows),
    }


def print_results(results: dict[str, list[dict]]) -> None:
    summaries = {label: summarize(rows) for label, rows in results.items()}
    for label, rows in results.items():
        summary = summaries[label]
        print(f"=== {label} ===")
        print(
            "avg_reward={average_reward:.3f} total_oracle_gain={total_oracle_mastery_gain:.3f} "
            "avg_oracle_gain={average_oracle_mastery_gain:.3f} checkpoint_acc={checkpoint_accuracy:.3f}".format(
                **summary
            )
        )
        for row in rows:
            top_action = max(row["action_counts"], key=row["action_counts"].get)
            print(
                f"  {row['user_id']}: reward={row['average_reward']:.3f} "
                f"oracle_gain={row['total_oracle_mastery_gain']:.3f} "
                f"checkpoint_acc={row['checkpoint_accuracy']:.3f} "
                f"top_action={top_action}"
            )
        print()

    print("=== Lift vs fixed_no_change ===")
    baseline = summaries["fixed_no_change"]
    for label in ["generic", "personalized", "random"]:
        print(
            f"{label}: reward_delta={summaries[label]['average_reward'] - baseline['average_reward']:+.3f} "
            f"oracle_gain_delta={summaries[label]['total_oracle_mastery_gain'] - baseline['total_oracle_mastery_gain']:+.3f}"
        )

    print("=== Lift: personalized - generic ===")
    print(
        f"reward_delta={summaries['personalized']['average_reward'] - summaries['generic']['average_reward']:+.3f} "
        f"oracle_gain_delta={summaries['personalized']['total_oracle_mastery_gain'] - summaries['generic']['total_oracle_mastery_gain']:+.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare policy modes on the hidden-knowledge student simulator.")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    print_results(run_comparison(turns=args.turns, seed=args.seed))


if __name__ == "__main__":
    main()
