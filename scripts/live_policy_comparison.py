from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import ACTION_BANK, DecisionEngine
from emotiv_learn.eeg import EEGObservationContext, SyntheticEEGProvider, estimate_time_on_chunk
from emotiv_learn.live_training import LIVE_FEATURE_NAMES, LiveLLMStateBuilder, LiveStateInput
from emotiv_learn.llm_contracts import (
    StudentPromptInput,
    TutorPromptInput,
    build_student_messages,
    build_tutor_messages,
    compute_reward_from_interpreted,
    normalize_student_output,
)
from emotiv_learn.openai_client import OpenAIChatClient
from emotiv_learn.schemas import Outcome, RewardEvent, SemanticSignals, TaskResult
from emotiv_learn.student_model import HiddenKnowledgeStudent
from scripts.knowledge_policy_comparison import (
    CONTENT_STEPS,
    USER_PROFILES,
    _initial_state_for_user,
)


POLICY_MODES = ["personalized", "generic", "fixed_no_change", "random"]

LEARNER_PROFILE = {
    "example_preference": 0.8,
    "challenge_preference": 0.4,
    "structure_preference": 0.75,
    "verbosity_tolerance": 0.45,
}


def run_live_policy_comparison(
    *,
    turns: int,
    seed: int,
    model: str | None,
    output_path: Path,
    events_output_path: Path | None = None,
    policy_modes: list[str] | None = None,
    user_ids: list[str] | None = None,
) -> dict:
    client = OpenAIChatClient(model=model)
    selected_modes = policy_modes or POLICY_MODES
    selected_users = user_ids or list(USER_PROFILES)
    event_writer = JsonlEventWriter(events_output_path)
    results: dict[str, dict] = {}
    event_writer.write(
        "experiment_started",
        {
            "turns": turns,
            "seed": seed,
            "model": model,
            "policy_modes": selected_modes,
            "user_ids": selected_users,
        },
    )

    for policy_mode in selected_modes:
        print(f"=== Running {policy_mode} ===")
        event_writer.write("policy_mode_started", {"policy_mode": policy_mode})
        results[policy_mode] = run_policy_mode_live(
            policy_mode=policy_mode,
            turns=turns,
            seed=seed,
            user_ids=selected_users,
            client=client,
            event_writer=event_writer,
        )
        event_writer.write(
            "policy_mode_completed",
            {
                "policy_mode": policy_mode,
                "summary": summarize_policy(results[policy_mode]["users"]),
            },
        )

    output = {
        "turns": turns,
        "seed": seed,
        "model": model,
        "results": results,
        "summary": {policy_mode: summarize_policy(rows["users"]) for policy_mode, rows in results.items()},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    event_writer.write(
        "experiment_completed",
        {
            "output_path": str(output_path),
            "summary": output["summary"],
        },
    )
    print(f"Wrote live policy comparison to {output_path}")
    print_results(output["summary"], results)
    return output


def run_policy_mode_live(
    *,
    policy_mode: str,
    turns: int,
    seed: int,
    user_ids: list[str],
    client: OpenAIChatClient,
    event_writer: JsonlEventWriter,
) -> dict:
    rng = random.Random(seed)
    state_builder = LiveLLMStateBuilder()
    eeg_provider = SyntheticEEGProvider(seed=seed)
    students = {
        user_id: HiddenKnowledgeStudent(_initial_state_for_user(user_id), seed=seed + index * 101)
        for index, user_id in enumerate(user_ids)
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

    trackers = {user_id: _new_tracker() for user_id in user_ids}
    turn_logs: list[dict] = []

    for turn_index in range(1, turns + 1):
        concept_id, checkpoint_expected = CONTENT_STEPS[(turn_index - 1) % len(CONTENT_STEPS)]
        for user_id, student in students.items():
            tracker = trackers[user_id]
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
                    eeg_window=tracker["previous_eeg_window"],
                )
            )
            action_id = _select_action(policy_mode, engine, state, rng)
            event_writer.write(
                "turn_started",
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "concept_id": concept_id,
                    "checkpoint_expected": checkpoint_expected,
                    "action_id": action_id,
                    "state": asdict(state),
                },
            )
            tutor_message = _generate_tutor_message(
                client=client,
                concept_id=concept_id,
                action_id=action_id,
                checkpoint_expected=checkpoint_expected,
                tracker=tracker,
            )
            event_writer.write(
                "tutor_message",
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "action_id": action_id,
                    "tutor_message": tutor_message,
                },
            )
            transition = student.step(
                concept_id=concept_id,
                action_id=action_id,
                tutor_message=tutor_message,
                checkpoint_expected=checkpoint_expected,
            )
            event_writer.write(
                "student_transition",
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "action_id": action_id,
                    "student_transition": transition.to_dict(),
                },
            )
            interpreted = _transition_to_interpreted(transition)
            student_response = _generate_student_response(
                client=client,
                transition=transition,
                tutor_message=tutor_message,
                checkpoint_expected=checkpoint_expected,
            )
            interpreted["followup_type"] = transition.sampled_response_type
            reward = compute_reward_from_interpreted(interpreted)
            time_on_chunk = estimate_time_on_chunk(tutor_message)
            eeg_window = eeg_provider.observe(
                EEGObservationContext(
                    timestamp=turn_index,
                    user_id=user_id,
                    concept_id=concept_id,
                    action_id=action_id,
                    tutor_message=tutor_message,
                    time_on_chunk=time_on_chunk,
                    hidden_state=transition.to_dict()["hidden_state_after"],
                    observable_signals=transition.observable_signals,
                )
            )

            if engine is not None:
                outcome = _make_outcome(turn_index, user_id, action_id, interpreted, student_response, tutor_message)
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
                update_trace = asdict(engine.update_history[-1])
            else:
                update_trace = None

            _update_tracker(
                tracker=tracker,
                action_id=action_id,
                transition=transition,
                interpreted=interpreted,
                student_response=student_response,
                reward=reward,
                eeg_window=eeg_window,
            )
            turn_logs.append(
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "state": asdict(state),
                    "action_id": action_id,
                    "tutor_message": tutor_message,
                    "student_response": student_response,
                    "interpreted": interpreted,
                    "eeg_window": asdict(eeg_window),
                    "reward": reward,
                    "student_transition": transition.to_dict(),
                    "update_trace": update_trace,
                }
            )
            event_writer.write(
                "turn_completed",
                {
                    "policy_mode": policy_mode,
                    "turn_index": turn_index,
                    "user_id": user_id,
                    "concept_id": concept_id,
                    "checkpoint_expected": checkpoint_expected,
                    "action_id": action_id,
                    "tutor_message": tutor_message,
                    "student_response": student_response,
                    "interpreted": interpreted,
                    "reward": reward,
                    "oracle_mastery_gain": transition.oracle_mastery_gain,
                    "response_type": transition.sampled_response_type,
                    "checkpoint_correct": transition.checkpoint_correct,
                    "student_transition": transition.to_dict(),
                    "update_trace": update_trace,
                },
            )
            print(
                f"{policy_mode} turn={turn_index} user={user_id} action={action_id} "
                f"followup={transition.sampled_response_type} reward={reward:.3f} "
                f"oracle_gain={transition.oracle_mastery_gain:.3f}"
            )

    return {
        "users": [_summarize_user(user_id, trackers[user_id], turns) for user_id in user_ids],
        "turn_logs": turn_logs,
    }


def _new_tracker() -> dict:
    return {
        "previous_interpreted": None,
        "previous_student_response": None,
        "previous_reward": 0.0,
        "previous_eeg_window": None,
        "conversation_summary": "The learner is beginning a short lesson.",
        "total_reward": 0.0,
        "total_oracle_gain": 0.0,
        "checkpoint_correct": 0,
        "checkpoint_count": 0,
        "followups": Counter(),
        "action_counts": Counter(),
    }


class JsonlEventWriter:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.event_index = 0
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def write(self, event_type: str, payload: dict) -> None:
        if self.path is None:
            return
        self.event_index += 1
        event = {
            "event_id": f"evt_{self.event_index:06d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


def _generate_tutor_message(
    *,
    client: OpenAIChatClient,
    concept_id: str,
    action_id: str,
    checkpoint_expected: bool,
    tracker: dict,
) -> str:
    messages = build_tutor_messages(
        TutorPromptInput(
            topic="gradient descent",
            concept_id=concept_id,
            conversation_summary=tracker["conversation_summary"],
            load_level=_load_level(tracker["previous_interpreted"]),
            behavior_summary=_behavior_summary(tracker["previous_interpreted"]),
            last_followup_type=(tracker["previous_interpreted"] or {}).get("followup_type", "unknown"),
            action_id=action_id,
            length_target="short",
            difficulty_target="medium",
            include_checkpoint=checkpoint_expected,
        )
    )
    return client.complete_text(messages, max_tokens=700, temperature=0.35)


def _generate_student_response(
    *,
    client: OpenAIChatClient,
    transition,
    tutor_message: str,
    checkpoint_expected: bool,
) -> dict:
    messages = build_student_messages(
        StudentPromptInput(
            learner_profile=LEARNER_PROFILE,
            hidden_state=transition.to_dict()["hidden_state_after"],
            observable_signals=transition.observable_signals,
            tutor_message=tutor_message,
            checkpoint_expected=checkpoint_expected,
            sampled_response_type=transition.sampled_response_type,
            checkpoint_answer=transition.checkpoint_answer,
        )
    )
    student_response = normalize_student_output(client.complete_json(messages, max_tokens=500, temperature=0.45))
    student_response["response_type"] = transition.sampled_response_type
    if transition.checkpoint_answer is not None:
        student_response["checkpoint_answer"] = transition.checkpoint_answer
    return student_response


def _update_tracker(
    *,
    tracker: dict,
    action_id: str,
    transition,
    interpreted: dict,
    student_response: dict,
    reward: float,
    eeg_window,
) -> None:
    tracker["total_reward"] += reward
    tracker["total_oracle_gain"] += transition.oracle_mastery_gain
    tracker["action_counts"][action_id] += 1
    tracker["followups"][transition.sampled_response_type] += 1
    if transition.checkpoint_correct is not None:
        tracker["checkpoint_count"] += 1
        tracker["checkpoint_correct"] += int(transition.checkpoint_correct)
    tracker["previous_interpreted"] = interpreted
    tracker["previous_student_response"] = student_response
    tracker["previous_reward"] = reward
    tracker["previous_eeg_window"] = eeg_window
    tracker["conversation_summary"] = (
        f"Previous action {action_id}; student responded {interpreted['followup_type']} "
        f"with reward {reward:.2f} and oracle mastery gain {transition.oracle_mastery_gain:.3f}."
    )


def _select_action(policy_mode: str, engine: DecisionEngine | None, state, rng: random.Random) -> str:
    if policy_mode == "fixed_no_change":
        return "no_change"
    if policy_mode == "random":
        return rng.choice([action.action_id for action in ACTION_BANK])
    if engine is None:
        raise ValueError(f"policy mode requires engine: {policy_mode}")
    action_scores = engine.score_actions(state, ACTION_BANK)
    return engine.select_action(action_scores).action_id


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


def _make_outcome(
    turn_index: int,
    user_id: str,
    action_id: str,
    interpreted: dict,
    student_response: dict,
    tutor_message: str,
 ) -> Outcome:
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
            followup_text=student_response.get("student_message"),
            followup_type=interpreted["followup_type"],
            confusion_score=interpreted["confusion_score"],
            comprehension_score=interpreted["comprehension_score"],
            engagement_score=interpreted["engagement_score"],
            pace_fast_score=interpreted["pace_fast_score"],
            pace_slow_score=interpreted["pace_slow_score"],
        ),
        raw={
            "student_response": student_response,
            "interpreted": interpreted,
            "tutor_message": tutor_message,
        },
    )


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


def summarize_policy(rows: list[dict]) -> dict:
    return {
        "average_reward": sum(row["average_reward"] for row in rows) / len(rows),
        "total_oracle_mastery_gain": sum(row["total_oracle_mastery_gain"] for row in rows) / len(rows),
        "average_oracle_mastery_gain": sum(row["average_oracle_mastery_gain"] for row in rows) / len(rows),
        "checkpoint_accuracy": sum(row["checkpoint_accuracy"] for row in rows) / len(rows),
    }


def print_results(summary: dict[str, dict], results: dict[str, dict]) -> None:
    for policy_mode, policy_summary in summary.items():
        print(f"=== {policy_mode} ===")
        print(
            "avg_reward={average_reward:.3f} total_oracle_gain={total_oracle_mastery_gain:.3f} "
            "avg_oracle_gain={average_oracle_mastery_gain:.3f} checkpoint_acc={checkpoint_accuracy:.3f}".format(
                **policy_summary
            )
        )
        for row in results[policy_mode]["users"]:
            top_action = max(row["action_counts"], key=row["action_counts"].get)
            print(
                f"  {row['user_id']}: reward={row['average_reward']:.3f} "
                f"oracle_gain={row['total_oracle_mastery_gain']:.3f} "
                f"checkpoint_acc={row['checkpoint_accuracy']:.3f} "
                f"top_action={top_action}"
            )
        print()

    if "fixed_no_change" in summary:
        baseline = summary["fixed_no_change"]
        print("=== Lift vs fixed_no_change ===")
        for policy_mode in summary:
            if policy_mode == "fixed_no_change":
                continue
            print(
                f"{policy_mode}: reward_delta={summary[policy_mode]['average_reward'] - baseline['average_reward']:+.3f} "
                f"oracle_gain_delta={summary[policy_mode]['total_oracle_mastery_gain'] - baseline['total_oracle_mastery_gain']:+.3f}"
            )

    if "personalized" in summary and "generic" in summary:
        print("=== Lift: personalized - generic ===")
        print(
            f"reward_delta={summary['personalized']['average_reward'] - summary['generic']['average_reward']:+.3f} "
            f"oracle_gain_delta={summary['personalized']['total_oracle_mastery_gain'] - summary['generic']['total_oracle_mastery_gain']:+.3f}"
        )


def _load_level(interpreted: dict | None) -> str:
    if not interpreted:
        return "unknown"
    confusion = interpreted.get("confusion_score", 0.5)
    if confusion >= 0.65:
        return "high"
    if confusion >= 0.35:
        return "medium"
    return "low"


def _behavior_summary(interpreted: dict | None) -> str:
    if not interpreted:
        return "no prior behavioral signal"
    return (
        f"confusion={interpreted['confusion_score']:.2f}, "
        f"comprehension={interpreted['comprehension_score']:.2f}, "
        f"engagement={interpreted['engagement_score']:.2f}"
    )


def _parse_csv(value: str | None, allowed: set[str]) -> list[str] | None:
    if not value:
        return None
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in parsed if item not in allowed]
    if unknown:
        raise ValueError(f"unknown values {unknown}; allowed values are {sorted(allowed)}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live LLM policy comparison on hidden-knowledge students.")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", default=None)
    parser.add_argument("--modes", default="personalized,generic,fixed_no_change,random")
    parser.add_argument("--users", default=",".join(USER_PROFILES))
    parser.add_argument("--output", default="artifacts/live_policy_comparison.json")
    parser.add_argument("--events-output", default=None)
    args = parser.parse_args()

    run_live_policy_comparison(
        turns=args.turns,
        seed=args.seed,
        model=args.model,
        output_path=Path(args.output),
        events_output_path=Path(args.events_output) if args.events_output else None,
        policy_modes=_parse_csv(args.modes, set(POLICY_MODES)),
        user_ids=_parse_csv(args.users, set(USER_PROFILES)),
    )


if __name__ == "__main__":
    main()
