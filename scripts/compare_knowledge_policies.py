from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import (  # noqa: E402
    BACKPROP_SCENARIO,
    DEFAULT_USER_TO_STEW_SUBJECT,
    GenericPolicyInput,
    GenericTutorPolicy,
    KnowledgeAgent,
    KnowledgeAgentProfile,
    PersonalizedTutorPolicy,
    build_eeg_provider,
    evaluate_knowledge_state,
)
from emotiv_learn.openai_client import OpenAIChatClient  # noqa: E402


DEFAULT_USERS = ["advanced_concise", "example_builder", "visual_scanner"]
DEFAULT_POLICIES = ["generic", "personalized"]

USER_PROFILES = {
    "advanced_concise": KnowledgeAgentProfile(
        user_id="advanced_concise",
        initial_knowledge_level=0.42,
        curiosity=0.70,
        confidence=0.54,
        fatigue=0.10,
        engagement=0.76,
        attention=0.78,
    ),
    "example_builder": KnowledgeAgentProfile(
        user_id="example_builder",
        initial_knowledge_level=0.34,
        curiosity=0.66,
        confidence=0.42,
        fatigue=0.14,
        engagement=0.72,
        attention=0.71,
    ),
    "visual_scanner": KnowledgeAgentProfile(
        user_id="visual_scanner",
        initial_knowledge_level=0.28,
        curiosity=0.60,
        confidence=0.38,
        fatigue=0.18,
        engagement=0.68,
        attention=0.64,
    ),
}


class JsonlEventWriter:
    def __init__(self, output_path: Path | None) -> None:
        self.output_path = output_path
        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text("", encoding="utf-8")

    def write(self, event_type: str, payload: dict) -> None:
        if self.output_path is None:
            return
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")


def run_comparison(
    *,
    max_steps: int,
    seed: int,
    model: str | None,
    output_path: Path,
    events_output_path: Path | None,
    users: list[str],
    policies: list[str],
    eeg_mode: str,
    stew_dir: str,
    stew_index_path: str | None,
    eeg_epoch_sec: int,
    eeg_stride_sec: int,
) -> dict:
    scenario = BACKPROP_SCENARIO
    client = OpenAIChatClient(model=model)
    event_writer = JsonlEventWriter(events_output_path)
    eeg_provider = build_eeg_provider(
        eeg_mode=eeg_mode,
        seed=seed,
        stew_dir=stew_dir,
        index_path=stew_index_path,
        user_to_subject=DEFAULT_USER_TO_STEW_SUBJECT,
        epoch_sec=eeg_epoch_sec,
        stride_sec=eeg_stride_sec,
    )
    generic_policy = GenericTutorPolicy(client)

    results: dict[str, dict] = {}
    event_writer.write(
        "knowledge_experiment_started",
        {
            "scenario": scenario.topic_id,
            "max_steps": max_steps,
            "seed": seed,
            "users": users,
            "policies": policies,
            "eeg_mode": eeg_mode,
        },
    )

    for policy_name in policies:
        policy_results: list[dict] = []
        event_writer.write("policy_started", {"policy_name": policy_name})
        for user_index, user_id in enumerate(users):
            profile = USER_PROFILES[user_id]
            agent = KnowledgeAgent(scenario, profile, seed=seed + user_index * 1000)
            personalized_policy = None
            if policy_name == "personalized":
                personalized_policy = PersonalizedTutorPolicy(
                    client=client,
                    eeg_provider=eeg_provider,
                    seed=seed + user_index * 1000,
                )

            user_result = run_user_episode(
                policy_name=policy_name,
                scenario=scenario,
                agent=agent,
                generic_policy=generic_policy,
                personalized_policy=personalized_policy,
                max_steps=max_steps,
                event_writer=event_writer,
            )
            policy_results.append(user_result)
            event_writer.write(
                "user_completed",
                {
                    "policy_name": policy_name,
                    "user_id": user_id,
                    "steps_taken": user_result["steps_taken"],
                    "steps_to_goal": user_result["steps_to_goal"],
                    "final_goal_coverage": user_result["final_evaluation"]["goal_coverage_score"],
                    "final_knowledge_quality": user_result["final_evaluation"]["knowledge_quality_score"],
                },
            )

        results[policy_name] = {
            "users": policy_results,
            "summary": summarize_policy(policy_results),
        }
        event_writer.write(
            "policy_completed",
            {"policy_name": policy_name, "summary": results[policy_name]["summary"]},
        )

    output = {
        "scenario": scenario.topic_id,
        "max_steps": max_steps,
        "seed": seed,
        "model": model,
        "eeg_mode": eeg_mode,
        "results": results,
        "comparison": compare_policy_summaries(results),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    event_writer.write(
        "knowledge_experiment_completed",
        {"output_path": str(output_path), "comparison": output["comparison"]},
    )
    print_summary(output)
    return output


def run_user_episode(
    *,
    policy_name: str,
    scenario,
    agent: KnowledgeAgent,
    generic_policy: GenericTutorPolicy,
    personalized_policy: PersonalizedTutorPolicy | None,
    max_steps: int,
    event_writer: JsonlEventWriter,
) -> dict:
    step_logs: list[dict] = []
    learner_message: str | None = None
    steps_to_goal: int | None = None

    for step_index in range(1, max_steps + 1):
        concept_id = agent.current_concept_id()
        checkpoint_expected = step_index % 2 == 0

        if policy_name == "generic":
            action_id = "generic_standard"
            tutor_message = generic_policy.next_tutor_step(
                GenericPolicyInput(
                    scenario=scenario,
                    concept_id=concept_id,
                    learner_state=agent.state,
                    learner_message=learner_message,
                    checkpoint_expected=checkpoint_expected,
                )
            )
            policy_metadata = {}
        else:
            assert personalized_policy is not None
            action_id, tutor_message = personalized_policy.next_action_and_message(
                timestamp=step_index,
                user_id=agent.profile.user_id,
                scenario=scenario,
                concept_id=concept_id,
                learner_state=agent.state,
                learner_message=learner_message,
                checkpoint_expected=checkpoint_expected,
            )
            policy_metadata = {"action_id": action_id}

        knowledge_turn = agent.consume_tutor_step(
            concept_id=concept_id,
            tutor_message=tutor_message,
            action_id=action_id,
            checkpoint_expected=checkpoint_expected,
        )

        personalized_turn = None
        if personalized_policy is not None:
            personalized_turn = personalized_policy.observe_turn(
                timestamp=step_index,
                user_id=agent.profile.user_id,
                scenario=scenario,
                concept_id=concept_id,
                action_id=action_id,
                tutor_message=tutor_message,
                knowledge_turn=knowledge_turn,
            )
            policy_metadata |= {
                "interpreted": personalized_turn.interpreted,
                "reward": personalized_turn.reward,
                "eeg": {
                    "feature_names": personalized_turn.eeg_window.feature_names,
                    "features": personalized_turn.eeg_window.features,
                    "metadata": personalized_turn.eeg_window.metadata,
                },
            }

        evaluation_before_advance = evaluate_knowledge_state(scenario, agent.state)
        step_logs.append(
            {
                "step_index": step_index,
                "concept_id": concept_id,
                "checkpoint_expected": checkpoint_expected,
                "tutor_message": tutor_message,
                "knowledge_turn": knowledge_turn.to_dict(),
                "evaluation": asdict(evaluation_before_advance),
                "policy_metadata": policy_metadata,
            }
        )
        event_writer.write(
            "knowledge_step_completed",
            {
                "policy_name": policy_name,
                "user_id": agent.profile.user_id,
                "step_index": step_index,
                "concept_id": concept_id,
                "checkpoint_expected": checkpoint_expected,
                "action_id": action_id,
                "tutor_message": tutor_message,
                "knowledge_turn": knowledge_turn.to_dict(),
                "evaluation": asdict(evaluation_before_advance),
                "policy_metadata": policy_metadata,
                "knowledge_base": list(agent.state.knowledge_base),
                "goal_coverage": evaluation_before_advance.goal_coverage_score,
                "knowledge_quality": evaluation_before_advance.knowledge_quality_score,
                "goal_reached": evaluation_before_advance.goal_reached,
                "response_type": knowledge_turn.response_type,
            },
        )

        if evaluation_before_advance.goal_reached and steps_to_goal is None:
            steps_to_goal = step_index
            break

        agent.advance_if_ready(concept_id)
        learner_message = knowledge_turn.reprompt

    final_evaluation = evaluate_knowledge_state(scenario, agent.state)
    return {
        "user_id": agent.profile.user_id,
        "subject_id": DEFAULT_USER_TO_STEW_SUBJECT.get(agent.profile.user_id, agent.profile.user_id),
        "steps_taken": len(step_logs),
        "steps_to_goal": steps_to_goal,
        "goal_reached_within_budget": final_evaluation.goal_reached,
        "initial_knowledge_base": list(scenario.initial_knowledge_base),
        "final_knowledge_base": list(agent.state.knowledge_base),
        "final_state": asdict(agent.state),
        "final_evaluation": asdict(final_evaluation),
        "step_logs": step_logs,
    }


def summarize_policy(user_rows: list[dict]) -> dict:
    steps_to_goal_values = [row["steps_to_goal"] for row in user_rows if row["steps_to_goal"] is not None]
    final_coverages = [row["final_evaluation"]["goal_coverage_score"] for row in user_rows]
    final_qualities = [row["final_evaluation"]["knowledge_quality_score"] for row in user_rows]
    return {
        "users_run": len(user_rows),
        "goals_reached": sum(1 for row in user_rows if row["goal_reached_within_budget"]),
        "avg_final_goal_coverage": round(sum(final_coverages) / max(len(final_coverages), 1), 4),
        "avg_final_knowledge_quality": round(sum(final_qualities) / max(len(final_qualities), 1), 4),
        "avg_steps_taken": round(sum(row["steps_taken"] for row in user_rows) / max(len(user_rows), 1), 4),
        "avg_steps_to_goal": round(sum(steps_to_goal_values) / len(steps_to_goal_values), 4) if steps_to_goal_values else None,
    }


def compare_policy_summaries(results: dict[str, dict]) -> dict:
    generic = results.get("generic", {}).get("summary")
    personalized = results.get("personalized", {}).get("summary")
    if not generic or not personalized:
        return {}

    efficiency_delta = None
    if generic["avg_steps_to_goal"] is not None and personalized["avg_steps_to_goal"] is not None:
        efficiency_delta = round(generic["avg_steps_to_goal"] - personalized["avg_steps_to_goal"], 4)

    return {
        "personalized_minus_generic_goal_coverage": round(
            personalized["avg_final_goal_coverage"] - generic["avg_final_goal_coverage"], 4
        ),
        "personalized_minus_generic_knowledge_quality": round(
            personalized["avg_final_knowledge_quality"] - generic["avg_final_knowledge_quality"], 4
        ),
        "generic_minus_personalized_steps_to_goal": efficiency_delta,
    }


def print_summary(output: dict) -> None:
    print("=== Knowledge Policy Comparison ===")
    print(f"Scenario: {output['scenario']}")
    print(f"Max steps: {output['max_steps']}")
    for policy_name, policy_rows in output["results"].items():
        summary = policy_rows["summary"]
        print(
            f"{policy_name}: coverage={summary['avg_final_goal_coverage']:.4f}, "
            f"quality={summary['avg_final_knowledge_quality']:.4f}, "
            f"steps_to_goal={summary['avg_steps_to_goal']}"
        )
    if output["comparison"]:
        print(f"Comparison deltas: {output['comparison']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare generic vs personalized knowledge-learning policies.")
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/knowledge_policy_comparison.json"))
    parser.add_argument("--events-output", type=Path, default=None)
    parser.add_argument("--users", type=lambda value: [part.strip() for part in value.split(",") if part.strip()], default=DEFAULT_USERS)
    parser.add_argument("--policies", type=lambda value: [part.strip() for part in value.split(",") if part.strip()], default=DEFAULT_POLICIES)
    parser.add_argument("--eeg-mode", type=str, default="matched_real", choices=["matched_real", "synthetic"])
    parser.add_argument("--stew-dir", type=str, default="stew_dataset")
    parser.add_argument("--stew-index-path", type=str, default="artifacts/stew_feature_index.json")
    parser.add_argument("--eeg-epoch-sec", type=int, default=30)
    parser.add_argument("--eeg-stride-sec", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    unknown_users = [user_id for user_id in args.users if user_id not in USER_PROFILES]
    if unknown_users:
        raise ValueError(f"Unknown users requested: {unknown_users}")
    run_comparison(
        max_steps=args.max_steps,
        seed=args.seed,
        model=args.model,
        output_path=args.output,
        events_output_path=args.events_output,
        users=args.users,
        policies=args.policies,
        eeg_mode=args.eeg_mode,
        stew_dir=args.stew_dir,
        stew_index_path=args.stew_index_path,
        eeg_epoch_sec=args.eeg_epoch_sec,
        eeg_stride_sec=args.eeg_stride_sec,
    )


if __name__ == "__main__":
    main()
