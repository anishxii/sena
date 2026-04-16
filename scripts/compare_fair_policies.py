from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.decision_engine import DecisionEngine  # noqa: E402
from emotiv_learn.knowledge_agent import KnowledgeAgent, KnowledgeAgentProfile, KnowledgeTurn  # noqa: E402
from emotiv_learn.knowledge_evaluator import evaluate_knowledge_state  # noqa: E402
from emotiv_learn.knowledge_scenarios import BACKPROP_SCENARIO, TopicScenario  # noqa: E402
from emotiv_learn.live_training import LIVE_FEATURE_NAMES, LiveLLMStateBuilder, LiveStateInput  # noqa: E402
from emotiv_learn.llm_contracts import ACTION_INSTRUCTIONS, TutorPromptInput, build_tutor_messages  # noqa: E402
from emotiv_learn.openai_client import OpenAIChatClient  # noqa: E402
from emotiv_learn.reward_model import compute_observable_learning_reward  # noqa: E402
from emotiv_learn.schemas import ACTION_BANK, Outcome, RewardEvent, SemanticSignals, TaskResult  # noqa: E402


GENERIC_SYSTEM_PROMPT = """You are a clear machine-learning tutor.

Teach the current concept in a standard, non-personalized way.
Do not infer hidden learner state. Do not mention EEG, reward, policy, personalization, or simulation.
If checkpoint is requested, end with one short comprehension question.
Return only the tutor message."""


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare fair generic vs observable-only personalized policy loops.")
    parser.add_argument("--turns", type=int, default=8)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--model", default=None)
    parser.add_argument("--users", default="advanced_concise,example_builder,visual_scanner")
    parser.add_argument("--output", type=Path, default=Path("artifacts/fair_policy_comparison.json"))
    args = parser.parse_args()

    users = [part.strip() for part in args.users.split(",") if part.strip()]
    client = OpenAIChatClient(model=args.model)
    scenario = BACKPROP_SCENARIO
    results = {
        "generic": [_run_episode("generic", scenario, USER_PROFILES[user_id], client, args.turns, args.seed) for user_id in users],
        "personalized": [
            _run_episode("personalized", scenario, USER_PROFILES[user_id], client, args.turns, args.seed)
            for user_id in users
        ],
    }
    output = {
        "turns": args.turns,
        "seed": args.seed,
        "users": users,
        "summary": {policy: _summarize(rows) for policy, rows in results.items()},
        "comparison": _compare(results),
        "results": results,
        "reward_definition": "observable_learning_reward_v1",
        "state_contract": "personalized policy observes previous interpreted response/confidence/reward/context only; hidden learner state is not included",
        "generic_contract": "generic tutor receives only topic, concept, prior learner message, and checkpoint flag; no mastery/confusion/fatigue/attention/knowledge preview",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"summary": output["summary"], "comparison": output["comparison"]}, indent=2))
    print(f"Wrote {args.output}")


def _run_episode(
    policy_name: str,
    scenario: TopicScenario,
    profile: KnowledgeAgentProfile,
    client: OpenAIChatClient,
    turns: int,
    seed: int,
) -> dict:
    agent = KnowledgeAgent(scenario=scenario, profile=profile, seed=seed + _stable_offset(profile.user_id))
    state_builder = LiveLLMStateBuilder()
    engine = DecisionEngine(
        feature_dim=len(LIVE_FEATURE_NAMES),
        epsilon=0.18,
        seed=seed + _stable_offset(profile.user_id),
        reward_clip_abs=1.5,
        update_clip_abs=0.2,
        l2_weight_decay=0.001,
    )
    learner_message = "I am ready to learn the next part."
    previous_interpreted: dict | None = None
    previous_student_response: dict | None = None
    previous_reward = 0.0
    logs = []

    for turn_index in range(1, turns + 1):
        concept_id = agent.current_concept_id()
        concept = scenario.concept(concept_id)
        checkpoint_expected = turn_index % 2 == 0
        state = None
        action_scores = None

        if policy_name == "generic":
            action_id = "generic_standard"
            tutor_message = _generic_tutor_message(
                client=client,
                scenario=scenario,
                concept_id=concept_id,
                learner_message=learner_message,
                checkpoint_expected=checkpoint_expected,
            )
        else:
            state = state_builder.build_state(
                LiveStateInput(
                    timestamp=turn_index,
                    user_id=profile.user_id,
                    topic_id=scenario.topic_id,
                    task_type="learn",
                    difficulty="medium",
                    turn_index=turn_index,
                    max_turns=turns,
                    interpreted=previous_interpreted,
                    student_response=previous_student_response,
                    previous_reward=previous_reward,
                )
            )
            action_scores = engine.score_actions(state, ACTION_BANK)
            action = engine.select_action(action_scores)
            action_id = action.action_id
            tutor_message = client.complete_text(
                build_tutor_messages(
                    TutorPromptInput(
                        topic=scenario.title,
                        concept_id=concept_id,
                        conversation_summary=learner_message,
                        load_level=_observable_load_level(previous_interpreted),
                        behavior_summary=_observable_behavior_summary(previous_interpreted, previous_student_response),
                        last_followup_type=(previous_interpreted or {}).get("followup_type", "unknown"),
                        action_id=action_id,
                        length_target="short",
                        difficulty_target="medium",
                        include_checkpoint=checkpoint_expected,
                    )
                ),
                max_tokens=650,
                temperature=0.35,
            )

        before = agent.state
        turn = agent.consume_tutor_step(
            concept_id=concept_id,
            tutor_message=tutor_message,
            action_id=action_id,
            checkpoint_expected=checkpoint_expected,
        )
        interpreted = _observable_interpreted(turn, previous_student_response, checkpoint_expected=checkpoint_expected)
        reward = compute_observable_learning_reward(interpreted)

        if policy_name == "personalized" and state is not None:
            assert action_scores is not None
            engine.update(
                RewardEvent(
                    timestamp=turn_index,
                    user_id=profile.user_id,
                    state_features=state.features,
                    action_id=action_id,
                    reward=reward,
                    outcome=_make_outcome(turn_index, profile.user_id, action_id, interpreted, turn),
                )
            )

        evaluation = evaluate_knowledge_state(scenario, agent.state)
        logs.append(
            {
                "turn_index": turn_index,
                "concept_id": concept_id,
                "concept_label": concept.label,
                "policy": policy_name,
                "action_id": action_id,
                "checkpoint_expected": checkpoint_expected,
                "tutor_message": tutor_message,
                "observable_interpreted": interpreted,
                "reward": reward,
                "learner_response_type": turn.response_type,
                "learner_reprompt": turn.reprompt,
                "checkpoint_correct": turn.checkpoint_correct,
                "hidden_progress_signal_for_eval_only": turn.progress_signal,
                "state_before_eval_only": asdict(before),
                "state_after_eval_only": asdict(agent.state),
                "evaluation": asdict(evaluation),
                "action_scores": asdict(action_scores) if action_scores is not None else None,
                "update_trace": asdict(engine.update_history[-1]) if policy_name == "personalized" else None,
            }
        )
        previous_interpreted = interpreted
        previous_student_response = {
            "response_type": turn.response_type,
            "self_reported_confidence": turn.self_reported_confidence,
        }
        previous_reward = reward
        learner_message = turn.reprompt or "Please continue."
        agent.advance_if_ready(concept_id, checkpoint_correct=turn.checkpoint_correct)

    final_evaluation = evaluate_knowledge_state(scenario, agent.state)
    return {
        "user_id": profile.user_id,
        "steps_taken": len(logs),
        "final_state_eval_only": asdict(agent.state),
        "final_evaluation": asdict(final_evaluation),
        "turn_logs": logs,
    }


def _generic_tutor_message(
    *,
    client: OpenAIChatClient,
    scenario: TopicScenario,
    concept_id: str,
    learner_message: str,
    checkpoint_expected: bool,
) -> str:
    concept = scenario.concept(concept_id)
    prompt = f"""Topic: {scenario.title}
Current concept: {concept.label}
Learning goal: {concept.canonical_claim}
Learner's last visible message: {learner_message}
Checkpoint requested: {checkpoint_expected}

Teach this concept in a standard way. Do not use any private learner-state estimate."""
    return client.complete_text(
        [
            {"role": "system", "content": GENERIC_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=650,
        temperature=0.35,
    )


def _observable_interpreted(
    turn: KnowledgeTurn,
    previous_student_response: dict | None,
    *,
    checkpoint_expected: bool,
) -> dict:
    confidence = float(turn.self_reported_confidence)
    previous_confidence = 0.5 if previous_student_response is None else float(previous_student_response.get("self_reported_confidence", 0.5))
    confidence_delta = max(0.0, min(1.0, (confidence - previous_confidence + 0.5)))
    response_type = turn.response_type
    confusion_score = {
        "continue": max(0.0, 0.35 - 0.20 * confidence),
        "branch": max(0.0, 0.28 - 0.12 * confidence),
        "clarify": min(1.0, 0.55 + 0.35 * (1.0 - confidence)),
    }[response_type]
    engagement_score = {
        "continue": 0.62,
        "branch": 0.82,
        "clarify": 0.50,
    }[response_type]
    return {
        "followup_type": response_type,
        "checkpoint_expected": checkpoint_expected,
        "checkpoint_attempted": turn.checkpoint_correct is not None,
        "checkpoint_correct": turn.checkpoint_correct,
        "checkpoint_score": 1.0 if turn.checkpoint_correct is True else 0.0 if turn.checkpoint_correct is False else None,
        "confusion_score": round(confusion_score, 4),
        "comprehension_score": round(confidence, 4),
        "engagement_score": round(engagement_score, 4),
        "progress_signal": round(confidence_delta, 4),
        "pace_fast_score": 0.0,
        "pace_slow_score": 0.35 if response_type == "clarify" else 0.0,
        "evidence": {
            "confusion_phrases": [turn.reprompt] if response_type == "clarify" and turn.reprompt else [],
            "understanding_phrases": [turn.reprompt] if response_type == "continue" and turn.reprompt else [],
            "curiosity_phrases": [turn.reprompt] if response_type == "branch" and turn.reprompt else [],
        },
    }


def _make_outcome(turn_index: int, user_id: str, action_id: str, interpreted: dict, turn: KnowledgeTurn) -> Outcome:
    return Outcome(
        timestamp=turn_index,
        user_id=user_id,
        action_id=action_id,
        task_result=TaskResult(
            correct=int(bool(turn.checkpoint_correct)) if turn.checkpoint_correct is not None else None,
            latency_s=None,
            reread=None,
            completed=1,
            abandoned=0,
        ),
        semantic_signals=SemanticSignals(
            followup_text=turn.reprompt,
            followup_type=interpreted["followup_type"],
            confusion_score=interpreted["confusion_score"],
            comprehension_score=interpreted["comprehension_score"],
            engagement_score=interpreted["engagement_score"],
            pace_fast_score=interpreted["pace_fast_score"],
            pace_slow_score=interpreted["pace_slow_score"],
        ),
        raw={"observable_interpreted": interpreted},
    )


def _observable_load_level(interpreted: dict | None) -> str:
    if interpreted is None:
        return "medium"
    if interpreted["confusion_score"] >= 0.65:
        return "high"
    if interpreted["confusion_score"] <= 0.25:
        return "low"
    return "medium"


def _observable_behavior_summary(interpreted: dict | None, student_response: dict | None) -> str:
    if interpreted is None:
        return "no previous observable learner response"
    return (
        f"last_response={interpreted['followup_type']}, "
        f"confidence={(student_response or {}).get('self_reported_confidence', 0.5):.2f}, "
        f"confusion={interpreted['confusion_score']:.2f}, "
        f"comprehension={interpreted['comprehension_score']:.2f}"
    )


def _summarize(rows: list[dict]) -> dict:
    qualities = [row["final_evaluation"]["knowledge_quality_score"] for row in rows]
    coverages = [row["final_evaluation"]["goal_coverage_score"] for row in rows]
    rewards = [turn["reward"] for row in rows for turn in row["turn_logs"]]
    return {
        "users_run": len(rows),
        "avg_final_knowledge_quality": round(sum(qualities) / max(len(qualities), 1), 4),
        "avg_final_goal_coverage": round(sum(coverages) / max(len(coverages), 1), 4),
        "avg_reward": round(sum(rewards) / max(len(rewards), 1), 4),
        "avg_steps_taken": round(sum(row["steps_taken"] for row in rows) / max(len(rows), 1), 4),
    }


def _compare(results: dict[str, list[dict]]) -> dict:
    generic = _summarize(results["generic"])
    personalized = _summarize(results["personalized"])
    return {
        "personalized_minus_generic_knowledge_quality": round(
            personalized["avg_final_knowledge_quality"] - generic["avg_final_knowledge_quality"],
            4,
        ),
        "personalized_minus_generic_goal_coverage": round(
            personalized["avg_final_goal_coverage"] - generic["avg_final_goal_coverage"],
            4,
        ),
        "personalized_minus_generic_avg_reward": round(personalized["avg_reward"] - generic["avg_reward"], 4),
    }


def _stable_offset(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(value))


if __name__ == "__main__":
    main()
