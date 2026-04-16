from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.knowledge_agent import KnowledgeAgent, KnowledgeAgentProfile  # noqa: E402
from emotiv_learn.knowledge_evaluator import evaluate_knowledge_state  # noqa: E402
from emotiv_learn.knowledge_scenarios import BACKPROP_SCENARIO  # noqa: E402
from emotiv_learn.llm_contracts import build_tutor_messages, TutorPromptInput  # noqa: E402
from emotiv_learn.openai_client import OpenAIChatClient  # noqa: E402


ACTION_SEQUENCE = [
    "worked_example",
    "step_by_step",
    "analogy",
    "highlight_key_points",
    "simplify",
    "deepen",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a multi-turn cognitive learner-state experiment with real tutor LLM calls.")
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/cognitive_state_experiment.json"))
    args = parser.parse_args()

    scenario = BACKPROP_SCENARIO
    agent = KnowledgeAgent(
        scenario=scenario,
        profile=KnowledgeAgentProfile(
            user_id="long_demo_learner",
            initial_knowledge_level=0.42,
            curiosity=0.68,
            confidence=0.44,
            fatigue=0.14,
            engagement=0.72,
            attention=0.70,
        ),
        seed=args.seed,
    )
    client = OpenAIChatClient(model=args.model)
    learner_message = "I know neural networks are trainable, but I need the training process built up carefully."
    turn_logs = []

    for turn_index in range(1, args.turns + 1):
        concept_id = agent.current_concept_id()
        concept = scenario.concept(concept_id)
        action_id = ACTION_SEQUENCE[(turn_index - 1) % len(ACTION_SEQUENCE)]
        checkpoint_expected = turn_index % 2 == 0
        before = agent.state

        messages = build_tutor_messages(
            TutorPromptInput(
                topic=scenario.title,
                concept_id=concept_id,
                conversation_summary=learner_message,
                load_level=_load_level(before),
                behavior_summary=_behavior_summary(before, concept_id),
                last_followup_type="unknown" if not turn_logs else turn_logs[-1]["learner_response_type"],
                action_id=action_id,
                length_target="short",
                difficulty_target="medium",
                include_checkpoint=checkpoint_expected,
            )
        )
        tutor_message = client.complete_text(messages, max_tokens=650, temperature=0.35)
        turn = agent.consume_tutor_step(
            concept_id=concept_id,
            tutor_message=tutor_message,
            action_id=action_id,
            checkpoint_expected=checkpoint_expected,
        )
        evaluation = evaluate_knowledge_state(scenario, agent.state)
        after = agent.state
        turn_log = {
            "turn_index": turn_index,
            "concept_id": concept_id,
            "concept_label": concept.label,
            "action_id": action_id,
            "checkpoint_expected": checkpoint_expected,
            "tutor_message": tutor_message,
            "instructional_signals": turn.instructional_signals,
            "cognitive_appraisal": turn.cognitive_appraisal,
            "state_before": _state_summary(before, concept_id),
            "state_after": _state_summary(after, concept_id),
            "state_delta": _state_delta(before, after, concept_id),
            "learner_response_type": turn.response_type,
            "learner_reprompt": turn.reprompt,
            "checkpoint_correct": turn.checkpoint_correct,
            "progress_signal": turn.progress_signal,
            "evaluation": asdict(evaluation),
        }
        turn_logs.append(turn_log)
        learner_message = turn.reprompt or "Please continue."
        agent.advance_if_ready(concept_id)

    output = {
        "turns": args.turns,
        "seed": args.seed,
        "scenario": scenario.topic_id,
        "final_state": asdict(agent.state),
        "final_evaluation": asdict(evaluate_knowledge_state(scenario, agent.state)),
        "turn_logs": turn_logs,
        "summary": _summary(turn_logs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Wrote {args.output}")


def _load_level(state) -> str:
    if state.confusion >= 0.62 or state.fatigue >= 0.65:
        return "high"
    if state.confusion <= 0.32 and state.attention >= 0.65:
        return "low"
    return "medium"


def _behavior_summary(state, concept_id: str) -> str:
    return (
        f"mastery={state.concept_mastery.get(concept_id, 0.0):.2f}, "
        f"confusion={state.confusion:.2f}, confidence={state.confidence:.2f}, "
        f"fatigue={state.fatigue:.2f}, attention={state.attention:.2f}"
    )


def _state_summary(state, concept_id: str) -> dict:
    return {
        "concept_mastery": round(float(state.concept_mastery.get(concept_id, 0.0)), 4),
        "confusion": round(float(state.confusion), 4),
        "confidence": round(float(state.confidence), 4),
        "curiosity": round(float(state.curiosity), 4),
        "fatigue": round(float(state.fatigue), 4),
        "engagement": round(float(state.engagement), 4),
        "attention": round(float(state.attention), 4),
        "current_concept_index": state.current_concept_index,
    }


def _state_delta(before, after, concept_id: str) -> dict:
    before_summary = _state_summary(before, concept_id)
    after_summary = _state_summary(after, concept_id)
    return {
        key: round(float(after_summary[key]) - float(before_summary[key]), 4)
        for key in before_summary
        if key != "current_concept_index"
    }


def _summary(turn_logs: list[dict]) -> dict:
    first = turn_logs[0]["state_before"]
    last = turn_logs[-1]["state_after"]
    response_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for turn in turn_logs:
        response_counts[turn["learner_response_type"]] = response_counts.get(turn["learner_response_type"], 0) + 1
        action_counts[turn["action_id"]] = action_counts.get(turn["action_id"], 0) + 1
    return {
        "turns": len(turn_logs),
        "response_counts": response_counts,
        "action_counts": action_counts,
        "checkpoint_correct": sum(1 for turn in turn_logs if turn["checkpoint_correct"] is True),
        "checkpoint_attempts": sum(1 for turn in turn_logs if turn["checkpoint_correct"] is not None),
        "final_goal_coverage": turn_logs[-1]["evaluation"]["goal_coverage_score"],
        "final_knowledge_quality": turn_logs[-1]["evaluation"]["knowledge_quality_score"],
        "start_confusion": first["confusion"],
        "end_confusion": last["confusion"],
        "start_confidence": first["confidence"],
        "end_confidence": last["confidence"],
        "start_fatigue": first["fatigue"],
        "end_fatigue": last["fatigue"],
        "start_attention": first["attention"],
        "end_attention": last["attention"],
        "avg_progress_signal": round(sum(turn["progress_signal"] for turn in turn_logs) / len(turn_logs), 4),
        "max_total_load": max(turn["cognitive_appraisal"]["total_load"] for turn in turn_logs),
        "avg_productive_challenge": round(
            sum(turn["cognitive_appraisal"]["productive_challenge"] for turn in turn_logs) / len(turn_logs),
            4,
        ),
    }


if __name__ == "__main__":
    main()
