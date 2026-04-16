from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.knowledge_agent import KnowledgeAgent, KnowledgeAgentProfile  # noqa: E402
from emotiv_learn.knowledge_scenarios import BACKPROP_SCENARIO  # noqa: E402
from emotiv_learn.openai_client import OpenAIChatClient  # noqa: E402


SYSTEM_PROMPT = """You are a concise machine-learning tutor.

Teach the requested concept to the learner. Use the requested strategy faithfully.
Keep the response focused and realistic for one tutoring turn.
Return only the tutor message."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Call the tutor LLM and show hidden learner-state changes.")
    parser.add_argument("--concept-id", default="gradient_descent")
    parser.add_argument("--action-id", default="worked_example")
    parser.add_argument("--learner-message", default="I understand neural networks and loss, but gradient descent still feels fuzzy.")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    scenario = BACKPROP_SCENARIO
    concept = scenario.concept(args.concept_id)
    profile = KnowledgeAgentProfile(
        user_id="demo_learner",
        initial_knowledge_level=0.42,
        curiosity=0.66,
        confidence=0.44,
        fatigue=0.16,
        engagement=0.72,
        attention=0.70,
    )
    agent = KnowledgeAgent(scenario=scenario, profile=profile, seed=args.seed)
    _advance_to_concept(agent, args.concept_id)

    prompt = f"""Topic: {scenario.title}
Current concept: {concept.label}
Canonical learning goal: {concept.canonical_claim}
Prerequisites: {concept.prerequisites}
Tutor strategy/action: {args.action_id}
Learner message: {args.learner_message}
Checkpoint expected: {args.checkpoint}

Write the next tutor response."""

    client = OpenAIChatClient(model=args.model)
    tutor_message = client.complete_text(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=700,
        temperature=0.35,
    )
    before = agent.state
    turn = agent.consume_tutor_step(
        concept_id=args.concept_id,
        tutor_message=tutor_message,
        action_id=args.action_id,
        checkpoint_expected=args.checkpoint,
    )
    after = turn.state_after

    output = {
        "concept_id": args.concept_id,
        "action_id": args.action_id,
        "tutor_message": tutor_message,
        "instructional_signals": turn.instructional_signals,
        "cognitive_appraisal": turn.cognitive_appraisal,
        "state_before": _state_summary(before, args.concept_id),
        "state_after": _state_summary(after, args.concept_id),
        "state_delta": _state_delta(before, after, args.concept_id),
        "learner_response_type": turn.response_type,
        "learner_reprompt": turn.reprompt,
        "checkpoint_correct": turn.checkpoint_correct,
        "progress_signal": turn.progress_signal,
    }
    print(json.dumps(output, indent=2))


def _advance_to_concept(agent: KnowledgeAgent, concept_id: str) -> None:
    concept_ids = agent.scenario.concept_ids
    target_index = concept_ids.index(concept_id)
    current_mastery = dict(agent.state.concept_mastery)
    for prior_id in concept_ids[:target_index]:
        current_mastery[prior_id] = max(current_mastery.get(prior_id, 0.0), 0.72)
    agent.state = type(agent.state)(
        knowledge_base=agent.state.knowledge_base,
        concept_mastery=current_mastery,
        confusion=agent.state.confusion,
        confidence=agent.state.confidence,
        curiosity=agent.state.curiosity,
        fatigue=agent.state.fatigue,
        engagement=agent.state.engagement,
        attention=agent.state.attention,
        current_concept_index=target_index,
        current_concept_steps=agent.state.current_concept_steps,
        steps_taken=agent.state.steps_taken,
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
        "knowledge_base_size": len(state.knowledge_base),
    }


def _state_delta(before, after, concept_id: str) -> dict:
    before_summary = _state_summary(before, concept_id)
    after_summary = _state_summary(after, concept_id)
    return {
        key: round(float(after_summary[key]) - float(before_summary[key]), 4)
        for key in before_summary
        if key != "knowledge_base_size"
    } | {
        "knowledge_base_size": after_summary["knowledge_base_size"] - before_summary["knowledge_base_size"]
    }


if __name__ == "__main__":
    main()
