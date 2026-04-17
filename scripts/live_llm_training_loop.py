from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import ACTION_BANK, DecisionEngine
from emotiv_learn.eeg import EEGObservationContext, build_eeg_provider
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
from emotiv_learn.student_model import HiddenKnowledgeStudent, default_hidden_knowledge_state


CONTENT_STEPS = [
    {
        "concept_id": "gradient",
        "checkpoint": False,
        "rubric": None,
    },
    {
        "concept_id": "learning_rate",
        "checkpoint": False,
        "rubric": None,
    },
    {
        "concept_id": "gradient_descent_update",
        "checkpoint": True,
        "rubric": "A good answer says the update subtracts learning_rate * gradient from the current parameter.",
    },
    {
        "concept_id": "overshooting",
        "checkpoint": False,
        "rubric": None,
    },
    {
        "concept_id": "convergence",
        "checkpoint": True,
        "rubric": "A good answer connects convergence to repeated updates that move parameters toward lower loss.",
    },
]

LEARNER_PROFILE = {
    "example_preference": 0.8,
    "challenge_preference": 0.4,
    "structure_preference": 0.75,
    "verbosity_tolerance": 0.45,
}


def run_live_training_loop(
    topic: str,
    user_id: str,
    max_turns: int,
    difficulty: str,
    model: str | None,
    db_path: str | None,
    output_path: Path,
    eeg_mode: str,
) -> list[dict]:
    client = OpenAIChatClient(model=model)
    engine = DecisionEngine(
        feature_dim=len(LIVE_FEATURE_NAMES),
        epsilon=0.10,
        seed=7,
        db_path=db_path,
        reward_clip_abs=1.5,
        update_clip_abs=0.2,
        l2_weight_decay=0.001,
    )
    state_builder = LiveLLMStateBuilder()
    eeg_provider = build_eeg_provider(
        eeg_mode=eeg_mode,
        seed=17,
    )

    previous_interpreted = None
    previous_student_response = None
    previous_reward = 0.0
    previous_eeg_features = None
    previous_eeg_proxy = None
    student = HiddenKnowledgeStudent(default_hidden_knowledge_state(), seed=13)
    conversation_summary = "The learner is beginning a short lesson."
    logs: list[dict] = []

    selected_steps = [CONTENT_STEPS[index % len(CONTENT_STEPS)] for index in range(max_turns)]
    for turn_index, content_step in enumerate(selected_steps, start=1):
        state = state_builder.build_state(
            LiveStateInput(
                timestamp=turn_index,
                user_id=user_id,
                topic_id=topic,
                task_type="learn",
                difficulty=difficulty,
                turn_index=turn_index,
                max_turns=len(selected_steps),
                interpreted=previous_interpreted,
                student_response=previous_student_response,
                previous_reward=previous_reward,
                eeg_features=previous_eeg_features,
                eeg_proxy_estimates=previous_eeg_proxy,
            )
        )
        action_scores = engine.score_actions(state, ACTION_BANK)
        action = engine.select_action(action_scores)

        tutor_messages = build_tutor_messages(
            TutorPromptInput(
                topic=topic,
                concept_id=content_step["concept_id"],
                conversation_summary=conversation_summary,
                load_level=_load_level(previous_interpreted),
                behavior_summary=_behavior_summary(previous_interpreted),
                last_followup_type=(previous_interpreted or {}).get("followup_type", "unknown"),
                action_id=action.action_id,
                length_target="short",
                difficulty_target=difficulty,
                include_checkpoint=bool(content_step["checkpoint"]),
                learner_format_hint=_format_hint_from_hidden_state(student.hidden_state),
            )
        )
        tutor_message = client.complete_text(tutor_messages, max_tokens=900, temperature=0.35)
        transition = student.step(
            concept_id=content_step["concept_id"],
            action_id=action.action_id,
            tutor_message=tutor_message,
            checkpoint_expected=bool(content_step["checkpoint"]),
        )
        interpreted = _transition_to_interpreted(transition)
        eeg_window = eeg_provider.observe(
            EEGObservationContext(
                timestamp=turn_index,
                user_id=user_id,
                concept_id=content_step["concept_id"],
                action_id=action.action_id,
                tutor_message=tutor_message,
                time_on_chunk=None,
                hidden_state=transition.to_dict()["hidden_state_after"],
                observable_signals=transition.observable_signals,
            )
        )
        eeg_proxy = dict(eeg_window.metadata.get("proxy_state", {}))

        student_messages = build_student_messages(
            StudentPromptInput(
                learner_profile=LEARNER_PROFILE,
                hidden_state=transition.to_dict()["hidden_state_after"],
                observable_signals=transition.observable_signals,
                tutor_message=tutor_message,
                checkpoint_expected=bool(content_step["checkpoint"]),
                sampled_response_type=transition.sampled_response_type,
                checkpoint_answer=transition.checkpoint_answer,
            )
        )
        student_response = normalize_student_output(
            client.complete_json(student_messages, max_tokens=700, temperature=0.45)
        )
        student_response["response_type"] = transition.sampled_response_type
        if transition.checkpoint_answer is not None:
            student_response["checkpoint_answer"] = transition.checkpoint_answer
        interpreted["followup_type"] = transition.sampled_response_type
        interpreted = _enforce_checkpoint_schedule(
            interpreted=interpreted,
            checkpoint_expected=bool(content_step["checkpoint"]),
        )
        reward = compute_reward_from_interpreted(interpreted)

        outcome = _make_outcome(
            turn_index=turn_index,
            user_id=user_id,
            action_id=action.action_id,
            student_response=student_response,
            interpreted=interpreted,
            tutor_message=tutor_message,
        )
        reward_event = RewardEvent(
            timestamp=turn_index,
            user_id=user_id,
            state_features=state.features,
            action_id=action.action_id,
            reward=reward,
            outcome=outcome,
        )
        engine.update(reward_event)

        logs.append(
            {
                "turn_index": turn_index,
                "state": asdict(state),
                "action_scores": asdict(action_scores),
                "action": asdict(action),
                "tutor_message": tutor_message,
                "student_response": student_response,
                "interpreted": interpreted,
                "reward": reward,
                "eeg_window": asdict(eeg_window),
                "eeg_proxy_estimate": eeg_proxy,
                "student_transition": transition.to_dict(),
                "update_trace": asdict(engine.update_history[-1]),
            }
        )

        content_mix = _content_mix_from_transition(transition)
        previous_interpreted = interpreted
        previous_student_response = student_response
        previous_reward = reward
        previous_eeg_features = eeg_window.features
        previous_eeg_proxy = eeg_proxy
        conversation_summary = (
            f"Previous concept {content_step['concept_id']} used action {action.action_id}. "
            f"Student responded as {interpreted['followup_type']} with reward {reward:.2f}."
        )

        print(
            f"turn={turn_index} action={action.action_id} "
            f"mix={_dominant_content_mix(content_mix)} "
            f"(d={content_mix['text_description']:.2f}, "
            f"e={content_mix['text_examples']:.2f}, "
            f"v={content_mix['visual']:.2f}) "
            f"followup={interpreted['followup_type']} reward={reward:.3f} "
            f"oracle_gain={transition.oracle_mastery_gain:.3f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(logs, indent=2), encoding="utf-8")
    print(f"Wrote {len(logs)} live turns to {output_path}")
    return logs


def _make_outcome(
    turn_index: int,
    user_id: str,
    action_id: str,
    student_response: dict,
    interpreted: dict,
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


def _state_summary(interpreted: dict | None) -> str:
    if not interpreted:
        return "first turn; no prior interpreted outcome"
    return json.dumps(interpreted, indent=2)


def _format_hint_from_hidden_state(hidden_state) -> str:
    learning_style = getattr(hidden_state.knowledge_state, "learning_style", {})
    text_description = float(learning_style.get("text_description", 0.5))
    text_examples = float(learning_style.get("text_examples", 0.3))
    visual = float(learning_style.get("visual", 0.2))

    if visual >= max(text_description, text_examples):
        return (
            "Use scan-friendly structure: short labeled sections, bullets, numbered steps, "
            "compact traces, and arrows. Avoid dense paragraphs."
        )
    if text_examples >= text_description:
        return (
            "Teach example-first: begin with a concrete worked example or numeric trace, "
            "then connect it back to the concept."
        )
    return (
        "Teach description-first: begin with a concise conceptual explanation, then add at most "
        "one short example if needed."
    )


def _transition_to_interpreted(transition) -> dict[str, Any]:
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


def _content_mix_from_transition(transition) -> dict[str, float]:
    evaluation = transition.evaluation
    return {
        "text_description": float(evaluation.get("content_text_description", 0.0)),
        "text_examples": float(evaluation.get("content_text_examples", 0.0)),
        "visual": float(evaluation.get("content_visual", 0.0)),
    }


def _dominant_content_mix(content_mix: dict[str, float]) -> str:
    return max(content_mix, key=content_mix.get)


def _enforce_checkpoint_schedule(interpreted: dict[str, Any], checkpoint_expected: bool) -> dict[str, Any]:
    if checkpoint_expected:
        return interpreted
    corrected = dict(interpreted)
    corrected["checkpoint_correct"] = None
    corrected["checkpoint_score"] = None
    if corrected["followup_type"] == "checkpoint_answer":
        corrected["followup_type"] = "mixed"
    return corrected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live LLM-backed personalization training loop.")
    parser.add_argument("--topic", default="gradient descent")
    parser.add_argument("--user-id", default="live_user_a")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--db-path", default="artifacts/live_llm_engine.sqlite")
    parser.add_argument("--output", default="artifacts/live_llm_turns.json")
    parser.add_argument("--eeg-mode", default="synthetic", choices=["synthetic"])
    args = parser.parse_args()

    run_live_training_loop(
        topic=args.topic,
        user_id=args.user_id,
        max_turns=args.turns,
        difficulty=args.difficulty,
        model=args.model,
        db_path=args.db_path,
        output_path=Path(args.output),
        eeg_mode=args.eeg_mode,
    )


if __name__ == "__main__":
    main()
