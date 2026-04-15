from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn.llm_contracts import (
    InterpreterPromptInput,
    StudentPromptInput,
    TutorPromptInput,
    build_interpreter_messages,
    build_student_messages,
    build_tutor_messages,
    compute_reward_from_interpreted,
    normalize_interpreter_output,
    normalize_student_output,
)
from emotiv_learn.openai_client import OpenAIChatClient


def run_live_contract(topic: str, action_id: str, model: str | None = None) -> None:
    client = OpenAIChatClient(model=model)
    hidden_state = {
        "mastery": 0.45,
        "confusion": 0.62,
        "curiosity": 0.55,
        "fatigue": 0.20,
        "engagement": 0.70,
    }

    tutor_messages = build_tutor_messages(
        TutorPromptInput(
            topic=topic,
            concept_id="learning_rate",
            conversation_summary="The learner has seen gradients and loss functions.",
            load_level="medium-high",
            behavior_summary="slow pace and one reread",
            last_followup_type="clarify",
            action_id=action_id,
            length_target="short",
            difficulty_target="medium",
            include_checkpoint=False,
        )
    )
    tutor_message = client.complete_text(tutor_messages, max_tokens=800, temperature=0.35)

    student_messages = build_student_messages(
        StudentPromptInput(
            learner_profile={"example_preference": 0.8, "challenge_preference": 0.4},
            hidden_state=hidden_state,
            observable_signals={"cognitive_load": 0.72, "reread_count": 1},
            tutor_message=tutor_message,
        )
    )
    student_response = normalize_student_output(
        client.complete_json(student_messages, max_tokens=600, temperature=0.45)
    )

    interpreter_messages = build_interpreter_messages(
        InterpreterPromptInput(
            tutor_message=tutor_message,
            student_response=student_response,
            checkpoint_rubric=None,
            topic=topic,
            concept_id="learning_rate",
            action_id=action_id,
            state_summary="moderate mastery with elevated confusion",
        )
    )
    interpreted = normalize_interpreter_output(
        client.complete_json(interpreter_messages, max_tokens=700, temperature=0.10)
    )
    reward = compute_reward_from_interpreted(interpreted)

    print("=== Tutor Message ===")
    print(tutor_message)
    print("\n=== Student Response ===")
    print(student_response)
    print("\n=== Interpreted Signals ===")
    print(interpreted)
    print("\n=== Deterministic Reward ===")
    print(reward)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the LLM contract path with real OpenAI calls.")
    parser.add_argument("--topic", default="gradient descent")
    parser.add_argument("--action-id", default="worked_example")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    run_live_contract(topic=args.topic, action_id=args.action_id, model=args.model)


if __name__ == "__main__":
    main()
