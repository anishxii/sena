from __future__ import annotations

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


def mock_student_response(hidden_state: dict[str, float]) -> dict:
    if hidden_state["confusion"] > 0.6:
        response_type = "clarify"
        message = "I get the high-level idea, but I am lost on the actual update step."
        confidence = 0.35
    elif hidden_state["curiosity"] > 0.65:
        response_type = "branch"
        message = "Can we explore why this matters for training neural networks?"
        confidence = 0.72
    else:
        response_type = "continue"
        message = "That makes sense. I am ready to keep going."
        confidence = 0.80
    return normalize_student_output(
        {
            "response_type": response_type,
            "student_message": message,
            "checkpoint_answer": None,
            "self_reported_confidence": confidence,
            "rationale_for_simulation": "Mocked from hidden state thresholds.",
        }
    )


def mock_interpreter(student_response: dict) -> dict:
    response_type = student_response["response_type"]
    if response_type == "clarify":
        raw = {
            "followup_type": "clarify",
            "checkpoint_correct": None,
            "checkpoint_score": None,
            "confusion_score": 0.75,
            "comprehension_score": 0.35,
            "engagement_score": 0.65,
            "progress_signal": 0.25,
            "pace_fast_score": 0.0,
            "pace_slow_score": 0.3,
            "evidence": {"confusion_phrases": ["lost on the actual update step"]},
        }
    elif response_type == "branch":
        raw = {
            "followup_type": "branch",
            "checkpoint_correct": None,
            "checkpoint_score": None,
            "confusion_score": 0.15,
            "comprehension_score": 0.75,
            "engagement_score": 0.85,
            "progress_signal": 0.65,
            "pace_fast_score": 0.0,
            "pace_slow_score": 0.0,
            "evidence": {"curiosity_phrases": ["Can we explore"]},
        }
    else:
        raw = {
            "followup_type": "continue",
            "checkpoint_correct": None,
            "checkpoint_score": None,
            "confusion_score": 0.10,
            "comprehension_score": 0.80,
            "engagement_score": 0.75,
            "progress_signal": 0.70,
            "pace_fast_score": 0.0,
            "pace_slow_score": 0.0,
            "evidence": {"understanding_phrases": ["makes sense"]},
        }
    return normalize_interpreter_output(raw)


def main() -> None:
    hidden_state = {
        "mastery": 0.45,
        "confusion": 0.62,
        "curiosity": 0.55,
        "fatigue": 0.20,
        "engagement": 0.70,
    }
    tutor_input = TutorPromptInput(
        topic="gradient descent",
        concept_id="learning_rate",
        conversation_summary="The learner has seen gradients and loss functions.",
        load_level="medium-high",
        behavior_summary="slow pace and one reread",
        last_followup_type="clarify",
        action_id="worked_example",
        length_target="short",
        difficulty_target="medium",
        include_checkpoint=False,
    )
    tutor_messages = build_tutor_messages(tutor_input)
    tutor_message = "Let's walk through one concrete gradient descent update with small numbers."

    student_messages = build_student_messages(
        StudentPromptInput(
            learner_profile={"example_preference": 0.8, "challenge_preference": 0.4},
            hidden_state=hidden_state,
            observable_signals={"cognitive_load": 0.72, "reread_count": 1},
            tutor_message=tutor_message,
        )
    )
    student_response = mock_student_response(hidden_state)
    interpreter_messages = build_interpreter_messages(
        InterpreterPromptInput(
            tutor_message=tutor_message,
            student_response=student_response,
            checkpoint_rubric=None,
            topic="gradient descent",
            concept_id="learning_rate",
            action_id=tutor_input.action_id,
            state_summary="moderate mastery with elevated confusion",
        )
    )
    interpreted = mock_interpreter(student_response)
    reward = compute_reward_from_interpreted(interpreted)

    print("=== Tutor Prompt Preview ===")
    print(tutor_messages[1]["content"][:600])
    print("\n=== Student Prompt Preview ===")
    print(student_messages[1]["content"][:600])
    print("\n=== Mock Student Response ===")
    print(student_response)
    print("\n=== Interpreter Prompt Preview ===")
    print(interpreter_messages[1]["content"][:600])
    print("\n=== Interpreted Signals ===")
    print(interpreted)
    print("\n=== Deterministic Reward ===")
    print(reward)


if __name__ == "__main__":
    main()
