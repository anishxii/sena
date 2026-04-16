import json

import pytest

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
    parse_json_object,
)


def test_prompt_builders_include_schema_and_action_instruction() -> None:
    tutor_messages = build_tutor_messages(
        TutorPromptInput(
            topic="gradient descent",
            concept_id="learning_rate",
            conversation_summary="The learner has seen gradients.",
            load_level="medium",
            behavior_summary="low reread count",
            last_followup_type="continue",
            action_id="worked_example",
            length_target="short",
            difficulty_target="medium",
            include_checkpoint=True,
        )
    )

    assert "concrete worked example" in tutor_messages[1]["content"]
    assert "include checkpoint: True" in tutor_messages[1]["content"]

    student_messages = build_student_messages(
        StudentPromptInput(
            learner_profile={"example_preference": 0.8},
            hidden_state={
                "mastery": 0.4,
                "confusion": 0.6,
                "curiosity": 0.5,
                "fatigue": 0.2,
                "engagement": 0.7,
            },
            observable_signals={"cognitive_load": 0.7},
            tutor_message="Here is a worked example.",
            checkpoint_expected=True,
        )
    )
    assert '"response_type": "continue | clarify | branch"' in student_messages[1]["content"]
    assert "Checkpoint expected:\nTrue" in student_messages[1]["content"]
    assert "If checkpoint_expected is true" in student_messages[1]["content"]
    assert "choose continue if mastery >= 0.70" in student_messages[1]["content"]

    interpreter_messages = build_interpreter_messages(
        InterpreterPromptInput(
            tutor_message="Explain gradients.",
            student_response={"response_type": "clarify"},
            checkpoint_rubric=None,
            topic="gradient descent",
            concept_id="gradient",
            action_id="simplify",
            state_summary="high confusion",
        )
    )
    assert '"confusion_score": 0.0' in interpreter_messages[1]["content"]
    assert "partial understanding rather than pure failure" in interpreter_messages[0]["content"]


def test_json_parse_and_normalization() -> None:
    parsed = parse_json_object(
        "```json\n"
        + json.dumps(
            {
                "response_type": "branch",
                "student_message": "Can we explore this?",
                "checkpoint_answer": None,
                "self_reported_confidence": 1.4,
            }
        )
        + "\n```"
    )

    normalized = normalize_student_output(parsed)
    assert normalized["response_type"] == "branch"
    assert normalized["self_reported_confidence"] == 1.0


def test_reward_is_deterministic_from_interpreted_signals() -> None:
    interpreted = normalize_interpreter_output(
        {
            "followup_type": "branch",
            "checkpoint_correct": True,
            "checkpoint_score": 1.0,
            "confusion_score": 0.1,
            "comprehension_score": 0.9,
            "engagement_score": 0.8,
            "progress_signal": 0.85,
            "pace_fast_score": 0.0,
            "pace_slow_score": 0.0,
            "evidence": {"curiosity_phrases": ["Can we explore"]},
        }
    )

    assert compute_reward_from_interpreted(interpreted) == 1.5


def test_clarify_followup_gets_explicit_penalty() -> None:
    interpreted = normalize_interpreter_output(
        {
            "followup_type": "clarify",
            "checkpoint_correct": None,
            "checkpoint_score": None,
            "confusion_score": 0.5,
            "comprehension_score": 0.3,
            "engagement_score": 0.4,
            "progress_signal": 0.2,
            "pace_fast_score": 0.0,
            "pace_slow_score": 0.0,
            "evidence": {"confusion_phrases": ["I am confused"]},
        }
    )

    assert compute_reward_from_interpreted(interpreted) == pytest.approx(-0.23)
