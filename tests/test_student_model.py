from dataclasses import replace

from emotiv_learn.student_model import HiddenKnowledgeStudent, default_hidden_knowledge_state


def test_student_updates_hidden_knowledge_without_reward() -> None:
    student = HiddenKnowledgeStudent(default_hidden_knowledge_state(), seed=1)

    transition = student.step(
        concept_id="learning_rate",
        action_id="worked_example",
        tutor_message=(
            "Let's use a worked example. Step 1: start with a parameter. "
            "Step 2: compute the gradient. Step 3: multiply by the learning rate "
            "and subtract that amount from the parameter."
        ),
        checkpoint_expected=False,
    )

    before = transition.hidden_state_before.concept_mastery["learning_rate"]
    after = transition.hidden_state_after.concept_mastery["learning_rate"]
    assert after > before
    assert transition.oracle_mastery_gain == after - before
    assert "reward" not in transition.evaluation


def test_response_probabilities_are_sampled_from_hidden_state() -> None:
    student = HiddenKnowledgeStudent(default_hidden_knowledge_state(), seed=2)
    transition = student.step(
        concept_id="gradient",
        action_id="simplify",
        tutor_message="A gradient tells us which direction increases the loss the fastest.",
        checkpoint_expected=False,
    )

    assert set(transition.response_type_probs) == {"continue", "clarify", "branch"}
    assert abs(sum(transition.response_type_probs.values()) - 1.0) < 1e-9
    assert transition.sampled_response_type in transition.response_type_probs
    assert transition.observable_signals["followup_type"] == "unknown"


def test_checkpoint_expected_emits_checkpoint_signal() -> None:
    student = HiddenKnowledgeStudent(default_hidden_knowledge_state(), seed=3)
    transition = student.step(
        concept_id="gradient_descent_update",
        action_id="step_by_step",
        tutor_message=(
            "Step 1: compute the gradient. Step 2: multiply the gradient by the "
            "learning rate. Step 3: subtract that product from the current parameter. "
            "What gets subtracted during the update?"
        ),
        checkpoint_expected=True,
    )

    assert transition.checkpoint_correct in {True, False}
    assert transition.checkpoint_answer is not None
    assert transition.observable_signals["checkpoint_score"] is not None


def test_advanced_mastery_prefers_continue_over_clarify() -> None:
    hidden_state = default_hidden_knowledge_state()
    concept_mastery = dict(hidden_state.concept_mastery)
    concept_mastery["gradient"] = 0.92
    hidden_state = replace(hidden_state, concept_mastery=concept_mastery, confidence=0.9, fatigue=0.1)
    student = HiddenKnowledgeStudent(hidden_state, seed=4)

    transition = student.step(
        concept_id="gradient",
        action_id="summarize",
        tutor_message="A gradient is the direction of steepest increase.",
        checkpoint_expected=False,
    )

    probs = transition.response_type_probs
    assert probs["continue"] > probs["clarify"]
