from systems.system3b_tutor.student_model import HiddenKnowledgeStudent, default_hidden_knowledge_state


def test_student_step_updates_hidden_state_and_observables() -> None:
    student = HiddenKnowledgeStudent(default_hidden_knowledge_state(), seed=11)
    before_mastery = student.hidden_state.knowledge_state.concept_mastery["learning_rate"]

    transition = student.step(
        concept_id="learning_rate",
        action_id="worked_example",
        tutor_message="Example: the learning rate scales how large each gradient descent step should be.",
        checkpoint_expected=False,
    )

    after_mastery = transition.hidden_state_after.knowledge_state.concept_mastery["learning_rate"]

    assert after_mastery >= before_mastery
    assert transition.sampled_response_type in {"continue", "clarify", "branch"}
    assert set(transition.response_type_probs) == {"continue", "clarify", "branch"}
    assert abs(sum(transition.response_type_probs.values()) - 1.0) < 1e-6
    assert "confusion_score" in transition.observable_signals
    assert "workload" in transition.observable_signals
