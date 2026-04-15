from emotiv_learn.learner_simulator import (
    HiddenLearnerSimulator,
    HiddenLearnerState,
    LearnerProfile,
    estimate_content_complexity,
)


def test_content_complexity_is_bounded() -> None:
    simple = estimate_content_complexity("A derivative measures change.")
    dense = estimate_content_complexity(
        "Backpropagation recursively applies the multivariate chain rule through "
        "intermediate activations, parameterized transformations, and nonlinear loss surfaces."
    )

    assert 0.0 <= simple <= 1.0
    assert 0.0 <= dense <= 1.0
    assert dense > simple


def test_support_action_reduces_confusion_for_overloaded_learner() -> None:
    simulator = HiddenLearnerSimulator(
        profile=LearnerProfile(user_id="u1", structure_preference=0.9, example_preference=0.8),
        initial_state=HiddenLearnerState(
            mastery=0.20,
            confusion=0.80,
            fatigue=0.35,
            curiosity=0.30,
            engagement=0.65,
        ),
        seed=7,
    )

    step = simulator.step(
        content_text="Gradient descent updates parameters by moving opposite the gradient.",
        action_id="step_by_step",
    )

    assert step.next_state.confusion < step.previous_state.confusion
    assert step.next_state.mastery > step.previous_state.mastery
    assert len(step.eeg_features) == 62


def test_deepen_can_overload_low_mastery_learner() -> None:
    simulator = HiddenLearnerSimulator(
        profile=LearnerProfile(user_id="u2", challenge_preference=0.2),
        initial_state=HiddenLearnerState(
            mastery=0.10,
            confusion=0.65,
            fatigue=0.20,
            curiosity=0.35,
            engagement=0.60,
        ),
        seed=11,
    )

    step = simulator.step(
        content_text="The Hessian eigenspectrum characterizes local curvature in nonconvex optimization.",
        action_id="deepen",
    )

    assert step.next_state.confusion >= step.previous_state.confusion


def test_checkpoint_signal_is_emitted_when_requested() -> None:
    simulator = HiddenLearnerSimulator(
        profile=LearnerProfile(user_id="u3"),
        initial_state=HiddenLearnerState(
            mastery=0.90,
            confusion=0.05,
            fatigue=0.05,
            curiosity=0.60,
            engagement=0.90,
        ),
        seed=3,
    )

    step = simulator.step(
        content_text="A derivative is the instantaneous rate of change.",
        action_id="no_change",
        checkpoint=True,
    )

    assert step.checkpoint_correct in (0, 1)
    assert step.reward_signals["checkpoint_occurred"] == 1
    assert step.reward_signals["checkpoint_correct"] in (0, 1)
