from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import ACTION_BANK, DecisionEngine
from emotiv_learn.learner_simulator import (
    HiddenLearnerSimulator,
    HiddenLearnerState,
    LearnerProfile,
    LearnerStep,
)
from emotiv_learn.schemas import Outcome, RewardEvent, SemanticSignals, State, StateMetadata, TaskResult


CONTENT_STEPS = [
    "A derivative measures how quickly a function changes at a point.",
    "The power rule says that the derivative of x to the n is n times x to the n minus one.",
    "The chain rule is used when one function is nested inside another function.",
    "Gradient descent updates parameters by moving opposite the gradient of the loss.",
    "Backpropagation applies the chain rule backward through a computational graph.",
    "A learning rate controls how large each gradient descent step is.",
    "Overfitting happens when a model memorizes training examples instead of generalizing.",
    "Regularization discourages overly complex models by adding constraints or penalties.",
]


FEATURE_NAMES = [
    *[f"eeg_{index}" for index in range(62)],
    "time_on_chunk_norm",
    "scroll_rate",
    "reread_count_norm",
    "turn_index_norm",
    "difficulty_easy",
    "difficulty_medium",
    "difficulty_hard",
]


@dataclass(frozen=True)
class UserScenario:
    profile: LearnerProfile
    initial_state: HiddenLearnerState
    difficulty: str


USER_SCENARIOS = [
    UserScenario(
        profile=LearnerProfile(
            user_id="support_seeker",
            example_preference=0.85,
            structure_preference=0.90,
            challenge_preference=0.20,
            verbosity_tolerance=0.40,
        ),
        initial_state=HiddenLearnerState(
            mastery=0.18,
            confusion=0.78,
            fatigue=0.25,
            curiosity=0.35,
            engagement=0.65,
        ),
        difficulty="hard",
    ),
    UserScenario(
        profile=LearnerProfile(
            user_id="curious_builder",
            example_preference=0.45,
            abstraction_preference=0.80,
            structure_preference=0.45,
            challenge_preference=0.85,
            verbosity_tolerance=0.75,
        ),
        initial_state=HiddenLearnerState(
            mastery=0.55,
            confusion=0.25,
            fatigue=0.15,
            curiosity=0.75,
            engagement=0.75,
        ),
        difficulty="medium",
    ),
    UserScenario(
        profile=LearnerProfile(
            user_id="scanner",
            example_preference=0.35,
            abstraction_preference=0.50,
            structure_preference=0.95,
            challenge_preference=0.45,
            verbosity_tolerance=0.25,
        ),
        initial_state=HiddenLearnerState(
            mastery=0.38,
            confusion=0.45,
            fatigue=0.45,
            curiosity=0.50,
            engagement=0.60,
        ),
        difficulty="medium",
    ),
]


class HiddenLearnerStateBuilder:
    def build_state(
        self,
        step: LearnerStep,
        turn_index: int,
        difficulty: str,
    ) -> State:
        cues = step.behavioral_cues
        difficulty_flags = {
            "easy": 1.0 if difficulty == "easy" else 0.0,
            "medium": 1.0 if difficulty == "medium" else 0.0,
            "hard": 1.0 if difficulty == "hard" else 0.0,
        }
        features = [
            *step.eeg_features,
            min(float(cues["time_on_chunk"]) / 180.0, 1.0),
            float(cues["scroll_rate"]),
            min(float(cues["reread_count"]) / 6.0, 1.0),
            min(turn_index / len(CONTENT_STEPS), 1.0),
            difficulty_flags["easy"],
            difficulty_flags["medium"],
            difficulty_flags["hard"],
        ]
        return State(
            timestamp=turn_index,
            user_id=step.user_id,
            features=features,
            feature_names=FEATURE_NAMES,
            metadata=StateMetadata(
                task_type="learn",
                difficulty=difficulty,
                topic_id="ml_fundamentals",
            ),
        )


class HiddenLearnerOutcomeAdapter:
    def make_outcome(self, step: LearnerStep, turn_index: int) -> Outcome:
        signals = step.reward_signals
        return Outcome(
            timestamp=turn_index,
            user_id=step.user_id,
            action_id=step.action_id,
            task_result=TaskResult(
                correct=step.checkpoint_correct,
                latency_s=float(step.behavioral_cues["time_on_chunk"]),
                reread=int(step.behavioral_cues["reread_count"]),
                completed=1,
                abandoned=int(signals["abandonment"]),
            ),
            semantic_signals=SemanticSignals(
                followup_text=step.learner_response_type,
                followup_type=step.learner_response_type,
                confusion_score=step.next_state.confusion,
                comprehension_score=step.next_state.mastery,
                engagement_score=step.next_state.engagement,
                pace_fast_score=None,
                pace_slow_score=None,
            ),
            raw={
                "hidden_previous_state": step.previous_state.__dict__,
                "hidden_next_state": step.next_state.__dict__,
                "content_complexity": step.content_complexity,
                "reward_signals": step.reward_signals,
            },
        )


class HiddenLearnerRewardModel:
    def compute_reward(self, step: LearnerStep) -> float:
        signals = step.reward_signals
        mastery_gain = step.next_state.mastery - step.previous_state.mastery
        reward = 0.0
        reward += 1.00 * signals["checkpoint_correct"]
        reward += 1.50 * mastery_gain
        reward += 0.30 * signals["deeper_request"]
        reward += 0.20 * signals["continue_signal"]
        reward -= 0.60 * signals["clarify_signal"]
        reward -= 0.40 * signals["repeated_clarify_same_concept"]
        reward -= 0.20 * signals["hesitation_signal"]
        reward -= 1.00 * signals["abandonment"]
        return max(-1.5, min(1.5, reward))

    def make_reward_event(self, state: State, step: LearnerStep, outcome: Outcome) -> RewardEvent:
        return RewardEvent(
            timestamp=outcome.timestamp,
            user_id=state.user_id,
            state_features=state.features,
            action_id=step.action_id,
            reward=self.compute_reward(step),
            outcome=outcome,
        )


def bootstrap_observation(
    simulator: HiddenLearnerSimulator,
    content_text: str,
    checkpoint: bool,
) -> LearnerStep:
    return simulator.step(content_text=content_text, action_id="no_change", checkpoint=checkpoint)


def run_scenario(
    scenario: UserScenario,
    use_personalization: bool,
    seed: int,
) -> dict:
    simulator = HiddenLearnerSimulator(
        profile=scenario.profile,
        initial_state=scenario.initial_state,
        seed=seed,
    )
    engine = DecisionEngine(
        feature_dim=len(FEATURE_NAMES),
        epsilon=0.10,
        use_personalization=use_personalization,
        seed=seed,
        reward_clip_abs=1.5,
        update_clip_abs=0.2,
        l2_weight_decay=0.001,
    )
    state_builder = HiddenLearnerStateBuilder()
    outcome_adapter = HiddenLearnerOutcomeAdapter()
    reward_model = HiddenLearnerRewardModel()

    total_reward = 0.0
    checkpoint_correct = 0
    checkpoint_count = 0
    clarify_count = 0
    branch_count = 0
    action_counts = {action.action_id: 0 for action in ACTION_BANK}
    initial_mastery = scenario.initial_state.mastery

    observation_step = bootstrap_observation(
        simulator=simulator,
        content_text=CONTENT_STEPS[0],
        checkpoint=False,
    )

    for turn_index, content_text in enumerate(CONTENT_STEPS, start=1):
        state = state_builder.build_state(
            step=observation_step,
            turn_index=turn_index,
            difficulty=scenario.difficulty,
        )
        action_scores = engine.score_actions(state, ACTION_BANK)
        action = engine.select_action(action_scores)
        checkpoint = turn_index % 3 == 0
        learner_step = simulator.step(
            content_text=content_text,
            action_id=action.action_id,
            checkpoint=checkpoint,
        )
        outcome = outcome_adapter.make_outcome(learner_step, turn_index)
        reward_event = reward_model.make_reward_event(state, learner_step, outcome)
        engine.update(reward_event)

        total_reward += reward_event.reward
        action_counts[action.action_id] += 1
        clarify_count += int(learner_step.learner_response_type == "clarify")
        branch_count += int(learner_step.learner_response_type == "branch")
        if learner_step.checkpoint_correct is not None:
            checkpoint_count += 1
            checkpoint_correct += learner_step.checkpoint_correct

        observation_step = learner_step

    final_mastery = simulator.state.mastery
    return {
        "user_id": scenario.profile.user_id,
        "average_reward": total_reward / len(CONTENT_STEPS),
        "mastery_gain": final_mastery - initial_mastery,
        "final_mastery": final_mastery,
        "checkpoint_accuracy": checkpoint_correct / checkpoint_count if checkpoint_count else 0.0,
        "clarify_rate": clarify_count / len(CONTENT_STEPS),
        "branch_rate": branch_count / len(CONTENT_STEPS),
        "action_counts": action_counts,
    }


def run_experiment(seed: int = 7) -> dict:
    generic_results = [
        run_scenario(scenario=scenario, use_personalization=False, seed=seed)
        for scenario in USER_SCENARIOS
    ]
    personalized_results = [
        run_scenario(scenario=scenario, use_personalization=True, seed=seed)
        for scenario in USER_SCENARIOS
    ]
    return {
        "generic": generic_results,
        "personalized": personalized_results,
    }


def summarize(results: list[dict]) -> dict:
    return {
        "average_reward": sum(row["average_reward"] for row in results) / len(results),
        "mastery_gain": sum(row["mastery_gain"] for row in results) / len(results),
        "checkpoint_accuracy": sum(row["checkpoint_accuracy"] for row in results) / len(results),
        "clarify_rate": sum(row["clarify_rate"] for row in results) / len(results),
        "branch_rate": sum(row["branch_rate"] for row in results) / len(results),
    }


def print_results(results: dict) -> None:
    generic_summary = summarize(results["generic"])
    personalized_summary = summarize(results["personalized"])

    for label, rows, summary in [
        ("Generic", results["generic"], generic_summary),
        ("Personalized", results["personalized"], personalized_summary),
    ]:
        print(f"=== {label} ===")
        print(
            "avg_reward={average_reward:.3f} mastery_gain={mastery_gain:.3f} "
            "checkpoint_acc={checkpoint_accuracy:.3f} clarify_rate={clarify_rate:.3f} "
            "branch_rate={branch_rate:.3f}".format(**summary)
        )
        for row in rows:
            top_action = max(row["action_counts"], key=row["action_counts"].get)
            print(
                f"  {row['user_id']}: reward={row['average_reward']:.3f} "
                f"mastery_gain={row['mastery_gain']:.3f} "
                f"checkpoint_acc={row['checkpoint_accuracy']:.3f} "
                f"top_action={top_action}"
            )
        print()

    print("=== Lift: Personalized - Generic ===")
    for key in generic_summary:
        print(f"{key}: {personalized_summary[key] - generic_summary[key]:+.3f}")


def main() -> None:
    print_results(run_experiment())


if __name__ == "__main__":
    main()
