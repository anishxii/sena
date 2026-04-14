from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import ACTION_BANK, DecisionEngine
from emotiv_learn.schemas import Outcome, RewardEvent, SemanticSignals, State, StateMetadata, TaskResult


FEATURE_NAMES = [
    "eeg_theta",
    "eeg_alpha",
    "eeg_beta",
    "eeg_beta_alpha_ratio",
    "eeg_load_proxy",
    "turn_index_norm",
    "difficulty_easy",
    "difficulty_medium",
    "difficulty_hard",
    "prev_correct",
    "prev_latency_norm",
    "prev_reread",
    "recent_clarify_count_norm",
    "recent_progress_norm",
    "checkpoint_mode_flag",
]


class DemoStateBuilder:
    """Stub System 2 component that maps a raw observation into a fixed state."""

    def build_state(self, raw_observation: dict) -> State:
        eeg = raw_observation["eeg"]
        context = raw_observation["context"]
        behavior = raw_observation["raw_behavior"]
        difficulty = context["difficulty"]

        features = [
            float(eeg["theta"]),
            float(eeg["alpha"]),
            float(eeg["beta"]),
            float(eeg["beta_alpha_ratio"]),
            float(eeg["load_proxy"]),
            min(context["turn_index"] / 10.0, 1.0),
            1.0 if difficulty == "easy" else 0.0,
            1.0 if difficulty == "medium" else 0.0,
            1.0 if difficulty == "hard" else 0.0,
            float(behavior["last_correct"] or 0),
            min(float(behavior["last_latency_s"] or 0.0) / 10.0, 1.0),
            float(behavior["last_reread"] or 0),
            min(float(behavior["clarification_count"] or 0) / 3.0, 1.0),
            0.75 if (behavior["last_correct"] or 0) == 1 else 0.25,
            1.0 if context["task_type"] == "quiz" else 0.0,
        ]

        return State(
            timestamp=raw_observation["timestamp"],
            user_id=raw_observation["user_id"],
            features=features,
            feature_names=FEATURE_NAMES,
            metadata=StateMetadata(
                task_type=context["task_type"],
                difficulty=difficulty,
                topic_id=context["topic_id"],
            ),
        )


class DemoOutcomeInterpreter:
    """Stub System 2 component that derives simple reward signals from outcome text."""

    def interpret_outcome(self, outcome: Outcome) -> dict:
        followup_type = outcome.semantic_signals.followup_type or "unknown"
        correct = outcome.task_result.correct or 0
        latency_s = outcome.task_result.latency_s or 0.0
        abandoned = outcome.task_result.abandoned or 0

        clarify_signal = 1 if followup_type == "clarify" else 0
        continue_signal = 1 if followup_type == "continue" else 0
        deeper_request = 1 if followup_type == "deeper_request" else 0
        hesitation_signal = 1 if latency_s > 8.0 else 0
        repeated_clarify = 1 if clarify_signal and (outcome.task_result.reread or 0) > 0 else 0

        return {
            "timestamp": outcome.timestamp,
            "user_id": outcome.user_id,
            "signals": {
                "checkpoint_occurred": 1 if correct in (0, 1) else 0,
                "checkpoint_correct": correct,
                "progress_signal": 1.0 if correct else 0.25,
                "deeper_request": deeper_request,
                "continue_signal": continue_signal,
                "clarify_signal": clarify_signal,
                "repeated_clarify_same_concept": repeated_clarify,
                "hesitation_signal": hesitation_signal,
                "abandonment": abandoned,
            },
            "metadata": {
                "topic_id": outcome.raw.get("topic_id"),
                "concept_id": outcome.raw.get("concept_id"),
                "latency_s": latency_s,
                "followup_type": followup_type,
                "confidence": 0.8,
            },
        }


class DemoRewardModel:
    """Stub System 2 component that turns interpreted signals into a scalar reward."""

    def compute_reward(self, interpreted: dict) -> float:
        signals = interpreted["signals"]
        reward = 0.0
        reward += 0.8 * signals["checkpoint_correct"]
        reward += 0.5 * signals["progress_signal"]
        reward += 0.4 * signals["deeper_request"]
        reward += 0.2 * signals["continue_signal"]
        reward -= 0.6 * signals["clarify_signal"]
        reward -= 0.5 * signals["repeated_clarify_same_concept"]
        reward -= 0.3 * signals["hesitation_signal"]
        reward -= 1.0 * signals["abandonment"]
        return max(-1.5, min(1.5, reward))

    def make_reward_event(
        self,
        state: State,
        action_id: str,
        outcome: Outcome,
        interpreted: dict,
    ) -> RewardEvent:
        return RewardEvent(
            timestamp=outcome.timestamp,
            user_id=state.user_id,
            state_features=state.features,
            action_id=action_id,
            reward=self.compute_reward(interpreted),
            outcome=outcome,
        )


def build_demo_raw_observation() -> dict:
    return {
        "timestamp": 1,
        "user_id": "user_a",
        "eeg": {
            "theta": 0.42,
            "alpha": 0.55,
            "beta": 0.61,
            "beta_alpha_ratio": 1.11,
            "load_proxy": 0.67,
        },
        "context": {
            "task_type": "learn",
            "topic_id": "algebra",
            "difficulty": "medium",
            "turn_index": 3,
            "last_action": "simplify",
        },
        "raw_behavior": {
            "last_correct": 1,
            "last_latency_s": 2.5,
            "last_reread": 0,
            "last_followup_text": "continue",
            "clarification_count": 0,
        },
    }


def build_demo_outcome(action_id: str, user_id: str) -> Outcome:
    return Outcome(
        timestamp=2,
        user_id=user_id,
        action_id=action_id,
        task_result=TaskResult(
            correct=1,
            latency_s=3.2,
            reread=0,
            completed=1,
            abandoned=0,
        ),
        semantic_signals=SemanticSignals(
            followup_text="continue",
            followup_type="continue",
            confusion_score=0.1,
            comprehension_score=0.8,
            engagement_score=0.7,
            pace_fast_score=0.2,
            pace_slow_score=0.1,
        ),
        raw={"topic_id": "algebra", "concept_id": "linear_equations"},
    )


def main() -> None:
    raw_observation = build_demo_raw_observation()
    state_builder = DemoStateBuilder()
    outcome_interpreter = DemoOutcomeInterpreter()
    reward_model = DemoRewardModel()
    engine = DecisionEngine(feature_dim=len(FEATURE_NAMES), epsilon=0.0, seed=7)

    state = state_builder.build_state(raw_observation)
    action_scores = engine.score_actions(state, ACTION_BANK)
    action = engine.select_action(action_scores)

    outcome = build_demo_outcome(action.action_id, state.user_id)
    interpreted = outcome_interpreter.interpret_outcome(outcome)
    reward_event = reward_model.make_reward_event(state, action.action_id, outcome, interpreted)
    engine.update(reward_event)
    rescored = engine.score_actions(state, ACTION_BANK)

    print("=== State ===")
    print(asdict(state))
    print("\n=== First Action Scores ===")
    print(asdict(action_scores))
    print("\n=== Selected Action ===")
    print(asdict(action))
    print("\n=== Interpreted Outcome ===")
    print(interpreted)
    print("\n=== Reward Event ===")
    print(asdict(reward_event))
    print("\n=== Scores After Update ===")
    print(asdict(rescored))


if __name__ == "__main__":
    main()
