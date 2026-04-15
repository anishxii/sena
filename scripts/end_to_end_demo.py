from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotiv_learn import ACTION_BANK, DecisionEngine
from emotiv_learn.schemas import (
    InteractionEffect,
    Outcome,
    RewardEvent,
    SemanticSignals,
    State,
    StateMetadata,
    TaskResult,
    TurnLog,
)


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

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
TURN_LOG_PATH = ARTIFACTS_DIR / "turn_logs.json"


class DemoEnvironment:
    """Stub System 3 environment for a smoke-style end-to-end run."""

    def __init__(self) -> None:
        self._turn_index = 0
        self._difficulty = "medium"
        self._last_action: str | None = None
        self._last_correct = 1
        self._last_latency_s = 2.5
        self._last_reread = 0
        self._clarification_count = 0

    def reset(self, user_id: str, task_type: str) -> dict:
        self._turn_index = 1
        return self._make_raw_observation(user_id=user_id, task_type=task_type)

    def step(self, user_id: str, interaction_effect: InteractionEffect) -> tuple[dict, Outcome]:
        self._last_action = interaction_effect.action_id

        if interaction_effect.action_id in {"simplify", "step_by_step", "worked_example"}:
            followup_type = "continue"
            correct = 1
            latency_s = 2.2
            reread = 0
            self._clarification_count = 0
        elif interaction_effect.action_id in {"deepen", "analogy"}:
            followup_type = "deeper_request"
            correct = 1
            latency_s = 3.0
            reread = 0
        elif interaction_effect.action_id == "highlight_key_points":
            followup_type = "continue"
            correct = 1
            latency_s = 2.6
            reread = 0
        else:
            followup_type = "clarify"
            correct = 0
            latency_s = 6.5
            reread = 1
            self._clarification_count += 1

        outcome = Outcome(
            timestamp=self._turn_index,
            user_id=user_id,
            action_id=interaction_effect.action_id,
            task_result=TaskResult(
                correct=correct,
                latency_s=latency_s,
                reread=reread,
                completed=1,
                abandoned=0,
            ),
            semantic_signals=SemanticSignals(
                followup_text=followup_type,
                followup_type=followup_type,
                confusion_score=0.7 if followup_type == "clarify" else 0.1,
                comprehension_score=0.8 if correct else 0.2,
                engagement_score=0.7,
                pace_fast_score=0.2,
                pace_slow_score=0.2,
            ),
            raw={
                "topic_id": "algebra",
                "concept_id": f"concept_{self._turn_index}",
            },
        )

        self._last_correct = correct
        self._last_latency_s = latency_s
        self._last_reread = reread
        self._turn_index += 1
        next_raw = self._make_raw_observation(user_id=user_id, task_type="learn")
        return next_raw, outcome

    def _make_raw_observation(self, user_id: str, task_type: str) -> dict:
        load_proxy = min(0.45 + 0.10 * self._clarification_count, 0.95)
        return {
            "timestamp": self._turn_index,
            "user_id": user_id,
            "eeg": {
                "theta": 0.42 + 0.03 * self._clarification_count,
                "alpha": 0.55 - 0.02 * self._clarification_count,
                "beta": 0.61,
                "beta_alpha_ratio": 1.11 + 0.05 * self._clarification_count,
                "load_proxy": load_proxy,
            },
            "context": {
                "task_type": task_type,
                "topic_id": "algebra",
                "difficulty": self._difficulty,
                "turn_index": self._turn_index,
                "last_action": self._last_action,
            },
            "raw_behavior": {
                "last_correct": self._last_correct,
                "last_latency_s": self._last_latency_s,
                "last_reread": self._last_reread,
                "last_followup_text": None,
                "clarification_count": self._clarification_count,
            },
        }


class DemoStateBuilder:
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
            1.0 if context["turn_index"] % 4 == 0 else 0.0,
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


class DemoInteractionModel:
    ACTION_EFFECTS = {
        "no_change": {
            "difficulty_shift": 0.0,
            "verbosity_shift": 0.0,
            "structure_shift": 0.0,
            "interactivity_shift": 0.0,
            "cognitive_load_delta": 0.0,
        },
        "simplify": {
            "difficulty_shift": -0.5,
            "verbosity_shift": 0.1,
            "structure_shift": 0.2,
            "interactivity_shift": 0.0,
            "cognitive_load_delta": -0.4,
        },
        "deepen": {
            "difficulty_shift": 0.4,
            "verbosity_shift": 0.2,
            "structure_shift": 0.0,
            "interactivity_shift": 0.0,
            "cognitive_load_delta": 0.3,
        },
        "summarize": {
            "difficulty_shift": -0.1,
            "verbosity_shift": -0.5,
            "structure_shift": 0.1,
            "interactivity_shift": 0.0,
            "cognitive_load_delta": -0.2,
        },
        "highlight_key_points": {
            "difficulty_shift": 0.0,
            "verbosity_shift": -0.1,
            "structure_shift": 0.5,
            "interactivity_shift": 0.0,
            "cognitive_load_delta": -0.1,
        },
        "worked_example": {
            "difficulty_shift": -0.2,
            "verbosity_shift": 0.3,
            "structure_shift": 0.3,
            "interactivity_shift": 0.2,
            "cognitive_load_delta": -0.3,
        },
        "analogy": {
            "difficulty_shift": 0.0,
            "verbosity_shift": 0.2,
            "structure_shift": 0.1,
            "interactivity_shift": 0.1,
            "cognitive_load_delta": -0.1,
        },
        "step_by_step": {
            "difficulty_shift": -0.3,
            "verbosity_shift": 0.4,
            "structure_shift": 0.4,
            "interactivity_shift": 0.1,
            "cognitive_load_delta": -0.4,
        },
    }

    def apply_action(self, state: State, action) -> InteractionEffect:
        return InteractionEffect(
            timestamp=state.timestamp,
            user_id=state.user_id,
            action_id=action.action_id,
            semantic_effect=self.ACTION_EFFECTS[action.action_id],
            rendering_info={
                "style_label": action.action_id,
                "notes": f"Applied {action.action_id} to topic {state.metadata.topic_id}",
            },
        )


class DemoOutcomeInterpreter:
    def interpret_outcome(self, outcome: Outcome) -> dict:
        followup_type = outcome.semantic_signals.followup_type or "unknown"
        correct = outcome.task_result.correct or 0
        latency_s = outcome.task_result.latency_s or 0.0
        clarify_signal = 1 if followup_type == "clarify" else 0
        continue_signal = 1 if followup_type == "continue" else 0
        deeper_request = 1 if followup_type == "deeper_request" else 0
        repeated_clarify = 1 if clarify_signal and (outcome.task_result.reread or 0) > 0 else 0
        return {
            "timestamp": outcome.timestamp,
            "user_id": outcome.user_id,
            "signals": {
                "checkpoint_occurred": 1 if outcome.timestamp % 4 == 0 else 0,
                "checkpoint_correct": correct if outcome.timestamp % 4 == 0 else 0,
                "progress_signal": 1.0 if correct else 0.25,
                "deeper_request": deeper_request,
                "continue_signal": continue_signal,
                "clarify_signal": clarify_signal,
                "repeated_clarify_same_concept": repeated_clarify,
                "hesitation_signal": 1 if latency_s > 5.0 else 0,
                "abandonment": outcome.task_result.abandoned or 0,
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
        action,
        outcome: Outcome,
        interpreted: dict,
    ) -> RewardEvent:
        return RewardEvent(
            timestamp=outcome.timestamp,
            user_id=state.user_id,
            state_features=state.features,
            action_id=action.action_id,
            reward=self.compute_reward(interpreted),
            outcome=outcome,
        )


class DemoExperimentRunner:
    def run_episode(
        self,
        user_id: str,
        task_type: str,
        engine: DecisionEngine,
        state_builder: DemoStateBuilder,
        interaction_model: DemoInteractionModel,
        outcome_interpreter: DemoOutcomeInterpreter,
        reward_model: DemoRewardModel,
        env: DemoEnvironment,
        num_turns: int = 6,
    ) -> list[TurnLog]:
        raw_observation = env.reset(user_id=user_id, task_type=task_type)
        turn_logs: list[TurnLog] = []

        for _ in range(num_turns):
            state = state_builder.build_state(raw_observation)
            action_scores = engine.score_actions(state, ACTION_BANK)
            action = engine.select_action(action_scores)
            interaction_effect = interaction_model.apply_action(state, action)
            next_raw_observation, outcome = env.step(
                user_id=raw_observation["user_id"],
                interaction_effect=interaction_effect,
            )
            interpreted = outcome_interpreter.interpret_outcome(outcome)
            reward_event = reward_model.make_reward_event(
                state=state,
                action=action,
                outcome=outcome,
                interpreted=interpreted,
            )
            engine.update(reward_event)
            turn_logs.append(
                TurnLog(
                    raw_observation=raw_observation,
                    state=state,
                    action_scores=action_scores,
                    action=action,
                    interaction_effect=interaction_effect,
                    outcome=outcome,
                    reward_event=reward_event,
                )
            )
            raw_observation = next_raw_observation

        return turn_logs


def write_turn_logs(turn_logs: list[TurnLog]) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TURN_LOG_PATH.write_text(
        json.dumps([asdict(turn_log) for turn_log in turn_logs], indent=2),
        encoding="utf-8",
    )
    return TURN_LOG_PATH


def main() -> None:
    engine = DecisionEngine(feature_dim=len(FEATURE_NAMES), epsilon=0.10, seed=7)
    turn_logs = DemoExperimentRunner().run_episode(
        user_id="user_a",
        task_type="learn",
        engine=engine,
        state_builder=DemoStateBuilder(),
        interaction_model=DemoInteractionModel(),
        outcome_interpreter=DemoOutcomeInterpreter(),
        reward_model=DemoRewardModel(),
        env=DemoEnvironment(),
        num_turns=6,
    )
    output_path = write_turn_logs(turn_logs)
    print(f"Wrote {len(turn_logs)} turn logs to {output_path}")


if __name__ == "__main__":
    main()
