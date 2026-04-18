from __future__ import annotations

from dataclasses import dataclass

from .interfaces import (
    ActionRegistry,
    InteractionModel,
    OutcomeInterpreter,
    RewardModel,
    StateBuilder,
)
from .logging import TurnLogger
from .types import RawObservation, TurnContext, TurnLog
from .validation import validate_action_scores, validate_reward_event, validate_state


@dataclass
class TurnRuntime:
    state_builder: StateBuilder
    action_registry: ActionRegistry
    interaction_model: InteractionModel
    outcome_interpreter: OutcomeInterpreter
    reward_model: RewardModel
    logger: TurnLogger | None = None

    def run_turn(self, *, raw_observation: RawObservation, engine, environment) -> TurnContext:
        state = self.state_builder.build_state(raw_observation)
        validate_state(state)

        action_bank = self.action_registry.list_actions()
        action_scores = engine.score_actions(state, action_bank)
        validate_action_scores(action_scores, action_ids=[action.action_id for action in action_bank])
        action = engine.select_action(action_scores)

        interaction_effect = self.interaction_model.apply_action(state, action)
        next_raw_observation, outcome = environment.step(
            user_id=raw_observation.user_id,
            interaction_effect=interaction_effect,
        )
        interpreted = self.outcome_interpreter.interpret_outcome(outcome)
        reward_event = self.reward_model.make_reward_event(
            state=state,
            action=action,
            outcome=outcome,
            interpreted=interpreted,
        )
        validate_reward_event(reward_event)
        engine.update(reward_event)

        turn_context = TurnContext(
            state=state,
            action_scores=action_scores,
            action=action,
            interaction_effect=interaction_effect,
            outcome=outcome,
            interpreted_outcome=interpreted,
            reward_event=reward_event,
        )
        if self.logger is not None:
            self.logger.log_turn(
                TurnLog(
                    raw_observation=raw_observation,
                    state=state,
                    action_scores=action_scores,
                    action=action,
                    interaction_effect=interaction_effect,
                    outcome=outcome,
                    interpreted_outcome=interpreted,
                    reward_event=reward_event,
                )
            )
        return turn_context


def run_turn(*, runtime: TurnRuntime, raw_observation: RawObservation, engine, environment) -> TurnContext:
    return runtime.run_turn(raw_observation=raw_observation, engine=engine, environment=environment)
