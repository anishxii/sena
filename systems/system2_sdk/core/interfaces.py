from __future__ import annotations

from abc import ABC, abstractmethod

from .types import (
    Action,
    InteractionEffect,
    InterpretedOutcome,
    Outcome,
    RawObservation,
    RewardEvent,
    State,
)


class StateBuilder(ABC):
    @abstractmethod
    def build_state(self, raw_observation: RawObservation) -> State:
        """Build a fixed-length State from an application-specific RawObservation."""


class ActionRegistry(ABC):
    @abstractmethod
    def list_actions(self) -> list[Action]:
        """Return the canonical action bank for this application."""


class InteractionModel(ABC):
    @abstractmethod
    def apply_action(self, state: State, action: Action) -> InteractionEffect:
        """Map an Action into application-level interaction semantics."""


class OutcomeInterpreter(ABC):
    @abstractmethod
    def interpret_outcome(self, outcome: Outcome) -> InterpretedOutcome:
        """Convert a raw Outcome into structured signals."""


class RewardModel(ABC):
    @abstractmethod
    def compute_reward(self, outcome: Outcome, interpreted: InterpretedOutcome) -> float:
        """Produce a scalar reward from the interpreted outcome."""

    @abstractmethod
    def make_reward_event(
        self,
        *,
        state: State,
        action: Action,
        outcome: Outcome,
        interpreted: InterpretedOutcome,
    ) -> RewardEvent:
        """Build the canonical RewardEvent consumed by System 1."""
