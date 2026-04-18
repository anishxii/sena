from __future__ import annotations

from abc import ABC, abstractmethod

from .types import TurnLog


class TurnLogger(ABC):
    @abstractmethod
    def log_turn(self, turn_log: TurnLog) -> None:
        """Persist one canonical turn log."""
