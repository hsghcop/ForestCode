"""Stop policies for agent runs."""

from __future__ import annotations

from typing import Protocol

from .run_state import RunState


class StopPolicy(Protocol):
    def should_stop(self, state: RunState) -> bool:
        """Return True if the agent loop should stop."""


class MaxTurnsStopPolicy:
    def __init__(self, max_turns: int = 10) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive")
        self.max_turns = max_turns

    def should_stop(self, state: RunState) -> bool:
        return state.turns >= self.max_turns
