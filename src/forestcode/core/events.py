"""Event sinks for observing agent runs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from .types import RunEvent


class EventSink(Protocol):
    def emit(self, event: RunEvent) -> None:
        """Record or publish a run event."""


class InMemoryEventSink:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def emit(self, event: RunEvent) -> None:
        self.events.append(event)


class CallbackEventSink:
    def __init__(self, callback: Callable[[RunEvent], None]) -> None:
        self.callback = callback

    def emit(self, event: RunEvent) -> None:
        self.callback(event)
