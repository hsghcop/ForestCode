"""Single source of truth for the current task plan."""

from __future__ import annotations

from collections.abc import Callable

from .types import TodoItem


class PlanStore:
    def __init__(self, on_change: Callable[[list[TodoItem]], None] | None = None) -> None:
        self._items: list[TodoItem] = []
        self._on_change = on_change

    def get(self) -> list[TodoItem]:
        return list(self._items)

    def set(self, items: list[TodoItem]) -> None:
        self._items = list(items)
        if self._on_change is not None:
            self._on_change(self._items)

    def seed(self, items: list[TodoItem]) -> None:
        self._items = list(items)
