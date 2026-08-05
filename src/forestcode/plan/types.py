"""Task-plan layer types: the todo item and plan-store protocols."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol


TodoStatus = Literal["pending", "in_progress", "completed"]
MAX_TODOS = 20


@dataclass(frozen=True, slots=True)
class TodoItem:
    content: str
    active_form: str
    status: TodoStatus = "pending"


class PlanReader(Protocol):
    def get(self) -> list[TodoItem]: ...


class PlanStoreProtocol(PlanReader, Protocol):
    def set(self, items: list[TodoItem]) -> None: ...

    def seed(self, items: list[TodoItem]) -> None: ...
