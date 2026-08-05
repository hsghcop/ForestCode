"""Task-plan layer (agent state). Leaf package: no internal dependencies."""

from __future__ import annotations

from .serialization import normalize_todos, todos_from_dicts, todos_to_dicts
from .store import PlanStore
from .types import MAX_TODOS, PlanReader, PlanStoreProtocol, TodoItem, TodoStatus

__all__ = [
    "MAX_TODOS",
    "PlanReader",
    "PlanStore",
    "PlanStoreProtocol",
    "TodoItem",
    "TodoStatus",
    "normalize_todos",
    "todos_from_dicts",
    "todos_to_dicts",
]
