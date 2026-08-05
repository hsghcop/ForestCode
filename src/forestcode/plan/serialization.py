"""dict <-> TodoItem conversion and normalization.

Single source of constraint logic, shared by the write_todos validator
(strict=True, raises on violations) and CLI seed (strict=False, lenient).
"""

from __future__ import annotations

from typing import Any

from .types import MAX_TODOS, TodoItem

_VALID_STATUS = {"pending", "in_progress", "completed"}


def normalize_todos(raw: Any, *, strict: bool) -> list[TodoItem]:
    if not isinstance(raw, list):
        if strict:
            raise ValueError("todos must be a list.")
        return []

    items: list[TodoItem] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, dict):
            if strict:
                raise ValueError(f"todos[{index}] must be an object.")
            continue

        content = entry.get("content")
        if not isinstance(content, str) or not content.strip():
            if strict:
                raise ValueError(f"todos[{index}].content must be a non-empty string.")
            continue

        active_form = entry.get("active_form")
        if not isinstance(active_form, str) or not active_form.strip():
            active_form = content  # display-only fallback; never fail on it

        status = entry.get("status", "pending")
        if status not in _VALID_STATUS:
            if strict:
                raise ValueError(f"todos[{index}].status must be one of {sorted(_VALID_STATUS)}.")
            status = "pending"

        items.append(TodoItem(content=content, active_form=active_form, status=status))

    if len(items) > MAX_TODOS:
        if strict:
            raise ValueError(f"todos must not exceed {MAX_TODOS} items.")
        items = items[:MAX_TODOS]

    in_progress = [i for i, item in enumerate(items) if item.status == "in_progress"]
    if len(in_progress) > 1:
        if strict:
            raise ValueError("Only one todo may be in_progress at a time.")
        for i in in_progress[1:]:
            stale = items[i]
            items[i] = TodoItem(content=stale.content, active_form=stale.active_form, status="pending")

    return items


def todos_to_dicts(items: list[TodoItem]) -> list[dict[str, Any]]:
    return [
        {"content": item.content, "active_form": item.active_form, "status": item.status}
        for item in items
    ]


def todos_from_dicts(raw: Any) -> list[TodoItem]:
    return normalize_todos(raw, strict=False)
