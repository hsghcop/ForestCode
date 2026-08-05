"""write_todos: tool adapter over the plan layer (state-only scratchpad)."""

from __future__ import annotations

from typing import Any

from forestcode.plan.serialization import normalize_todos
from forestcode.plan.types import TodoItem

from .types import ToolContext, ToolDefinition


_WRITE_TODOS_DESCRIPTION = """\
Create and manage a structured task list for the current session. Use proactively for
multi-step work (3+ steps), and to show progress to the user.

Use when: task has 3+ steps, user gives multiple tasks, or after receiving new instructions.
Skip when: a single trivial task.

Rules:
- Pass the FULL todo list every time (this overwrites the previous list).
- Provide content (imperative, "Run tests"); optionally active_form (present continuous, "Running tests").
- Keep EXACTLY ONE item in_progress at a time.
- Mark a task completed IMMEDIATELY after finishing it; do not batch completions.
- Never mark completed if it failed, is partial, or had unresolved errors — keep it in_progress.
"""


def _validate_write_todos(arguments: dict[str, Any]) -> dict[str, Any]:
    raw = arguments.get("todos")
    if not isinstance(raw, list):
        raise ValueError("write_todos requires a 'todos' list.")
    return {"todos": normalize_todos(raw, strict=True)}


def _run_write_todos(context: ToolContext, todos: list[TodoItem]) -> str:
    if context.plan_store is None:
        raise ValueError("PlanStore is required for write_todos.")

    if todos and all(item.status == "completed" for item in todos):
        context.plan_store.set([])
        return f"All {len(todos)} tasks complete. Plan cleared."

    context.plan_store.set(todos)
    done = sum(1 for item in todos if item.status == "completed")
    in_progress = sum(1 for item in todos if item.status == "in_progress")
    pending = sum(1 for item in todos if item.status == "pending")
    return (
        f"Plan updated ({done} done / {in_progress} in progress / {pending} pending). "
        "Keep using the todo list to track progress; mark items complete as you finish them."
    )


def create_write_todos_tool() -> ToolDefinition:
    return ToolDefinition(
        name="write_todos",
        description=_WRITE_TODOS_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "Imperative form, e.g. 'Run tests'",
                            },
                            "active_form": {
                                "type": "string",
                                "description": "Present continuous, e.g. 'Running tests' (optional; falls back to content)",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                            },
                        },
                        "required": ["content"],
                    },
                },
            },
            "required": ["todos"],
        },
        runner=_run_write_todos,
        risk_level="state",
        is_read_only=False,
        validator=_validate_write_todos,
    )
