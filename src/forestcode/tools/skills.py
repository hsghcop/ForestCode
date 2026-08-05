"""load_skill: on-demand, read-only tool backed by a fixed SkillSnapshot (PRD R3)."""

from __future__ import annotations

from typing import Any

from forestcode.skills import SkillSnapshot, format_loaded_skill
from forestcode.tools.types import ToolContext, ToolDefinition

LOAD_SKILL_DESCRIPTION = (
    "Load the full instructions of one skill from the available-skills catalog. "
    "The catalog lists skills by name; call this only with a name from that list."
)


def _validate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    name = arguments.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    return {"name": name.strip()}


def _run_load_skill(_context: ToolContext, snapshot: SkillSnapshot, name: str) -> str:
    loaded = snapshot.load(name)
    if loaded is None:
        raise ValueError(f"Unknown skill: {name}")
    return format_loaded_skill(loaded)


def create_load_skill_tool(snapshot: SkillSnapshot) -> ToolDefinition:
    """Build the load_skill tool for one fixed snapshot.

    ``persist_result=False``: the ToolExecutor marks the result
    ``state_only``/``transient`` so the SessionRecorder never writes the body.
    The body still enters the current run's ``RunState.messages`` so later model
    iterations in the same run can see it.
    """
    return ToolDefinition(
        name="load_skill",
        description=LOAD_SKILL_DESCRIPTION,
        input_schema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
        validator=_validate_arguments,
        runner=lambda context, **kwargs: _run_load_skill(
            context, snapshot, kwargs["name"]
        ),
        risk_level="read_only",
        is_read_only=True,
        persist_result=False,
    )
