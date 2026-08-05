"""Slash command registry and shared command context."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TextIO

from forestcode.config import AgentRuntimeConfig
from forestcode.core.types import ModelClient
from forestcode.memory import SessionStore
from forestcode.plan import PlanStore
from forestcode.skills import PendingSkillSelection, SkillRegistry, SkillSnapshot
from forestcode.subagents import (
    AgentConfigSet,
    AgentRegistry,
    PendingSubagentSelection,
)
from forestcode.tools import ToolRuntimeServices

_COMMAND_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,30}$")


def looks_like_command(name: str) -> bool:
    return bool(_COMMAND_NAME_PATTERN.match(name))


@dataclass(slots=True)
class SlashCommand:
    name: str
    description: str
    handler: Callable[[SlashContext, str], SlashResult]
    source: Literal["builtin", "extension"] = "builtin"
    aliases: list[str] = field(default_factory=list)
    is_hidden: bool = False
    argument_hint: str = ""


@dataclass(slots=True)
class SlashContext:
    workspace_root: Path
    session_id: str | None
    session_store: SessionStore
    plan_store: PlanStore
    runtime: ToolRuntimeServices
    agent: AgentRuntimeConfig
    model: ModelClient
    stdout: TextIO
    stderr: TextIO
    input_func: Callable[[str], str]
    registry: SlashCommandRegistry
    # Skills runtime collaborators (design §Pending selection). None when the
    # caller does not wire skills; the /skills handler degrades gracefully.
    skill_registry: SkillRegistry | None = None
    skill_pending: PendingSkillSelection | None = None
    skill_selector: Callable[[SkillSnapshot], str | None] | None = None
    # Manual subagent selection collaborators. The selected name is process-local
    # and consumed by the next ordinary task; it is never written to history.
    agent_registry: AgentRegistry | None = None
    subagent_pending: PendingSubagentSelection | None = None
    subagent_selector: Callable[[AgentConfigSet], str | None] | None = None


@dataclass(slots=True)
class SlashResult:
    action: Literal["continue", "exit", "switch_session", "noop"] = "continue"
    new_session_id: str | None = None
    exit_code: int = 0
    prompt_text: str | None = None


class SlashCommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        if command.name != command.name.lower():
            raise ValueError(f"command name must be lowercase: {command.name}")
        if any(alias != alias.lower() for alias in command.aliases):
            raise ValueError(f"command aliases must be lowercase: {command.aliases}")

        if command.name in self._commands:
            raise ValueError(f"slash command already registered: {command.name}")

        for existing in self._commands.values():
            for alias in command.aliases:
                if alias == existing.name or alias in existing.aliases:
                    raise ValueError(f"slash command alias conflicts: {alias}")
            if command.name in existing.aliases:
                raise ValueError(
                    f"slash command name {command.name} conflicts with alias of {existing.name}"
                )
        self._commands[command.name] = command

    def get(self, name: str) -> SlashCommand | None:
        name = name.lower()
        cmd = self._commands.get(name)
        if cmd is not None:
            return cmd
        for command in self._commands.values():
            if name in command.aliases:
                return command
        return None

    def list(self, *, include_hidden: bool = False) -> list[SlashCommand]:
        items = self._commands.values()
        if not include_hidden:
            items = [command for command in items if not command.is_hidden]
        return sorted(items, key=lambda command: command.name)
