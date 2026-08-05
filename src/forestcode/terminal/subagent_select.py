"""Single-select UI for configured subagents."""

from __future__ import annotations

from collections.abc import Callable
from typing import TextIO

from forestcode.subagents import AgentConfigSet

from .select_menu import numbered_select, pt_select

_PLAIN_HEADER = (
    "Subagents> pick a subagent (enter number, empty to cancel)"
)
_PT_HEADER = (
    "Subagents> pick a subagent (↑/↓ move, Enter select, Esc/Ctrl+C cancel)"
)


def _options(snapshot: AgentConfigSet) -> list[tuple[str, str]]:
    return [
        (
            config.name,
            f"{config.description} · permission: {config.permission_profile}",
        )
        for config in sorted(snapshot.agents.values(), key=lambda item: item.name)
    ]


def make_numbered_subagent_selector(
    *, input_func: Callable[[str], str], stdout: TextIO
) -> Callable[[AgentConfigSet], str | None]:
    def select(snapshot: AgentConfigSet) -> str | None:
        return numbered_select(
            _options(snapshot),
            prefix="Subagents",
            header=_PLAIN_HEADER,
            input_func=input_func,
            stdout=stdout,
        )

    return select


def make_pt_subagent_selector() -> Callable[[AgentConfigSet], str | None]:
    def select(snapshot: AgentConfigSet) -> str | None:
        return pt_select(_options(snapshot), header=_PT_HEADER)

    return select
