"""Reusable single-select skill picker (PRD R5, design §Selection UI).

Full tier: a prompt_toolkit arrow-key menu (no Live pause needed — /skills runs
between turns). Plain/fallback: a numbered list read through the input function.
Both return the chosen skill name or None when cancelled; cancelling keeps the
pending selection unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TextIO

from forestcode.skills import SkillSnapshot

from .select_menu import menu_step, numbered_select, pt_select

_PLAIN_HEADER = "Skills> pick a skill for this turn (enter number, empty to cancel)"
_PT_HEADER = "Skills> pick a skill for this turn (↑/↓ move, Enter select, Esc/Ctrl+C cancel)"


def make_numbered_skill_selector(
    *,
    input_func: Callable[[str], str],
    stdout: TextIO,
) -> Callable[[SkillSnapshot], str | None]:
    """Plain-tier selector: prints a numbered list and reads one number."""

    def select(snapshot: SkillSnapshot) -> str | None:
        options = [(d.name, d.description) for d in snapshot.descriptors]
        return numbered_select(
            options,
            prefix="Skills",
            header=_PLAIN_HEADER,
            input_func=input_func,
            stdout=stdout,
        )

    return select


def make_pt_skill_selector() -> Callable[[SkillSnapshot], str | None]:
    """Full-tier selector: inline arrow-key menu (prompt_toolkit imported lazily)."""

    def select(snapshot: SkillSnapshot) -> str | None:
        return _run_pt_skill_menu(snapshot)

    return select


def _pt_menu_step(index: int, key: str, count: int) -> tuple[int, str]:
    """Pure key->state transition for the arrow-key menu (testable contract).

    Returns ``(new_index, action)`` where action is ``"select"``, ``"cancel"``
    or ``""`` (no exit). ``up``/``down`` wrap around; ``enter`` confirms the
    current selection; ``escape``/``c-c`` cancel.
    """
    return menu_step(index, key, count)


def _run_pt_skill_menu(snapshot: SkillSnapshot) -> str | None:
    options = [(d.name, d.description) for d in snapshot.descriptors]
    return pt_select(options, header=_PT_HEADER)
