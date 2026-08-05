"""Patch / command confirmation controller (plan §6).

Consumes the backend's structured ``ApprovalRequest`` (plan §11-B4) so the
frontend reads ``path`` / diff stats / command metadata directly instead of
parsing the preview string. Owns the *frontend* session-level allowlist
("always allow <resource>"): the key is derived here from neutral facts via
``allow_key`` — the backend has no allowlist concept (plan §6.2).

The decision UI is pluggable via ``chooser``: the default reads a y/N/a line
through the ``InputController`` (works in every tier); the prompt_toolkit
arrow-key menu is injected later (plan §6.1, §6.5).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from forestcode.tools import ApprovalRequest

from .input import InputController
from .renderer import TerminalRenderer

Choice = Literal["yes", "no", "always"]
Chooser = Callable[[ApprovalRequest], Choice]

CONFIRM_DIFF_MAX_LINES = 40


def allow_key(request: ApprovalRequest) -> tuple:
    """Resource-level allowlist key derived from neutral facts (plan §6.2).

    patch   -> ("patch", path)
    command -> ("command", cwd, normalized command)   # cwd matters: same
               command in a different directory is a different resource.
    """
    if request.kind == "patch":
        return ("patch", request.path)
    return ("command", request.cwd, _normalize(request.command))


def _normalize(command: str | None) -> str:
    return " ".join((command or "").split())


def _resource_label(request: ApprovalRequest) -> str:
    if request.kind == "patch":
        return f"编辑 {request.path}"
    return f"运行 {_normalize(request.command)}"


class ConfirmationController:
    """Renders a patch/command preview and resolves a yes/no/always decision."""

    def __init__(
        self,
        renderer: TerminalRenderer,
        input_controller: InputController,
        *,
        chooser: Chooser | None = None,
        allow_always: bool = True,
    ) -> None:
        self._renderer = renderer
        self._input = input_controller
        self._chooser = chooser or self._text_chooser
        self._allow_always = allow_always
        self._allowlist: set[tuple] = set()

    def confirm(self, request: ApprovalRequest) -> bool:
        key = allow_key(request)
        if key in self._allowlist:
            return True  # session allowlist hit: no prompt (plan §6.2)

        self._render_preview(request)
        try:
            choice = self._chooser(request)
        except EOFError:
            # EOF (Ctrl+D / closed stdin) = reject, safe default (plan §6.4).
            return False
        # KeyboardInterrupt is NOT caught here; it propagates to the turn loop /
        # TurnRunner which cancels the whole turn (plan §6.4 / §7).

        if choice == "always":
            self._allowlist.add(key)
            return True
        return choice == "yes"

    # -- preview rendering -------------------------------------------------
    def _render_preview(self, request: ApprovalRequest) -> None:
        if request.kind == "command":
            self._renderer.render_warning("Command> requires approval")
            self._renderer.render_command_preview(request.preview)
        else:
            stat = ""
            if request.added is not None and request.removed is not None:
                stat = f"  ({request.added}+/{request.removed}-)"
            self._renderer.render_warning(
                f"Patch> {request.tool_name} requires approval  {request.path or ''}{stat}"
            )
            self._renderer.render_diff(self._truncate_diff(request.preview))

    @staticmethod
    def _truncate_diff(diff: str) -> str:
        lines = diff.splitlines()
        if len(lines) <= CONFIRM_DIFF_MAX_LINES:
            return diff
        remaining = len(lines) - CONFIRM_DIFF_MAX_LINES
        kept = lines[:CONFIRM_DIFF_MAX_LINES]
        kept.append(f"… {remaining} more lines")
        return "\n".join(kept)

    # -- default text chooser (y/N/a), used in every tier -----------------
    def _text_chooser(self, request: ApprovalRequest) -> Choice:
        if self._allow_always and not request.is_dangerous:
            prompt = f"Apply? [y/N/a]  (a = 总是允许 {_resource_label(request)})  "
        else:
            prompt = "Apply? [y/N] "
        answer = self._input.read_confirmation(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return "yes"
        if self._allow_always and not request.is_dangerous and answer in {"a", "always"}:
            return "always"
        return "no"
