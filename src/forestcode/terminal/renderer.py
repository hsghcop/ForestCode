"""Terminal renderers for the ForestCode interactive CLI.

`PlainRenderer` is the always-available text fallback and also owns the
`on_event` dispatch, so `RichRenderer` (see ``rich_renderer.py``) can subclass
it and override only the styled primitives. Plain output keeps grep-able
prefixes (`Assistant>`, `Tool>`, `Memory>`, …).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO, runtime_checkable

from forestcode.context import ModelInput
from forestcode.core.types import RunEvent


@dataclass(frozen=True, slots=True)
class Theme:
    """Semantic color tokens (rich style strings).

    Truecolor values; rich downgrades them to 256/16 colors based on terminal
    capability, so no explicit 8-color table is stored here. Renderers reference
    tokens by name and never hardcode literal colors.
    """

    primary: str
    accent: str
    success: str
    error: str
    warning: str
    info: str
    muted: str
    diff_add: str
    diff_del: str
    diff_hunk: str
    diff_meta: str


# Forest theme (§7.4): deep forest green primary, moss/sprout accents, bark amber
# warning, mushroom red error — low saturation, nature-leaning.
FOREST_THEME = Theme(
    primary="bold #2E8B57",
    accent="#7FB069",
    success="#5FA85F",
    error="#B5503C",
    warning="#C89B3C",
    info="#6FA8A0",
    muted="#7A8471",
    diff_add="#5FA85F",
    diff_del="#B5503C",
    diff_hunk="#6FA8A0",
    diff_meta="bold #8FA876",
)


@dataclass(slots=True)
class FrontendState:
    """Display-only frontend state. Never feeds backend decisions (§16)."""

    workspace_root: Path
    session_id: str | None
    model_name: str
    command_tools_enabled: bool = False
    show_reasoning: bool = False
    debug_context: bool = False


@runtime_checkable
class TerminalRenderer(Protocol):
    def render_welcome(self, state: FrontendState) -> None: ...
    def render_prompt(self, state: FrontendState) -> str: ...
    def render_user_error(self, message: str) -> None: ...
    def render_system_error(self, message: str) -> None: ...
    def render_warning(self, message: str) -> None: ...
    def render_assistant_text(self, text: str) -> None: ...
    def render_skill_activated(self, name: str) -> None: ...
    def render_reasoning(self, text: str, provider: str, kind: str) -> None: ...
    def render_tool_started(self, tool_name: str, tool_call_id: str) -> None: ...
    def render_tool_finished(
        self, tool_name: str, ok: bool, summary: str | None
    ) -> None: ...
    def render_subagent_status(
        self, task_id: str, agent_name: str, status: str
    ) -> None: ...
    def render_subagent_tool(
        self,
        task_id: str,
        agent_name: str,
        tool_name: str,
        started: bool,
        ok: bool | None = None,
    ) -> None: ...
    def render_plan_summary(self, summary: str) -> None: ...
    def render_memory_status(self, message: str) -> None: ...
    def render_session_status(self, message: str) -> None: ...
    def render_diff(self, diff: str) -> None: ...
    def render_command_preview(self, preview: str) -> None: ...
    def render_context_debug(self, model_inputs: list[ModelInput]) -> None: ...
    def on_event(self, event: RunEvent) -> None: ...
    def begin_turn(self) -> None: ...
    def end_turn(self) -> None: ...
    def newline(self) -> None: ...


class PlainRenderer:
    """Plain-text renderer and shared `on_event` dispatch."""

    def __init__(
        self,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
        *,
        verbose_events: bool = False,
        show_reasoning: bool = False,
        reasoning_display_max_chars: int = 2000,
    ) -> None:
        self._out = stdout
        self._err = stderr
        self._verbose = verbose_events
        self._show_reasoning = show_reasoning
        self._reasoning_max = reasoning_display_max_chars

    # -- low-level emit ----------------------------------------------------
    def _line(self, stream: TextIO, text: str) -> None:
        print(text, file=stream, flush=True)

    # -- structured render methods ----------------------------------------
    def render_welcome(self, state: FrontendState) -> None:
        self._line(self._out, "ForestCode")
        self._line(self._out, f"Workspace> {state.workspace_root}")
        self._line(self._out, f"Session> {state.session_id or 'disabled'}")
        self._line(self._out, f"Model> {state.model_name}")

    def render_prompt(self, state: FrontendState) -> str:
        return "ForestCode> "

    def render_user_error(self, message: str) -> None:
        self._line(self._err, message)

    def render_system_error(self, message: str) -> None:
        self._line(self._err, message)

    def render_warning(self, message: str) -> None:
        self._line(self._out, message)

    def render_assistant_text(self, text: str) -> None:
        # Empty stop emits assistant_text_received{text: ""}; skip it (§5.4).
        if text.strip():
            self._line(self._out, f"Assistant> {text}")

    def render_skill_activated(self, name: str) -> None:
        self._line(self._out, f"Skill> 已加载 {name}")

    def render_reasoning(self, text: str, provider: str, kind: str) -> None:
        self._line(self._out, f"Reasoning> {self._truncate(text)}")

    def render_tool_started(self, tool_name: str, tool_call_id: str) -> None:
        self._line(self._out, f"Tool> {tool_name}")

    def render_tool_finished(
        self, tool_name: str, ok: bool, summary: str | None
    ) -> None:
        self._line(self._out, f"Tool> {tool_name} {'ok' if ok else 'error'}")

    def render_subagent_status(
        self, task_id: str, agent_name: str, status: str
    ) -> None:
        short_id = task_id if len(task_id) <= 12 else task_id[:12]
        marker = {
            "queued": "queued",
            "running": "started",
            "waiting_approval": "approval",
            "completed": "completed",
            "failed": "failed",
            "cancelling": "cancelling",
            "cancelled": "cancelled",
        }.get(status, status)
        self._line(self._out, f"Subagent> [{short_id}] {agent_name} {marker}")

    def render_subagent_tool(
        self,
        task_id: str,
        agent_name: str,
        tool_name: str,
        started: bool,
        ok: bool | None = None,
    ) -> None:
        short_id = task_id if len(task_id) <= 12 else task_id[:12]
        suffix = "" if started else (" ok" if ok else " error")
        self._line(
            self._out, f"Subagent> [{short_id}] {agent_name} tool {tool_name}{suffix}"
        )

    def render_plan_summary(self, summary: str) -> None:
        self._line(self._out, f"Plan> {summary}")

    def render_memory_status(self, message: str) -> None:
        self._line(self._out, f"Memory> {message}")

    def render_session_status(self, message: str) -> None:
        self._line(self._out, f"Session> {message}")

    def render_diff(self, diff: str) -> None:
        self._line(self._out, diff)

    def render_command_preview(self, preview: str) -> None:
        self._line(self._out, preview)

    def render_context_debug(self, model_inputs: list[ModelInput]) -> None:
        if not model_inputs:
            self._line(self._out, "Context> no model request captured")
            return
        for index, model_input in enumerate(model_inputs, start=1):
            metadata = model_input.metadata
            sources = metadata.get("context_sources") or []
            tools = metadata.get("selected_tools") or []
            self._line(self._out, f"Context> turn {index}")
            self._line(
                self._out,
                f"Context> sources: {', '.join(sources) if sources else '(none)'}",
            )
            self._line(
                self._out,
                f"Context> messages: {metadata.get('message_count', len(model_input.messages))}",
            )
            self._line(
                self._out, f"Context> tools: {', '.join(tools) if tools else '(none)'}"
            )
            self._line(
                self._out, f"Context> chars: {metadata.get('char_count', 'unknown')}"
            )
            self._line(
                self._out, f"Context> truncated: {metadata.get('truncated', False)}"
            )

    def newline(self) -> None:
        self._line(self._out, "")

    # -- turn lifecycle ---------------------------------------------------
    def begin_turn(self) -> None:
        """No-op: the plain renderer has no animated live region (§3)."""

    def end_turn(self) -> None:
        """No-op companion to begin_turn."""

    # -- event dispatch (shared with RichRenderer) ------------------------
    def on_event(self, event: RunEvent) -> None:
        kind = event.type
        payload = event.payload
        if kind == "subagent_status_changed":
            self.render_subagent_status(
                str(payload.get("task_id", "")),
                str(payload.get("agent_name", "")),
                str(payload.get("status", "")),
            )
        elif kind == "subagent_tool_call_started":
            self.render_subagent_tool(
                str(payload.get("task_id", "")),
                str(payload.get("agent_name", "")),
                str(payload.get("tool_name", "")),
                started=True,
            )
        elif kind == "subagent_tool_call_finished":
            self.render_subagent_tool(
                str(payload.get("task_id", "")),
                str(payload.get("agent_name", "")),
                str(payload.get("tool_name", "")),
                started=False,
                ok=bool(payload.get("ok")),
            )
        elif kind == "tool_call_started":
            self.render_tool_started(
                payload.get("tool_name", ""), payload.get("tool_call_id", "")
            )
        elif kind == "tool_call_finished":
            self.render_tool_finished(
                payload.get("tool_name", ""),
                bool(payload.get("ok")),
                payload.get("summary"),
            )
            if payload.get("tool_name") == "write_todos" and payload.get("summary"):
                self.render_plan_summary(str(payload["summary"]))
        elif kind == "assistant_reasoning_received":
            if self._show_reasoning:
                self.render_reasoning(
                    str(payload.get("text", "")),
                    str(payload.get("provider", "")),
                    str(payload.get("kind", "")),
                )
        elif kind == "assistant_text_received":
            self.render_assistant_text(str(payload.get("text", "")))
        elif kind == "skill_activated":
            self.render_skill_activated(str(payload.get("name", "")))
        elif kind == "memory_record_failed":
            self.render_memory_status(f"record failed: {payload.get('error')}")
        elif kind == "session_compaction_finished":
            self.render_memory_status(f"compacted {payload.get('kind')}")
        elif kind == "session_compaction_failed":
            self.render_memory_status(
                f"compaction failed ({payload.get('kind')}): {payload.get('error')}"
            )
        elif self._verbose:
            self._line(self._out, f"Event> {kind} {payload}")

    # -- helpers ----------------------------------------------------------
    def _truncate(self, text: str) -> str:
        if len(text) <= self._reasoning_max:
            return text
        return text[: self._reasoning_max] + "... [truncated]"


def build_renderer(
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    *,
    no_color: bool = False,
    verbose_events: bool = False,
    show_reasoning: bool = False,
    reasoning_display_max_chars: int = 2000,
    theme: Theme = FOREST_THEME,
) -> TerminalRenderer:
    """Pick a renderer (§7.4). Falls back to PlainRenderer when color is off.

    Color is disabled by ``--no-color``, the ``NO_COLOR`` env var, or a
    non-TTY stdout. Step 8 extends this to try ``RichRenderer`` first.
    """
    import os

    color_off = (
        no_color
        or bool(os.environ.get("NO_COLOR"))
        or not getattr(stdout, "isatty", lambda: False)()
    )
    plain = PlainRenderer(
        stdout,
        stderr,
        verbose_events=verbose_events,
        show_reasoning=show_reasoning,
        reasoning_display_max_chars=reasoning_display_max_chars,
    )
    if color_off:
        return plain
    try:
        from .rich_renderer import RichRenderer

        return RichRenderer(
            stdout,
            stderr,
            theme=theme,
            verbose_events=verbose_events,
            show_reasoning=show_reasoning,
            reasoning_display_max_chars=reasoning_display_max_chars,
        )
    except ImportError:
        return plain  # rich not installed -> plain fallback (§7.4)
