"""Optional rich-backed renderer (§7.3 / §7.4).

Subclasses `PlainRenderer` and overrides only the styled primitives, so the
`on_event` dispatch and empty-text/skip rules stay in one place. Selected by
`build_renderer` only when color is on and ``rich`` is importable; otherwise the
plain fallback is used. The backend never imports this module.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TextIO

from .renderer import FOREST_THEME, FrontendState, PlainRenderer, Theme


class RichRenderer(PlainRenderer):
    def __init__(
        self,
        stdout: TextIO = sys.stdout,
        stderr: TextIO = sys.stderr,
        *,
        theme: Theme = FOREST_THEME,
        verbose_events: bool = False,
        show_reasoning: bool = False,
        reasoning_display_max_chars: int = 2000,
    ) -> None:
        super().__init__(
            stdout,
            stderr,
            verbose_events=verbose_events,
            show_reasoning=show_reasoning,
            reasoning_display_max_chars=reasoning_display_max_chars,
        )
        from rich.console import Console

        self._theme = theme
        self._out_console = Console(file=stdout, highlight=False, soft_wrap=True)
        self._err_console = Console(file=stderr, highlight=False, soft_wrap=True)
        # Active turn's Live region; None outside a turn (set by begin_turn).
        self._activity = None
        # Rendered key-args per in-flight tool_call_id, so the finished line can
        # echo the same args the started line showed.
        self._tool_args: dict[str, str] = {}

    def _emit(self, console, text: str, style: str | None = None) -> None:
        # Route stdout through the Live region's print_above while a turn is
        # active so the spinner does not fight with scrollback output (§3.3).
        if console is self._out_console and self._activity is not None and self._activity.active:
            from rich.text import Text

            self._activity.print_above(Text(text, style=style) if style else Text(text))
            return
        console.print(text, style=style, markup=False)

    def _scrollback(self, renderable) -> None:
        """Print a rich renderable to scrollback, Live-safe during a turn."""
        if self._activity is not None and self._activity.active:
            self._activity.print_above(renderable)
        else:
            self._out_console.print(renderable)

    @contextmanager
    def pause_live(self):
        """Stop the active turn Live region, yield, then restart it.
        Safe to call when _activity is None (outside a turn)."""
        if self._activity is not None:
            self._activity.stop()
        try:
            yield
        finally:
            if self._activity is not None:
                self._activity.start()

    # -- turn lifecycle (driven by TurnRunner in step 6) ------------------
    def begin_turn(self) -> None:
        from .activity import TurnActivity

        self._activity = TurnActivity(self._out_console, self._theme)
        self._activity.start()

    def end_turn(self) -> None:
        if self._activity is not None:
            self._activity.stop()
            self._activity = None

    def _banner_lines(self) -> list[str]:
        """Big 'ForestCode' wordmark via pyfiglet slant, or a plain title.

        pyfiglet is a pure logo enhancement and is independent of the rich /
        prompt_toolkit degradation tiers (plan §2.1, §17): if it is missing or
        errors, the wordmark degrades to the plain title "ForestCode" with no
        effect on the rest of the experience.
        """
        try:
            from pyfiglet import Figlet

            art = Figlet(font="slant").renderText("ForestCode").rstrip("\n")
            lines = [line for line in art.splitlines() if line.strip()]
            return lines or ["ForestCode"]
        except Exception:  # noqa: BLE001 - any pyfiglet issue falls back to plain title
            return ["ForestCode"]

    def _gradient_logo(self):
        """Render the wordmark with a per-line primary->accent green gradient."""
        from rich.text import Text

        lines = self._banner_lines()
        start = _hex_rgb(self._theme.primary)
        end = _hex_rgb(self._theme.accent)
        logo = Text()
        last = max(len(lines) - 1, 1)
        for index, line in enumerate(lines):
            ratio = index / last
            logo.append(line + "\n", style=_lerp_hex(start, end, ratio))
        return logo

    def render_welcome(self, state: FrontendState) -> None:
        from rich.text import Text

        self._out_console.print(self._gradient_logo())
        subtitle = Text("terminal-first coding agent", style=self._theme.muted)
        self._out_console.print(subtitle)
        # Info lines are also surfaced in the input status bar (plan §1.1.12),
        # but kept here so the rich-only tier (no prompt_toolkit, no status bar)
        # does not lose them.
        # Black+green theme: labels in muted green, values in accent green
        # (avoids the washed-out teal that read as "white" against the logo).
        rows = [
            ("Workspace", str(state.workspace_root), self._theme.accent),
            ("Session", state.session_id or "disabled", self._theme.accent),
            ("Model", state.model_name, self._theme.accent),
        ]
        flags = []
        if state.command_tools_enabled:
            flags.append("command-tools")
        if state.show_reasoning:
            flags.append("show-reasoning")
        if state.debug_context:
            flags.append("debug-context")
        if flags:
            rows.append(("Flags", ", ".join(flags), self._theme.muted))
        info = Text()
        for index, (label, value, value_style) in enumerate(rows):
            if index:
                info.append("\n")
            info.append(f"{label:<9}  ", style=self._theme.muted)
            info.append(value, style=value_style)
        self._out_console.print(info)

    def render_user_error(self, message: str) -> None:
        self._emit(self._err_console, message, self._theme.error)

    def render_system_error(self, message: str) -> None:
        self._emit(self._err_console, message, self._theme.error)

    def render_warning(self, message: str) -> None:
        self._emit(self._out_console, message, self._theme.warning)

    def render_assistant_text(self, text: str) -> None:
        # Empty stop emits assistant_text_received{text: ""}; skip it (§5.4).
        if not text.strip():
            return
        from rich.markdown import Markdown
        from rich.table import Table
        from rich.text import Text

        # The agent's "voice": a bright accent-green `●` badge mirrors the user's
        # `›` marker. A two-column grid keeps the badge on the first row only and
        # aligns wrapped/continuation lines under the body (not under the badge).
        grid = Table.grid(expand=True)
        grid.add_column(width=2)
        grid.add_column(ratio=1)
        grid.add_row(Text("●", style=self._theme.accent), Markdown(text))
        self._scrollback(grid)

    def render_skill_activated(self, name: str) -> None:
        self._emit(
            self._out_console,
            f"Skill> 已加载 {name}",
            self._theme.accent,
        )

    def _reminder(self, glyph: str, label: str, message: str, *, error: bool = False) -> None:
        """A quiet system notice (memory / session / plan).

        Subordinate to the user/AI voices via muted green, but the glyph badge
        sits flush-left so it lines up with the `›` (user) and `●` (AI) markers.
        Errors break the green and go red so they stand out.
        """
        from rich.text import Text

        style = self._theme.error if error else self._theme.muted
        self._scrollback(Text(f"{glyph} {label} · {message}", style=style))

    def render_reasoning(self, text: str, provider: str, kind: str) -> None:
        self._emit(self._out_console, f"Reasoning> {self._truncate(text)}", self._theme.muted)

    def on_event(self, event) -> None:
        # Intercept the two tool events to use the full B1/B2 payload
        # (arguments / data) and the active-tool-line model; everything else
        # uses the shared PlainRenderer dispatch.
        kind = event.type
        payload = event.payload
        if kind == "tool_call_started":
            self._on_tool_started(
                str(payload.get("tool_name", "")),
                str(payload.get("tool_call_id", "")),
                payload.get("arguments"),
            )
            return
        if kind == "tool_call_finished":
            self._on_tool_finished(
                str(payload.get("tool_name", "")),
                str(payload.get("tool_call_id", "")),
                bool(payload.get("ok")),
                payload.get("data"),
            )
            if payload.get("tool_name") == "write_todos" and payload.get("summary"):
                self.render_plan_summary(str(payload["summary"]))
            return
        super().on_event(event)

    def _on_tool_started(self, tool_name: str, tool_call_id: str, arguments) -> None:
        from .tool_display import key_args

        # Remember the rendered key-args so the finished line can echo them.
        self._tool_args[tool_call_id] = key_args(tool_name, arguments)
        if self._activity is not None and self._activity.active:
            self._activity.set_active_tool(tool_name, arguments)
            return
        from .activity import tool_active_line

        self._scrollback(tool_active_line(tool_name, arguments, self._theme))

    def _on_tool_finished(self, tool_name: str, tool_call_id: str, ok: bool, data) -> None:
        from .activity import command_output_renderables, tool_finished_line

        args = self._tool_args.pop(tool_call_id, "")
        line = tool_finished_line(tool_name, ok, data, self._theme, args=args)
        self._scrollback(line)
        for row in command_output_renderables(data, self._theme):
            self._scrollback(row)

    def render_plan_summary(self, summary: str) -> None:
        self._reminder("◷", "plan", summary)

    def render_memory_status(self, message: str) -> None:
        self._reminder("✎", "memory", message, error="failed" in message)

    def render_session_status(self, message: str) -> None:
        self._reminder("⟳", "session", message)

    def render_command_preview(self, preview: str) -> None:
        self._emit(self._out_console, preview, self._theme.warning)

    def render_diff(self, diff: str) -> None:
        for line in diff.splitlines():
            self._emit(self._out_console, line, self._diff_style(line))

    def _diff_style(self, line: str) -> str | None:
        if line.startswith("+"):
            return self._theme.diff_add
        if line.startswith("-"):
            return self._theme.diff_del
        if line.startswith("@@"):
            return self._theme.diff_hunk
        return None

    def render_context_debug(self, model_inputs) -> None:
        if not model_inputs:
            self._emit(self._out_console, "Context> no model request captured", self._theme.muted)
            return
        for index, model_input in enumerate(model_inputs, start=1):
            metadata = model_input.metadata
            sources = metadata.get("context_sources") or []
            tools = metadata.get("selected_tools") or []
            muted = self._theme.muted
            self._emit(self._out_console, f"Context> turn {index}", muted)
            self._emit(self._out_console, f"Context> sources: {', '.join(sources) if sources else '(none)'}", muted)
            self._emit(self._out_console, f"Context> messages: {metadata.get('message_count', len(model_input.messages))}", muted)
            self._emit(self._out_console, f"Context> tools: {', '.join(tools) if tools else '(none)'}", muted)
            self._emit(self._out_console, f"Context> chars: {metadata.get('char_count', 'unknown')}", muted)
            self._emit(self._out_console, f"Context> truncated: {metadata.get('truncated', False)}", muted)

    def newline(self) -> None:
        self._emit(self._out_console, "")


def _hex_rgb(style: str) -> tuple[int, int, int]:
    """Extract an (r, g, b) triple from a theme style string like 'bold #2E8B57'."""
    token = style.split("#", 1)
    if len(token) != 2:
        return (255, 255, 255)
    digits = token[1][:6]
    try:
        return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16))
    except ValueError:
        return (255, 255, 255)


def _lerp_hex(start: tuple[int, int, int], end: tuple[int, int, int], ratio: float) -> str:
    r = round(start[0] + (end[0] - start[0]) * ratio)
    g = round(start[1] + (end[1] - start[1]) * ratio)
    b = round(start[2] + (end[2] - start[2]) * ratio)
    return f"#{r:02X}{g:02X}{b:02X}"
