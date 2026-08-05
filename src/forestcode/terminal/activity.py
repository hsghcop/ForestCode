"""Turn-level activity region: spinner + active tool line + Live (plan §3, §9).

During a turn the screen has two layers (plan §3): a scrollback region that
grows normally, and a bottom *live region* (managed by ``rich.live.Live``)
holding an optional active-tool line and the always-present turn spinner. A
finished tool line is printed *above* the live region via ``print_above`` and
the active line is cleared, so each tool call occupies a single line that
updates in place (● → ✓/✗) before scrolling up.

The frame/verb/shimmer/elapsed math is factored into side-effect-free helpers so
it is unit-testable without a TTY (plan §14). ``rich`` is imported lazily so the
pure helpers (and their tests) do not require it. This module is only imported
by ``RichRenderer``; the backend never touches it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .renderer import Theme
from .tool_display import command_output_lines, key_args, metric

# Braille spinner frames and forest verbs (plan §9).
BRAILLE_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
FOREST_VERBS = ("生长中", "抽枝中", "扎根中", "萌发中", "探路中", "结果中")

FRAME_INTERVAL = 0.08  # seconds per braille frame
VERB_INTERVAL = 2.0  # seconds per verb rotation
SHIMMER_INTERVAL = 0.12  # seconds per shimmer step


# -- pure spinner math (no rich, no I/O) -----------------------------------
def spinner_frame(elapsed: float, *, frames: str = BRAILLE_FRAMES, interval: float = FRAME_INTERVAL) -> str:
    return frames[int(elapsed / interval) % len(frames)]


def spinner_verb(elapsed: float, *, verbs: tuple[str, ...] = FOREST_VERBS, interval: float = VERB_INTERVAL) -> str:
    return verbs[int(elapsed / interval) % len(verbs)]


def elapsed_label(elapsed: float) -> str:
    return f"({int(elapsed)}s)"


def shimmer_index(length: int, elapsed: float, *, interval: float = SHIMMER_INTERVAL) -> int:
    """Index of the highlighted character within a verb, sweeping left→right.

    Returns -1 for an empty string so callers can skip highlighting.
    """
    if length <= 0:
        return -1
    return int(elapsed / interval) % length


class SpinnerRenderable:
    """A rich renderable recomputing frame/verb/shimmer/elapsed from a clock.

    Handed to ``Live`` once; ``Live``'s periodic refresh re-invokes
    ``__rich_console__`` so the animation runs without a self-managed thread.
    """

    def __init__(self, theme: Theme, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._theme = theme
        self._clock = clock
        self._start = clock()

    @property
    def elapsed(self) -> float:
        return self._clock() - self._start

    def __rich_console__(self, console: Any, options: Any):
        from rich.text import Text

        elapsed = self.elapsed
        verb = spinner_verb(elapsed)
        hi = shimmer_index(len(verb), elapsed)

        text = Text()
        text.append(spinner_frame(elapsed) + " ", style=self._theme.primary)
        for index, char in enumerate(verb):
            # The shimmer char is rendered one notch brighter (bold) than the
            # rest of the verb.
            style = f"bold {self._theme.accent}" if index == hi else self._theme.primary
            text.append(char, style=style)
        text.append(" " + elapsed_label(elapsed), style=self._theme.muted)
        yield text


def tool_active_line(tool_name: str, arguments: dict[str, Any] | None, theme: Theme):
    """The ``● <tool> <args>`` active line (plan §4.1)."""
    from rich.text import Text

    line = Text()
    line.append("● ", style=theme.accent)
    line.append(tool_name, style=theme.primary)
    args = key_args(tool_name, arguments)
    if args:
        line.append("  " + args, style=theme.muted)
    return line


def tool_finished_line(tool_name: str, ok: bool, data: dict[str, Any] | None, theme: Theme, *, args: str = ""):
    """The committed ``✓/✗ <tool> <args> (<metric>)`` line (plan §4.2)."""
    from rich.text import Text

    line = Text()
    if ok:
        line.append("✓ ", style=theme.success)
    else:
        line.append("✗ ", style=theme.error)
    line.append(tool_name, style=theme.primary)
    if args:
        line.append("  " + args, style=theme.muted)
    metric_text = metric(tool_name, data)
    if metric_text:
        line.append("  " + metric_text, style=theme.success if ok else theme.error)
    return line


def command_output_renderables(data: dict[str, Any] | None, theme: Theme):
    """Indented ``│``-prefixed command output rows (plan §5)."""
    from rich.text import Text

    out = []
    for text, is_stderr in command_output_lines(data):
        row = Text()
        row.append("    │ ", style=theme.muted)
        row.append(text, style=theme.error if is_stderr else None)
        out.append(row)
    return out


class TurnActivity:
    """Owns the bottom Live region for one turn (plan §3.1–§3.3).

    All methods run on the main (UI) thread (plan §7.2). ``print_above`` emits a
    committed line into scrollback above the live region; when the Live region
    is not active it prints directly so callers work in either mode.
    """

    def __init__(self, console: Any, theme: Theme, *, clock: Callable[[], float] = time.monotonic) -> None:
        self._console = console
        self._theme = theme
        self._spinner = SpinnerRenderable(theme, clock=clock)
        self._active_tool: Any | None = None
        self._live: Any | None = None

    @property
    def active(self) -> bool:
        return self._live is not None

    def _group(self):
        from rich.console import Group

        if self._active_tool is not None:
            return Group(self._active_tool, self._spinner)
        return Group(self._spinner)

    def start(self) -> None:
        from rich.live import Live

        self._live = Live(
            self._group(),
            console=self._console,
            refresh_per_second=12,
            transient=True,
        )
        self._live.start()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._group())

    def set_active_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> None:
        self._active_tool = tool_active_line(tool_name, arguments, self._theme)
        self.refresh()

    def clear_active_tool(self) -> None:
        self._active_tool = None
        self.refresh()

    def print_above(self, renderable: Any) -> None:
        if self._live is not None:
            self._live.console.print(renderable)
        else:
            self._console.print(renderable)

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None
