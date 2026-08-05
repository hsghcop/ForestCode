"""prompt_toolkit input controller: the growing, rule-wrapped input box (§8).

Non-fullscreen (`Application(full_screen=False)`) so terminal scrollback is
preserved (§1.2). Layout (HSplit):

     slash completions, when active
    ────────────────────────────────   top ─ rule (accent green)
     > user input, multi-line, grows…
    ────────────────────────────────   bottom ─ rule
     session · model · workspace · Enter 提交  Ctrl+Enter/Ctrl+J 换行   status bar

Keys (§8.2): Enter submits · Ctrl+Enter / Ctrl+J / Alt+Enter newline · ↑/↓ history
(or move in the completion menu) · Tab/→ accept completion · Ctrl+C → KeyboardInterrupt ·
Ctrl+D on empty → EOFError.

prompt_toolkit is imported lazily so this module imports cleanly without it; the
pure helpers (`SlashCompleter`, `status_text`) are import-light and unit-tested
without a TTY (§14). Selected only in the full tier by ``build_input_controller``
(step 8); otherwise the plain ``StdinInputController`` is used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .renderer import FOREST_THEME, FrontendState, Theme

if TYPE_CHECKING:
    from prompt_toolkit.completion import Completion
    from prompt_toolkit.history import History

    from forestcode.tools import ApprovalRequest

    from .confirm import Choice, Chooser


def status_text(state: FrontendState | None) -> str:
    """Build the one-line status bar text (§8.1)."""
    hints = "Enter 提交  Ctrl+Enter/Ctrl+J 换行  ↑↓ 历史"
    if state is None:
        return hints
    workspace = Path(state.workspace_root).name or str(state.workspace_root)
    parts = [
        f"session:{state.session_id or 'disabled'}",
        state.model_name,
        workspace,
        hints,
    ]
    return "  ·  ".join(parts)


class SlashCompleter:
    """Completer for slash commands and leading Skill tokens (§8.2).

    ``commands`` is a callable returning ``(name, description)`` pairs so the
    list stays current as sessions change.
    """

    def __init__(
        self,
        commands: Callable[[], Iterable[tuple[str, str]]],
        skills: Callable[[], Iterable[tuple[str, str]]] | None = None,
    ) -> None:
        self._commands = commands
        self._skills = skills

    def matches(self, text: str) -> list[tuple[str, str]]:
        """Pure matching used by both the pt Completer and the unit tests."""
        if text.startswith("/"):
            head = text[1:]
            candidates = self._commands()
        elif text.startswith("$") and self._skills is not None:
            head = text[1:]
            candidates = self._skills()
        else:
            return []
        # Only complete the leading command/Skill token (no task text yet).
        if " " in head:
            return []
        return [(name, desc) for name, desc in candidates if name.startswith(head)]

    def get_completions(self, document, complete_event) -> Iterable[Completion]:  # pragma: no cover - pt glue
        from prompt_toolkit.completion import Completion

        text = document.text_before_cursor
        prefix = "$" if text.startswith("$") else "/"
        head = text[1:] if text.startswith(("/", "$")) else ""
        for name, desc in self.matches(text):
            yield Completion(
                name,
                start_position=-len(head),
                display=f"{prefix}{name}",
                display_meta=desc,
            )


def echo_fragments(text: str) -> list[tuple[str, str]]:
    """Style fragments for the post-submit echo line (§8.1).

    The first physical line gets the `› ` prompt marker; every later line aligns
    with two spaces. Pure (no prompt_toolkit) so it is unit-testable without a
    console.
    """
    fragments: list[tuple[str, str]] = []
    for index, line in enumerate(text.split("\n")):
        fragments.append(("class:prompt", "› " if index == 0 else "  "))
        fragments.append(("", line + "\n"))
    return fragments


def _pt_completer(pure: SlashCompleter | None):
    """Adapt a pure SlashCompleter to a real prompt_toolkit Completer, or None.

    Subclassing ``Completer`` is required: ``Buffer`` calls
    ``get_completions_async``, a method only the base class provides. Passing a
    bare object with just ``get_completions`` raises ``AttributeError:
    'SlashCompleter' object has no attribute 'get_completions_async'`` at
    completion time and crashes the event loop.
    """
    if pure is None:
        return None
    from prompt_toolkit.completion import Completer

    class _PTCompleter(Completer):
        def get_completions(self, document, complete_event):
            yield from pure.get_completions(document, complete_event)

    return _PTCompleter()


def _input_body_children(
    completion_menu: object,
    rule: Callable[[], object],
    input_window: object,
    status: object,
    marker: object | None = None,
) -> list[Any]:
    """Layout order for the prompt body, kept pure so it can be unit-tested.

    ``marker`` is the optional one-shot skill marker window; it renders above the
    input box and is never part of the editable buffer (PRD R5).
    """
    if marker is None:
        return [completion_menu, rule(), input_window, rule(), status]
    return [completion_menu, marker, rule(), input_window, rule(), status]


class PromptToolkitInputController:
    """`InputController` backed by a prompt_toolkit input-box Application (§8.4)."""

    def __init__(
        self,
        *,
        history: History | None = None,
        state_provider: Callable[[], FrontendState | None] | None = None,
        slash_commands: Callable[[], Iterable[tuple[str, str]]] | None = None,
        skill_candidates: Callable[[], Iterable[tuple[str, str]]] | None = None,
        theme: Theme = FOREST_THEME,
        skill_marker_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._history = history
        self._state_provider = state_provider
        self._completer = (
            SlashCompleter(slash_commands or (lambda: ()), skill_candidates)
            if slash_commands is not None or skill_candidates is not None
            else None
        )
        self._theme = theme
        self._skill_marker_provider = skill_marker_provider

    def read_user_input(self, prompt: str) -> str:
        app = self._build_application()
        result = app.run()
        if result is None:
            # Defensive: app.exit() without a result -> treat as EOF.
            raise EOFError
        # The input box is erased on submit (erase_when_done); echo a clean,
        # rule-free `› <text>` line into scrollback so the turn keeps a compact
        # record of what was sent (§8.1).
        self._echo(result)
        return result

    def _echo(self, text: str) -> None:
        from prompt_toolkit import print_formatted_text
        from prompt_toolkit.formatted_text import FormattedText
        from prompt_toolkit.styles import Style

        print_formatted_text(
            FormattedText(echo_fragments(text)),
            style=Style.from_dict({"prompt": self._theme.accent}),
        )

    def read_confirmation(self, prompt: str) -> str:
        # Single-line read for the text confirm fallback (§6.5). The arrow-key
        # menu is a separate chooser; this keeps the default path working inside
        # the prompt_toolkit environment.
        from prompt_toolkit import PromptSession

        session: PromptSession = PromptSession()
        return session.prompt(prompt)

    # -- application construction -----------------------------------------
    def _build_application(self):
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import (
            HSplit,
            Layout,
            Window,
        )
        from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
        from prompt_toolkit.layout.dimension import Dimension
        from prompt_toolkit.layout.menus import CompletionsMenu
        from prompt_toolkit.styles import Style

        # Wrap the pure SlashCompleter in a real pt Completer so the base class
        # supplies get_completions_async (which Buffer actually calls).
        completer = _pt_completer(self._completer)

        buffer = Buffer(
            multiline=True,
            history=self._history,
            completer=completer,
            complete_while_typing=True,
        )

        def line_prefix(line_number, wrap_count):
            from prompt_toolkit.formatted_text import to_formatted_text

            # `›` only at the very first cell of the whole input — not on
            # Ctrl+J newlines (line_number > 0) nor on wrapped continuation
            # rows (wrap_count > 0); those align with two spaces (§8.1).
            head = line_number == 0 and wrap_count == 0
            return to_formatted_text([("class:prompt", "› " if head else "  ")])

        input_window = Window(
            BufferControl(buffer=buffer),
            height=Dimension(min=1),
            wrap_lines=True,
            get_line_prefix=line_prefix,
            # Shrink strictly to the content's line count so the box collapses
            # back when multiline input is deleted (without this it stays at the
            # tallest height it ever reached).
            dont_extend_height=True,
        )
        rule = lambda: Window(height=1, char="─", style="class:rule")
        status = Window(
            FormattedTextControl(lambda: status_text(self._current_state())),
            height=1,
            style="class:status",
        )
        # Render completions as part of the input layout, above the prompt box.
        # This keeps slash suggestions away from the following terminal output
        # without leaking prompt_toolkit positioning details outside the frontend.
        completion_menu = CompletionsMenu(max_height=8, scroll_offset=1)
        # One-shot skill marker (PRD R5): a separate, non-editable control above
        # the input box. It is never part of the editable Buffer, so echo and
        # history contain only real user text.
        marker_window = None
        if self._skill_marker_provider is not None:
            marker_text = self._skill_marker_provider()
            if marker_text:
                marker_window = Window(
                    FormattedTextControl(marker_text + "\n"),
                    height=1,
                    style="class:skill_marker",
                )
        root = HSplit(
            _input_body_children(
                completion_menu, rule, input_window, status, marker_window
            )
        )

        kb = KeyBindings()

        @kb.add("enter")
        def _(event) -> None:
            event.app.exit(result=buffer.text)

        @kb.add("c-j")  # always-works newline (Ctrl+J is literally LF)
        @kb.add("escape", "enter")  # Alt+Enter
        def _(event) -> None:
            buffer.insert_text("\n")

        # Ctrl+Enter is not a bindable key name in every prompt_toolkit build
        # (e.g. 3.0.52 raises "Invalid key: c-enter"), and most terminals send
        # it identically to Enter anyway. Register it defensively so it works
        # where supported without crashing startup where it is not.
        def _newline(event) -> None:
            buffer.insert_text("\n")

        # Ctrl+Enter is not a bindable key name in every prompt_toolkit build
        # (e.g. 3.0.52 raises "Invalid key: c-enter"); register it defensively.
        with suppress(ValueError):
            kb.add("c-enter")(_newline)

        @kb.add("c-c")
        def _(event) -> None:
            event.app.exit(exception=KeyboardInterrupt)

        @kb.add("c-d")
        def _(event) -> None:
            if buffer.text:
                buffer.delete()
            else:
                event.app.exit(exception=EOFError)

        accent = self._theme.accent
        style = Style.from_dict(
            {
                "rule": accent,
                "prompt": accent,
                "status": self._theme.muted,
                # Completion dropdown in the black+green theme: dark bg + green
                # text, selected row green bg + black text, descriptions dim.
                "completion-menu": "bg:#0E1A12",
                "completion-menu.completion": f"bg:#0E1A12 {accent}",
                "completion-menu.completion.current": f"bg:{accent} #0E1A12",
                "completion-menu.meta.completion": f"bg:#0E1A12 {self._theme.muted}",
                "completion-menu.meta.completion.current": f"bg:{accent} #0E1A12",
                "scrollbar.background": "bg:#0E1A12",
                "scrollbar.button": f"bg:{accent}",
            }
        )
        return Application(
            layout=Layout(root, focused_element=input_window),
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,
            # Erase the whole input box (rules + status bar) on submit so it does
            # not pile up in scrollback; read_user_input then echoes a compact
            # `› <text>` line instead (§8.1).
            erase_when_done=True,
        )

    def _current_state(self) -> FrontendState | None:
        if self._state_provider is None:
            return None
        try:
            return self._state_provider()
        except Exception:  # noqa: BLE001 - status bar must never break input
            return None


def make_pt_chooser(
    pause_live: Callable[[], AbstractContextManager],
    *,
    allow_always: bool = True,
) -> Chooser:
    """Return a Chooser that pauses rich Live and shows an inline arrow-key menu."""

    def chooser(request: ApprovalRequest) -> Choice:
        with pause_live():
            return _run_approval_app(request, allow_always=allow_always)

    return chooser


def _build_approval_options(
    request: ApprovalRequest, *, allow_always: bool = True
) -> list[tuple[Choice, str]]:
    """Assemble the option list for the approval menu (pure, no prompt_toolkit)."""
    label = request.path or request.command or request.tool_name
    options: list[tuple[Choice, str]] = [
        ("yes", "Yes    本次允许"),
        ("no", "No     拒绝"),
    ]
    if allow_always and not request.is_dangerous:
        options.append(("always", f"Always 总是允许  {label}"))
    return options


def _run_approval_app(request: ApprovalRequest, *, allow_always: bool = True) -> Choice:
    """Non-fullscreen inline Application: ↑↓ navigate, Enter confirm."""
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    options = _build_approval_options(request, allow_always=allow_always)

    state = {"index": 0}

    def get_text():
        lines: list[tuple[str, str]] = []
        action = request.operation or request.preview[:80]
        lines.append(("class:header", f" {request.tool_name}  {action}\n"))
        for i, (_, display) in enumerate(options):
            prefix = " ▶ " if i == state["index"] else "   "
            opt_style = "class:selected" if i == state["index"] else ""
            lines.append((opt_style, f"{prefix}{display}\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        state["index"] = (state["index"] - 1) % len(options)

    @kb.add("down")
    def _(event):
        state["index"] = (state["index"] + 1) % len(options)

    @kb.add("enter")
    def _(event):
        event.app.exit(result=options[state["index"]][0])

    @kb.add("c-c")
    @kb.add("escape")
    def _(event):
        event.app.exit(result="no")

    style = Style.from_dict(
        {
            "header": "#a8d8a8 bold",
            "selected": "bg:#1a4a1a #a8d8a8 bold",
        }
    )
    app = Application(
        layout=Layout(HSplit([Window(FormattedTextControl(get_text))])),
        key_bindings=kb,
        style=style,
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,
    )
    return app.run() or "no"


def build_history(
    *,
    session_enabled: bool,
    workspace_root: Path,
    no_history: bool = False,
) -> History:
    """Pick the input history backend (§8.3).

    Persistent FileHistory only when a session is enabled AND --no-history is not
    set; otherwise InMemoryHistory (no on-disk trace), honoring the --no-session
    privacy expectation.
    """
    from prompt_toolkit.history import FileHistory, InMemoryHistory

    if session_enabled and not no_history:
        history_dir = Path(workspace_root) / ".forestcode"
        history_dir.mkdir(parents=True, exist_ok=True)
        return FileHistory(str(history_dir / "cli_history"))
    return InMemoryHistory()
