"""Shared single-selection menus for terminal catalogs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TextIO

MenuOption = tuple[str, str]


def numbered_select(
    options: Sequence[MenuOption],
    *,
    prefix: str,
    header: str,
    input_func: Callable[[str], str],
    stdout: TextIO,
) -> str | None:
    print(header, file=stdout)
    for index, (name, detail) in enumerate(options, start=1):
        print(f"{prefix}> {index}. {name} — {detail}", file=stdout)
    while True:
        answer = input_func(f"{prefix}> select [1-N]: ").strip()
        if not answer:
            return None
        try:
            number = int(answer)
        except ValueError:
            print(f"{prefix}> invalid selection", file=stdout)
            continue
        if 1 <= number <= len(options):
            return options[number - 1][0]
        print(f"{prefix}> invalid selection", file=stdout)


def menu_step(index: int, key: str, count: int) -> tuple[int, str]:
    if key == "up":
        return ((index - 1) % count, "")
    if key == "down":
        return ((index + 1) % count, "")
    if key == "enter":
        return (index, "select")
    if key in ("escape", "c-c"):
        return (index, "cancel")
    return (index, "")


def pt_select(options: Sequence[MenuOption], *, header: str) -> str | None:
    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    state = {"index": 0}

    def get_text():
        lines: list[tuple[str, str]] = [("class:header", f" {header}\n")]
        for index, (name, detail) in enumerate(options):
            prefix = " ▶ " if index == state["index"] else "   "
            style = "class:selected" if index == state["index"] else ""
            lines.append((style, f"{prefix}{name}  {detail}\n"))
        return lines

    kb = KeyBindings()

    @kb.add("up")
    def _(event):
        state["index"], _action = menu_step(state["index"], "up", len(options))

    @kb.add("down")
    def _(event):
        state["index"], _action = menu_step(state["index"], "down", len(options))

    @kb.add("enter")
    def _(event):
        _index, action = menu_step(state["index"], "enter", len(options))
        if action == "select":
            event.app.exit(result="select")

    @kb.add("c-c")
    @kb.add("escape")
    def _(event):
        _index, action = menu_step(state["index"], "escape", len(options))
        if action == "cancel":
            event.app.exit(result="cancel")

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
    if app.run() != "select":
        return None
    return options[state["index"]][0]
