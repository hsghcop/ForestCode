"""Pure formatting helpers for tool start/finish lines (plan §4, §5).

The backend stays presentation-agnostic: ``tool_call_started`` carries raw
``arguments`` (B1) and ``tool_call_finished`` carries raw ``data`` (B2). All
per-tool display logic — which argument to surface, how to phrase the metric,
how to truncate command output — lives here as side-effect-free functions so it
is exhaustively unit-testable without a TTY (plan §14).
"""

from __future__ import annotations

from typing import Any

# Command preview width and output head budget (plan §4.1, §5).
KEY_ARG_MAX_CHARS = 60
COMMAND_OUTPUT_HEAD_LINES = 10


def key_args(tool_name: str, arguments: dict[str, Any] | None) -> str:
    """Return the key display argument for a tool's start/finish line (§4.1).

    Never raises on odd input; unknown tools fall back to the first of
    path/command/pattern/query present, else "".
    """
    args = arguments or {}

    if tool_name in {"read_file", "write_file", "edit_file", "list_files"}:
        return _clip(_str(args.get("path")))
    if tool_name == "run_command":
        return _clip(_str(args.get("command")))
    if tool_name == "grep":
        pattern = _str(args.get("pattern"))
        shown = f'"{pattern}"' if pattern else ""
        scope = _str(args.get("glob")) or _str(args.get("path"))
        if shown and scope:
            return _clip(f"{shown} {scope}")
        return _clip(shown or scope)
    if tool_name == "glob":
        return _clip(_str(args.get("pattern")))
    if tool_name == "write_todos":
        todos = args.get("todos")
        if isinstance(todos, list):
            return f"({len(todos)} items)"
        return ""

    # Unknown / other tools: first present neutral field.
    for field in ("path", "command", "pattern", "query"):
        value = _str(args.get(field))
        if value:
            return _clip(value)
    return ""


def metric(tool_name: str, data: dict[str, Any] | None) -> str:
    """Return the parenthesized metric for a finish line, or "" (§4.2).

    Reads only the structured ``data`` passed through by the backend (B2).
    """
    if not data:
        return ""

    if tool_name == "run_command":
        exit_code = data.get("exit_code")
        if exit_code is not None:
            return f"(exit {exit_code})"
        return ""

    if tool_name in {"edit_file", "write_file"}:
        diff = data.get("diff")
        if isinstance(diff, str) and diff:
            added, removed = _count_diff(diff)
            return f"({added}+/{removed}-)"
        return ""

    if tool_name == "grep":
        lines = data.get("lines")
        if isinstance(lines, int):
            return f"({lines} matches)"
        return ""

    if tool_name in {"list_files", "glob"}:
        lines = data.get("lines")
        if isinstance(lines, int):
            return f"({lines} entries)"
        return ""

    # Generic read tools: line count.
    lines = data.get("lines")
    if isinstance(lines, int):
        return f"({lines} lines)"
    return ""


def command_output_lines(
    data: dict[str, Any] | None,
    *,
    head: int = COMMAND_OUTPUT_HEAD_LINES,
) -> list[tuple[str, bool]]:
    """Return ``(text, is_stderr)`` rows for a command's output block (§5).

    stdout first, then stderr; truncated to ``head`` total rows with a trailing
    ``… N more lines`` marker (reported as ``(marker, False)``). Only commands
    carry ``stdout``/``stderr`` in ``data`` (B2); other tools yield no rows.
    """
    if not data:
        return []

    rows: list[tuple[str, bool]] = []
    for text, is_stderr in ((data.get("stdout"), False), (data.get("stderr"), True)):
        if not isinstance(text, str) or not text:
            continue
        for line in text.splitlines():
            rows.append((line, is_stderr))

    if len(rows) <= head:
        return rows
    remaining = len(rows) - head
    kept = rows[:head]
    kept.append((f"… {remaining} more lines", False))
    return kept


# -- helpers ---------------------------------------------------------------
def _str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _clip(text: str, limit: int = KEY_ARG_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _count_diff(diff: str) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return added, removed
