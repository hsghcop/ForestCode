"""Shared YAML frontmatter parsing for Markdown-based manifests (Skills, later Subagents).

Contract (design §Skills runtime):
- frontmatter is a leading ``---`` block; the body is everything after the
  closing ``---``, preserved as original Markdown.
- the frontmatter must parse as a YAML mapping; anything else is an error.
- errors carry the offending line when the YAML parser can report one.
- ``yaml.safe_load`` only; arbitrary YAML tags / objects are rejected by design.
"""

from __future__ import annotations

from typing import Any

import yaml
from yaml import YAMLError


class FrontmatterError(ValueError):
    """Raised when a Markdown manifest has invalid or missing frontmatter."""

    def __init__(self, message: str, *, line: int | None = None) -> None:
        self.line = line
        location = f"line {line}" if line is not None else "frontmatter"
        super().__init__(f"{location}: {message}")


def _normalize_newlines(text: str) -> str:
    # \r\n and lone \r -> \n so delimiter detection and the body survive CRLF
    # files without leaking stray carriage returns into the model-visible body.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Parse leading YAML frontmatter from Markdown text.

    Returns ``(frontmatter, body)``. The body keeps its original Markdown
    content (the closing delimiter is not part of it). Raises
    ``FrontmatterError`` when the leading block is missing, unterminated,
    malformed YAML, or not a mapping.
    """
    text = _normalize_newlines(text)
    first_newline = text.find("\n")
    first_line = text[:first_newline] if first_newline != -1 else text
    if first_line.strip() != "---":
        raise FrontmatterError("missing leading '---' frontmatter")

    rest = text[first_newline + 1 :] if first_newline != -1 else ""
    lines = rest.split("\n")
    end: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == "---":
            end = index
            break
    if end is None:
        raise FrontmatterError("unterminated frontmatter (missing closing '---')")

    block = "\n".join(lines[:end])
    body = "\n".join(lines[end + 1 :])
    try:
        value = yaml.safe_load(block)
    except YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        problem = getattr(exc, "problem", None)
        if not problem:
            problem = str(exc)
        line = mark.line + 1 if mark is not None else None
        raise FrontmatterError(f"invalid YAML: {problem}", line=line) from exc

    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise FrontmatterError(f"frontmatter must be a mapping, got {type(value).__name__}")
    return value, body
