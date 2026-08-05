"""Markdown-backed long-term memory reader."""

from __future__ import annotations

import re
from pathlib import Path


MEMORY_TEMPLATE = "# Long-term Memory\n\n"
MEMORY_HEADER_RE = re.compile(r"^## \[[^\]\n]+\] (?P<name>[a-z0-9][a-z0-9-]*)\s*$", re.MULTILINE)


def render_upserted_content(existing_text: str, name: str, memory_type: str, content: str) -> str:
    """Return MEMORY.md content with one named section inserted or replaced."""
    text = existing_text if existing_text else MEMORY_TEMPLATE
    section = _render_section(name, memory_type, content)

    for match in MEMORY_HEADER_RE.finditer(text):
        if match.group("name") != name:
            continue
        section_end = _next_section_start(text, match.end())
        suffix = text[section_end:]
        if suffix:
            return text[: match.start()] + section.rstrip() + "\n\n---\n\n" + suffix.lstrip()
        return text[: match.start()] + section

    base = text.rstrip()
    if not base:
        return section
    separator = "\n\n---\n\n" if MEMORY_HEADER_RE.search(text) else "\n\n"
    return f"{base}{separator}{section}"


def _render_section(name: str, memory_type: str, content: str) -> str:
    return f"## [{memory_type}] {name}\n{content.strip()}\n"


def _next_section_start(text: str, position: int) -> int:
    next_match = re.search(r"(?m)^## ", text[position:])
    if next_match is None:
        return len(text)
    return position + next_match.start()


class MarkdownMemory:
    def __init__(self, workspace_root: str | Path, filename: str = "MEMORY.md") -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.path = self.workspace_root / filename

    def read(self, max_chars: int | None = None) -> str:
        if not self.path.exists():
            return ""
        text = self.path.read_text(encoding="utf-8", errors="replace")
        if max_chars is None:
            return text
        if len(text) <= max_chars:
            return text
        marker = f"\n...<truncated {len(text) - max_chars} chars>"
        keep = max(max_chars - len(marker), 0)
        return text[:keep] + marker
