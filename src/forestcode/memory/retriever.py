"""Memory retrievers used by the context layer."""

from __future__ import annotations

from pathlib import Path

from .markdown_memory import MarkdownMemory


class DirectMarkdownMemoryRetriever:
    def __init__(self, workspace_root: str | Path, max_chars: int = 8_000) -> None:
        self.memory = MarkdownMemory(workspace_root)
        self.max_chars = max_chars

    def retrieve(self) -> str:
        return self.memory.read(max_chars=self.max_chars)
