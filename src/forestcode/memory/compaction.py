"""Shared helpers for session compaction entries."""

from __future__ import annotations

from .types import MemoryEntry


def effective_compactions(entries: list[MemoryEntry]) -> list[tuple[int, MemoryEntry]]:
    """Return compactions active under the latest-major-wins rule."""
    compactions = [(index, entry) for index, entry in enumerate(entries) if entry.kind == "compaction"]
    latest_major = -1
    for index, entry in compactions:
        if entry.metadata.get("compaction_kind") == "major":
            latest_major = index
    if latest_major < 0:
        return compactions
    return [(index, entry) for index, entry in compactions if index >= latest_major]
