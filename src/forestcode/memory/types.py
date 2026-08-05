"""Serializable memory types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


MemoryEntryKind = Literal["message", "tool_result", "compaction"]


@dataclass(slots=True)
class MemoryEntry:
    kind: MemoryEntryKind
    content: str
    id: str = ""
    role: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SessionMemory:
    session_id: str
    version: int = 2
    title: str = ""
    created_at: str = ""
    updated_at: str = ""
    workspace_root: str = ""
    entries: list[MemoryEntry] = field(default_factory=list)
    runs: list[dict[str, Any]] = field(default_factory=list)
    plan: list[dict[str, Any]] = field(default_factory=list)
