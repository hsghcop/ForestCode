"""Session-level state for files read by tools."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

from ._path import canonical
from .types import ReadFileState


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class ReadStateStore:
    """Track complete file reads so edit tools can require current context."""

    def __init__(self) -> None:
        self._store: dict[str, ReadFileState] = {}
        self._lock = threading.Lock()

    def record(
        self,
        path: Path,
        content: str,
        mtime: float,
        is_partial: bool,
        offset: int | None = None,
        limit: int | None = None,
    ) -> None:
        resolved = canonical(path)
        state = ReadFileState(
            path=str(resolved),
            content_hash=_content_hash(content),
            mtime=mtime,
            is_partial=is_partial,
            offset=offset,
            limit=limit,
        )
        with self._lock:
            self._store[str(resolved)] = state

    def get(self, path: Path) -> ReadFileState | None:
        resolved = canonical(path)
        with self._lock:
            return self._store.get(str(resolved))

    def clear(self, path: Path) -> None:
        resolved = canonical(path)
        with self._lock:
            self._store.pop(str(resolved), None)
