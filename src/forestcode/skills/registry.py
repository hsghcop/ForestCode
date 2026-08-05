"""SkillRegistry: process-local, refreshable view over the two skill roots."""

from __future__ import annotations

import logging
from pathlib import Path

from .loader import SkillLoader
from .types import LoadedSkill, SkillDescriptor, SkillSnapshot

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Owns discovery roots and the current immutable snapshot.

    ``refresh()`` rescans project and user roots (stable order, project wins on
    name collision) and logs each discovered issue once. The returned snapshot
    is immutable and fixed for the run that consumes it; ``load`` re-validates
    at read time.
    """

    def __init__(self, workspace_root: str | Path, user_root: str | Path | None = None) -> None:
        self._workspace_root = Path(workspace_root).resolve()
        # Trusted config root for user-level entries; injectable for tests.
        self._user_root = Path(user_root).resolve() if user_root is not None else Path.home() / ".agents" / "skills"
        self._loader = SkillLoader()
        self._current: SkillSnapshot | None = None

    @property
    def project_root(self) -> Path:
        return self._workspace_root / ".agents" / "skills"

    def refresh(self) -> SkillSnapshot:
        snapshot = self._loader.build_snapshot(
            project_root=self.project_root,
            user_root=self._user_root,
        )
        self._current = snapshot
        for issue in snapshot.issues:
            logger.warning("skills: %s", issue.message)
        return snapshot

    def snapshot(self) -> SkillSnapshot | None:
        return self._current

    def list(self) -> tuple[SkillDescriptor, ...]:
        if self._current is None:
            return ()
        return self._current.descriptors

    def get(self, name: str) -> SkillDescriptor | None:
        if self._current is None:
            return None
        return self._current.get(name)

    def load(self, name: str) -> LoadedSkill | None:
        if self._current is None:
            return None
        return self._current.load(name)

    def load_descriptor(self, descriptor: SkillDescriptor) -> LoadedSkill | None:
        if self._loader is None:
            return None
        return self._loader.load_entry(descriptor)


def issues_text(snapshot: SkillSnapshot) -> str:
    """Diagnostic text for warnings (developer-facing, safe relative paths only)."""
    if not snapshot.issues:
        return ""
    return "\n".join(f"[{issue.code}] {issue.message}" for issue in snapshot.issues)
