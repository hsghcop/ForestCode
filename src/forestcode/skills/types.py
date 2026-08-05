"""Skill domain types (design §Skills runtime / Data Contracts)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

SkillSource = Literal["project", "user"]


@dataclass(frozen=True, slots=True)
class SkillDescriptor:
    """Validated metadata for one discovered SKILL.md entry.

    ``root`` is the skill directory (the one containing SKILL.md); resources
    are listed relative to it. ``entry_path`` is the absolute SKILL.md path.
    ``source_root`` is the discovery root the skill came from (project
    ``<workspace>/.agents/skills`` or the user skills dir); it anchors
    workspace-relative resource display paths (F2). Both roots are
    containment-verified at discovery time; ``load`` re-verifies.

    ``content_digest`` is the SHA-256 of the raw SKILL.md bytes at discovery
    time. ``load`` recomputes it and fails when the file changed, so a fixed
    snapshot can never load new content under a stale name/description (PRD R6
    review finding F4).
    """

    name: str
    description: str
    root: Path
    entry_path: Path
    source: SkillSource
    source_root: Path | None = None
    content_digest: str = ""


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    """The full, validated body of a skill plus its bounded resource list."""

    descriptor: SkillDescriptor
    instructions: str
    resource_paths: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillIssue:
    """A diagnostic for one skipped or degraded skill.

    ``path`` is a safe relative display (relative to the source root), never an
    absolute user-home path (PRD R7).
    """

    path: str
    code: str
    message: str


class SkillLoaderProtocol(Protocol):
    def load_entry(self, descriptor: SkillDescriptor) -> LoadedSkill | None: ...


@dataclass(frozen=True, slots=True)
class SkillSnapshot:
    """Immutable view of one refresh, fixed for the whole run it feeds.

    ``loader`` lets ``load(name)`` re-read and re-validate an entry against the
    same snapshot without consulting a mutable registry (so a refresh that
    happens between classify and the worker thread cannot change what the run
    sees).
    """

    descriptors: tuple[SkillDescriptor, ...] = ()
    issues: tuple[SkillIssue, ...] = ()
    loader: SkillLoaderProtocol | None = None

    def get(self, name: str) -> SkillDescriptor | None:
        for descriptor in self.descriptors:
            if descriptor.name == name:
                return descriptor
        return None

    def load(self, name: str) -> LoadedSkill | None:
        descriptor = self.get(name)
        if descriptor is None or self.loader is None:
            return None
        return self.loader.load_entry(descriptor)
