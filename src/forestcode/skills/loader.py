"""Skill discovery and content loading (PRD R1/R2, R3, R7).

- Scans project ``<workspace>/.agents/skills/`` and user ``~/.agents/skills/``.
- Every skill is a directory containing ``SKILL.md`` with YAML frontmatter.
- Discovery order is stable (canonical-relative path sort); symlinked
  directories are never followed; any resolved entry that escapes its source
  root is rejected.
- One broken file only produces a ``SkillIssue``; it never blocks other skills.

Budget contract (review finding F1): the model-facing text produced by
``format_loaded_skill`` is bounded at discovery time to
``MAX_LOADED_SKILL_CHARS``. The runtime then wraps that text as
``ok:load_skill:<tool_call_id>:<text>`` (``SKILL_RESULT_WRAP_OVERHEAD``) and
ContextProvider truncates at ``ContextBudget.max_skill_result_chars``
(default 20_000). The invariant, asserted by tests:

    MAX_LOADED_SKILL_CHARS + SKILL_RESULT_WRAP_OVERHEAD
        <= ContextBudget.max_skill_result_chars

A skill that fits the discovery bound therefore always fits the runtime budget
with room to spare: a valid skill's instructions, description and resource
list are injected complete, never silently truncated.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from forestcode.config.frontmatter import FrontmatterError, parse_frontmatter
from forestcode.core.types import MAX_TOOL_CALL_ID_CHARS

from .types import LoadedSkill, SkillDescriptor, SkillIssue, SkillSnapshot, SkillSource

logger = logging.getLogger(__name__)

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

# PRD R2 / design §Discovery: hard bounds on one skill and the catalog.
MAX_BODY_CHARS = 16_000
MAX_CATALOG_SKILLS = 100
MAX_CATALOG_DESC_CHARS = 10_000
MAX_RESOURCES = 20
# Single description upper bound so the catalog budget is composable (a 16k
# body plus a very long description could otherwise overflow the runtime
# budget even though each field is individually "valid").
MAX_DESCRIPTION_CHARS = 2_000
# Safety ceiling for a whole SKILL.md file (body bound is the real contract).
MAX_FILE_BYTES = 64_000
ENTRY_FILENAME = "SKILL.md"

# Budget contract (F1): maximum characters of model-facing text produced by
# format_loaded_skill at discovery time. Chosen well below the runtime budget
# (ContextBudget.max_skill_result_chars, default 20_000) so the runtime
# wrapper (ok:load_skill:<call_id>:) plus any provider padding still fits
# without truncation.
MAX_LOADED_SKILL_CHARS = 19_000
# ToolCall enforces this shared upper bound at construction, and model adapters
# convert violations to ModelAdapterError. The wrapper budget is therefore a
# real cross-layer invariant rather than an estimate about provider behavior.
SKILL_RESULT_WRAP_OVERHEAD = len("ok:load_skill:") + MAX_TOOL_CALL_ID_CHARS + 1


class SkillLoader:
    """Pure discovery/loading logic; no mutable catalog state (registry owns that)."""

    def build_snapshot(
        self,
        *,
        project_root: Path,
        user_root: Path,
    ) -> SkillSnapshot:
        issues: list[SkillIssue] = []
        project_candidates = self._discover(project_root, "project", issues)
        user_candidates = self._discover(user_root, "user", issues)
        project_valid = self._resolve_ambiguity(project_candidates, project_root, "project", issues)
        user_valid = self._resolve_ambiguity(user_candidates, user_root, "user", issues)
        merged: dict[str, SkillDescriptor] = dict(project_valid)
        for name, descriptor in user_valid.items():
            merged.setdefault(name, descriptor)
        descriptors = self._apply_catalog_limits(
            sorted(merged.values(), key=lambda d: d.name), issues
        )
        return SkillSnapshot(
            descriptors=tuple(descriptors),
            issues=tuple(issues),
            loader=self,
        )

    # -- discovery --------------------------------------------------------
    def _discover(
        self,
        root: Path,
        source: SkillSource,
        issues: list[SkillIssue],
    ) -> dict[str, list[SkillDescriptor]]:
        candidates: dict[str, list[SkillDescriptor]] = {}
        if not root.is_dir():
            return candidates
        for skill_dir in self._walk_skill_dirs(root):
            entry = skill_dir / ENTRY_FILENAME
            if not entry.is_file():
                continue
            descriptor = self._parse_entry(entry, root, source, issues)
            if descriptor is None:
                continue
            candidates.setdefault(descriptor.name, []).append(descriptor)
        return candidates

    def _walk_skill_dirs(self, root: Path):
        """Yield every directory (recursively) that contains a SKILL.md.

        Stable order via sorted children; symlinked directories and hidden
        directories are skipped so discovery cannot escape or get confused.
        """
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                children = sorted(current.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            has_skill = False
            for child in children:
                if child.name == ENTRY_FILENAME and child.is_file():
                    has_skill = True
                elif (
                    child.is_dir()
                    and not child.is_symlink()
                    and not child.name.startswith(".")
                ):
                    stack.append(child)
            if has_skill:
                yield current

    def _parse_entry(
        self,
        entry: Path,
        root: Path,
        source: SkillSource,
        issues: list[SkillIssue],
    ) -> SkillDescriptor | None:
        rel = entry.relative_to(root).as_posix()
        try:
            resolved = entry.resolve()
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            issues.append(
                SkillIssue(rel, "escape", f"{source}:{rel}: skill entry escapes its root")
            )
            return None
        try:
            if resolved.stat().st_size > MAX_FILE_BYTES:
                issues.append(
                    SkillIssue(rel, "too_large", f"{source}:{rel}: file exceeds {MAX_FILE_BYTES} bytes")
                )
                return None
            raw = resolved.read_bytes()
        except OSError as exc:
            issues.append(SkillIssue(rel, "unreadable", f"{source}:{rel}: {exc}"))
            return None
        # Strict UTF-8 (F5): a non-UTF-8 SKILL.md is a discovery-time issue,
        # never silently accepted through replacement characters.
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            issues.append(
                SkillIssue(rel, "invalid_encoding", f"{source}:{rel}: SKILL.md is not valid UTF-8 ({exc})")
            )
            return None
        # Content identity (F4): SHA-256 of the raw bytes recorded at discovery
        # time; load re-verifies so a stale snapshot cannot serve new content
        # under an old name/description.
        digest = hashlib.sha256(raw).hexdigest()
        try:
            frontmatter, body = parse_frontmatter(text)
        except FrontmatterError as exc:
            issues.append(SkillIssue(rel, "frontmatter", f"{source}:{rel}: {exc}"))
            return None

        name = frontmatter.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            issues.append(
                SkillIssue(
                    rel,
                    "invalid_name",
                    f"{source}:{rel}: frontmatter name must match {NAME_RE.pattern}",
                )
            )
            return None
        description = frontmatter.get("description")
        if not isinstance(description, str) or not description.strip():
            issues.append(
                SkillIssue(rel, "missing_description", f"{source}:{rel}: description must be a non-empty string")
            )
            return None
        if len(description.strip()) > MAX_DESCRIPTION_CHARS:
            issues.append(
                SkillIssue(
                    rel,
                    "description_too_large",
                    f"{source}:{rel}: description exceeds {MAX_DESCRIPTION_CHARS} characters",
                )
            )
            return None
        if len(body) > MAX_BODY_CHARS:
            issues.append(
                SkillIssue(
                    rel,
                    "body_too_large",
                    f"{source}:{rel}: instructions exceed {MAX_BODY_CHARS} characters",
                )
            )
            return None
        descriptor = SkillDescriptor(
            name=name,
            description=description.strip(),
            root=resolved.parent,
            entry_path=resolved,
            source=source,
            source_root=root.resolve(),
            content_digest=digest,
        )
        # Budget contract (F1): the model-facing text of this skill plus its
        # resource list must fit MAX_LOADED_SKILL_CHARS at discovery time. If a
        # valid body/description combination would not fit the runtime budget,
        # reject it here with a clear issue instead of silently truncating it
        # later in the context layer.
        resource_paths = self._list_resources(descriptor)
        loaded = LoadedSkill(
            descriptor=descriptor,
            instructions=body,
            resource_paths=resource_paths,
        )
        if len(format_loaded_skill(loaded)) > MAX_LOADED_SKILL_CHARS:
            issues.append(
                SkillIssue(
                    rel,
                    "loaded_too_large",
                    f"{source}:{rel}: skill would exceed {MAX_LOADED_SKILL_CHARS} chars after formatting",
                )
            )
            return None
        return descriptor

    # -- merge / limits ---------------------------------------------------
    def _resolve_ambiguity(
        self,
        candidates: dict[str, list[SkillDescriptor]],
        root: Path,
        source: SkillSource,
        issues: list[SkillIssue],
    ) -> dict[str, SkillDescriptor]:
        valid: dict[str, SkillDescriptor] = {}
        for name, descriptors in candidates.items():
            if len(descriptors) > 1:
                paths = ", ".join(
                    d.entry_path.relative_to(root.resolve()).as_posix()
                    for d in descriptors
                )
                issues.append(
                    SkillIssue(
                        "",
                        "ambiguous",
                        f"{source}: duplicate skill name {name!r} at {paths}",
                    )
                )
                continue
            valid[name] = descriptors[0]
        return valid

    def _apply_catalog_limits(
        self,
        descriptors: list[SkillDescriptor],
        issues: list[SkillIssue],
    ) -> list[SkillDescriptor]:
        result: list[SkillDescriptor] = []
        total_desc_chars = 0
        for descriptor in descriptors:
            if len(result) >= MAX_CATALOG_SKILLS:
                issues.append(
                    SkillIssue(
                        "",
                        "catalog_overflow",
                        f"catalog exceeds {MAX_CATALOG_SKILLS} skills; further skills excluded",
                    )
                )
                break
            desc_chars = len(descriptor.description)
            if total_desc_chars + desc_chars > MAX_CATALOG_DESC_CHARS:
                issues.append(
                    SkillIssue(
                        "",
                        "catalog_desc_budget",
                        f"catalog description budget {MAX_CATALOG_DESC_CHARS} exceeded; further skills excluded",
                    )
                )
                break
            total_desc_chars += desc_chars
            result.append(descriptor)
        return result

    # -- loading ----------------------------------------------------------
    def load_entry(self, descriptor: SkillDescriptor) -> LoadedSkill | None:
        """Read and re-validate one skill entry; list its bounded resources.

        Re-verifies containment and bounds at read time (defense in depth: the
        file may have changed since discovery, or a symlink may have been
        swapped in). Content identity (F4) is enforced by re-hashing the raw
        bytes: if the file changed since discovery the load fails rather than
        serving new content under the old descriptor. Returns None instead of
        raising when the entry is no longer valid.
        """
        try:
            resolved = descriptor.entry_path.resolve()
            resolved.relative_to(descriptor.root.resolve())
            if resolved.stat().st_size > MAX_FILE_BYTES:
                return None
            raw = resolved.read_bytes()
        except (OSError, ValueError):
            return None
        # Strict UTF-8 (F5): non-UTF-8 content fails the load, never replaced.
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None
        # Content identity (F4): the file must be byte-identical to discovery.
        if descriptor.content_digest and hashlib.sha256(raw).hexdigest() != descriptor.content_digest:
            return None
        try:
            frontmatter, body = parse_frontmatter(text)
        except FrontmatterError:
            return None
        name = frontmatter.get("name")
        description = frontmatter.get("description")
        if not isinstance(name, str) or not NAME_RE.match(name):
            return None
        if not isinstance(description, str) or not description.strip():
            return None
        if len(description.strip()) > MAX_DESCRIPTION_CHARS:
            return None
        if len(body) > MAX_BODY_CHARS:
            return None
        loaded = LoadedSkill(
            descriptor=descriptor,
            instructions=body,
            resource_paths=self._list_resources(descriptor),
        )
        # Resources may change after discovery even when SKILL.md (and therefore
        # its digest) does not. Recheck the complete model-facing payload so
        # newly added or renamed resources cannot re-open the truncation path.
        if len(format_loaded_skill(loaded)) > MAX_LOADED_SKILL_CHARS:
            return None
        return loaded

    def _list_resources(self, descriptor: SkillDescriptor) -> tuple[str, ...]:
        """Recursively list regular files under the skill dir (max 20).

        Excludes SKILL.md, hidden directories, symlinks, and anything outside
        the root (impossible by construction, but symlinks are skipped so a
        swapped-in link cannot reach out).

        Returned paths are directly consumable by the existing file tools
        (F2): project-level skills return workspace-relative paths
        (``.agents/skills/<name>/<sub>``), user-level skills return ``~``
        paths (``~/.agents/skills/<name>/<sub>``) so the sandbox expands them
        to absolute paths and routes them through the normal outside-workspace
        read approval.
        """
        paths: list[str] = []
        stack = [descriptor.root]
        while stack and len(paths) < MAX_RESOURCES:
            current = stack.pop()
            try:
                children = sorted(current.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            for child in children:
                if len(paths) >= MAX_RESOURCES:
                    break
                if child.name == ENTRY_FILENAME:
                    continue
                if child.is_symlink():
                    continue
                if child.is_dir():
                    if child.name.startswith("."):
                        continue
                    stack.append(child)
                    continue
                if child.is_file():
                    rel = child.relative_to(descriptor.root).as_posix()
                    paths.append(self._resource_display_path(descriptor, rel))
        return tuple(sorted(paths))

    @staticmethod
    def _resource_display_path(descriptor: SkillDescriptor, rel: str) -> str:
        """Turn a resource path relative to the skill dir into a tool-readable path.

        Project-level skills produce workspace-relative paths anchored at the
        actual skill directory (``.agents/skills/<rel>/<file>``, preserving
        recursive grouping like ``.agents/skills/group/skill/file``).
        User-level skills produce ``~/.agents/skills/<rel>/<file>`` so the
        sandbox expands ``~`` and routes the read through the normal
        outside-workspace approval.
        """
        if descriptor.source_root is not None:
            try:
                rel_dir = descriptor.root.relative_to(descriptor.source_root).as_posix()
            except ValueError:
                rel_dir = descriptor.root.name
        else:
            rel_dir = descriptor.root.name
        if descriptor.source == "project":
            return f".agents/skills/{rel_dir}/{rel}"
        return f"~/.agents/skills/{rel_dir}/{rel}"


def format_loaded_skill(loaded: LoadedSkill) -> str:
    """Model-facing text for a loaded skill (bounded: body + <= 20 resource paths)."""
    descriptor = loaded.descriptor
    lines = [
        f"Skill: {descriptor.name}",
        f"Description: {descriptor.description}",
        "",
        "Instructions:",
        loaded.instructions,
    ]
    if loaded.resource_paths:
        lines.append("")
        lines.append("Resource files (read them with the read_file tool):")
        lines.extend(f"- {path}" for path in loaded.resource_paths)
    return "\n".join(lines)
