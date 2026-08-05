"""Subagent configuration discovery and validation (design §Configuration Contract).

- Roots: project ``<workspace>/.agents/subagents/`` and user ``~/.agents/subagents/``.
- Markdown (YAML frontmatter, body = instructions) and JSON use the same nested
  field structure; the project root wins over the user root on a name collision
  and a same-source duplicate name (including ``.md``/``.json``) is ambiguous
  and excluded with an issue.
- One broken file only produces an ``AgentIssue``; it never blocks other agents.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from forestcode.config.frontmatter import FrontmatterError, parse_frontmatter
from forestcode.models.types import ModelAdapterError, ModelConfig

# Agent name rule is explicitly "the same 1-64 lowercase token rule as Skills";
# reuse the skills regex so the two contracts can never drift apart.
from forestcode.skills.loader import NAME_RE
from forestcode.skills.types import SkillSnapshot

from .types import (
    DEFAULT_TASK_TIMEOUT_SECONDS,
    INHERIT,
    MAX_AGENT_CATALOG_CHARS,
    MAX_DEFAULT_SKILLS,
    MAX_DESCRIPTION_CHARS,
    MAX_INSTRUCTIONS_CHARS,
    MAX_MODEL_BASE_URL_CHARS,
    MAX_MODEL_FIELD_CHARS,
    MAX_NAME_CHARS,
    MAX_TASK_TIMEOUT_SECONDS,
    MAX_TOOLS_ALLOW,
    MAX_TOOLS_DENY,
    MIN_TASK_TIMEOUT_SECONDS,
    VALID_PROFILES,
    AgentConfig,
    AgentConfigSet,
    AgentIssue,
    AgentSource,
    ModelOverride,
    ToolsSpec,
)

logger = logging.getLogger(__name__)

# design §Discovery: one config file is at most 64 KiB; at most 100 candidates.
MAX_FILE_BYTES = 64_000
MAX_CATALOG_AGENTS = 100

_ALLOWED_FIELDS = frozenset(
    {
        "name",
        "description",
        "instructions",
        "permission_profile",
        "tools",
        "default_skills",
        "model",
        "task_timeout_seconds",
    }
)
_ALLOWED_TOOLS_FIELDS = frozenset({"allow", "deny"})
_ALLOWED_MODEL_FIELDS = frozenset(
    {
        "api_type",
        "model",
        "base_url",
        "timeout",
        "reasoning_mode",
        "reasoning_effort",
        "api_key_env",
    }
)


class _ConfigFormatError(ValueError):
    """Format/schema problem in one config file, carrying a stable issue code."""

    def __init__(self, message: str, code: str = "format") -> None:
        super().__init__(message)
        self.code = code


def _as_finite_positive_timeout(value) -> float:
    """Convert a YAML/JSON timeout to a finite positive float, else a config error."""
    if isinstance(value, bool):
        raise _ConfigFormatError(
            "model.timeout must be a positive finite number", "invalid_model"
        )
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _ConfigFormatError(
            "model.timeout must be a positive finite number", "invalid_model"
        ) from None
    if not math.isfinite(result) or result <= 0:
        raise _ConfigFormatError(
            "model.timeout must be a positive finite number", "invalid_model"
        )
    return result


def _parse_markdown(text: str) -> tuple[dict, str]:
    """Parse a Markdown agent config: frontmatter fields + body as instructions."""
    try:
        frontmatter, body = parse_frontmatter(text)
    except FrontmatterError as exc:
        raise _ConfigFormatError(str(exc), "frontmatter") from exc
    if "instructions" in frontmatter:
        raise _ConfigFormatError(
            "instructions must live in the Markdown body, not the frontmatter",
            "instructions_in_frontmatter",
        )
    return frontmatter, body


def _parse_json(text: str) -> tuple[dict, str]:
    """Parse a JSON agent config: object fields, ``instructions`` required."""
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ConfigFormatError(f"invalid JSON: {exc}", "json") from exc
    if not isinstance(value, dict):
        raise _ConfigFormatError(
            f"JSON root must be an object, got {type(value).__name__}", "json"
        )
    instructions = value.get("instructions")
    if not isinstance(instructions, str):
        raise _ConfigFormatError(
            "instructions must be a string", "missing_instructions"
        )
    return value, instructions


@dataclass(frozen=True, slots=True)
class _ParsedEntry:
    config: AgentConfig
    display_path: str


class AgentConfigLoader:
    """Pure discovery/validation; no mutable catalog state (registry owns that)."""

    def build_set(self, *, project_root: Path, user_root: Path) -> AgentConfigSet:
        issues: list[AgentIssue] = []
        project_candidates = self._discover(project_root, "project", issues)
        user_candidates = self._discover(user_root, "user", issues)
        project_valid = self._resolve_ambiguity(project_candidates, "project", issues)
        user_valid = self._resolve_ambiguity(user_candidates, "user", issues)
        merged: dict[str, AgentConfig] = dict(project_valid)
        for name, config in user_valid.items():
            merged.setdefault(name, config)
        agents = self._apply_catalog_limit(
            sorted(merged.values(), key=lambda a: a.name), issues
        )
        return AgentConfigSet(agents={a.name: a for a in agents}, issues=tuple(issues))

    # -- discovery ---------------------------------------------------------
    def _discover(
        self,
        root: Path,
        source: AgentSource,
        issues: list[AgentIssue],
    ) -> dict[str, list[_ParsedEntry]]:
        candidates: dict[str, list[_ParsedEntry]] = {}
        if not root.is_dir():
            return candidates
        for path in self._walk_files(root):
            entry = self._parse_file(path, root, source, issues)
            if entry is None:
                continue
            candidates.setdefault(entry.config.name, []).append(entry)
        return candidates

    def _walk_files(self, root: Path):
        """Yield regular ``.md``/``.json`` files in stable sorted order.

        Symlinked files and directories are never followed (defense in depth:
        the ``resolve()`` containment check below would also catch escapes, but
        skipping links keeps discovery deterministic and confined to the root).
        Hidden directories are skipped, matching the Skills loader.
        """
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                children = sorted(current.iterdir(), key=lambda p: p.name)
            except OSError:
                continue
            for child in children:
                if (
                    child.is_dir()
                    and not child.is_symlink()
                    and not child.name.startswith(".")
                ):
                    stack.append(child)
                elif (
                    child.is_file()
                    and not child.is_symlink()
                    and child.suffix.lower() in (".md", ".json")
                ):
                    yield child

    def _parse_file(
        self,
        path: Path,
        root: Path,
        source: AgentSource,
        issues: list[AgentIssue],
    ) -> _ParsedEntry | None:
        rel = path.relative_to(root).as_posix()
        try:
            resolved = path.resolve()
            resolved.relative_to(root.resolve())
        except (OSError, ValueError):
            issues.append(
                AgentIssue(
                    rel, "escape", f"{source}:{rel}: config file escapes its root"
                )
            )
            return None
        try:
            if resolved.stat().st_size > MAX_FILE_BYTES:
                issues.append(
                    AgentIssue(
                        rel,
                        "too_large",
                        f"{source}:{rel}: file exceeds {MAX_FILE_BYTES} bytes",
                    )
                )
                return None
            raw = resolved.read_bytes()
        except OSError as exc:
            issues.append(AgentIssue(rel, "unreadable", f"{source}:{rel}: {exc}"))
            return None
        # Strict UTF-8: a non-UTF-8 config is a discovery-time issue, never
        # silently accepted through replacement characters.
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            issues.append(
                AgentIssue(
                    rel,
                    "invalid_encoding",
                    f"{source}:{rel}: config is not valid UTF-8 ({exc})",
                )
            )
            return None
        try:
            if path.suffix.lower() == ".json":
                fields, instructions = _parse_json(text)
            else:
                fields, instructions = _parse_markdown(text)
        except _ConfigFormatError as exc:
            issues.append(AgentIssue(rel, exc.code, f"{source}:{rel}: {exc}"))
            return None
        config = self._validate(
            fields, instructions=instructions, rel=rel, source=source, issues=issues
        )
        if config is None:
            return None
        return _ParsedEntry(config=config, display_path=rel)

    # -- validation --------------------------------------------------------
    def _validate(
        self,
        fields: dict,
        *,
        instructions: str,
        rel: str,
        source: AgentSource,
        issues: list[AgentIssue],
    ) -> AgentConfig | None:
        prefix = f"{source}:{rel}"
        unknown = sorted(set(fields) - _ALLOWED_FIELDS)
        if unknown:
            issues.append(
                AgentIssue(
                    rel,
                    "unknown_field",
                    f"{prefix}: unknown field(s): {', '.join(unknown)}",
                )
            )
            return None
        name = fields.get("name")
        if not isinstance(name, str) or not NAME_RE.match(name):
            issues.append(
                AgentIssue(
                    rel,
                    "invalid_name",
                    f"{prefix}: name must match {NAME_RE.pattern} "
                    f"({MAX_NAME_CHARS} chars, lowercase token)",
                )
            )
            return None
        description = fields.get("description")
        if not isinstance(description, str) or not description.strip():
            issues.append(
                AgentIssue(
                    rel,
                    "missing_description",
                    f"{prefix}: description must be a non-empty string",
                )
            )
            return None
        description = description.strip()
        if len(description) > MAX_DESCRIPTION_CHARS:
            issues.append(
                AgentIssue(
                    rel,
                    "description_too_large",
                    f"{prefix}: description exceeds {MAX_DESCRIPTION_CHARS} characters",
                )
            )
            return None
        if not instructions.strip():
            issues.append(
                AgentIssue(
                    rel,
                    "missing_instructions",
                    f"{prefix}: instructions must be a non-empty string",
                )
            )
            return None
        if len(instructions) > MAX_INSTRUCTIONS_CHARS:
            issues.append(
                AgentIssue(
                    rel,
                    "instructions_too_large",
                    f"{prefix}: instructions exceed {MAX_INSTRUCTIONS_CHARS} characters",
                )
            )
            return None
        profile = fields.get("permission_profile", "research")
        if profile not in VALID_PROFILES:
            issues.append(
                AgentIssue(
                    rel,
                    "invalid_profile",
                    f"{prefix}: permission_profile must be one of {sorted(VALID_PROFILES)}",
                )
            )
            return None
        try:
            tools = self._parse_tools(fields.get("tools"))
            default_skills = self._parse_default_skills(fields.get("default_skills"))
            model = self._parse_model(fields.get("model"))
        except _ConfigFormatError as exc:
            issues.append(AgentIssue(rel, exc.code, f"{prefix}: {exc}"))
            return None
        timeout = fields.get("task_timeout_seconds", DEFAULT_TASK_TIMEOUT_SECONDS)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, int)
            or not (MIN_TASK_TIMEOUT_SECONDS <= timeout <= MAX_TASK_TIMEOUT_SECONDS)
        ):
            issues.append(
                AgentIssue(
                    rel,
                    "invalid_timeout",
                    f"{prefix}: task_timeout_seconds must be an integer in "
                    f"[{MIN_TASK_TIMEOUT_SECONDS}, {MAX_TASK_TIMEOUT_SECONDS}]",
                )
            )
            return None
        return AgentConfig(
            name=name,
            description=description,
            instructions=instructions,
            permission_profile=profile,
            tools=tools,
            default_skills=default_skills,
            model=model,
            task_timeout_seconds=timeout,
        )

    def _parse_tools(self, value) -> ToolsSpec:
        if value is None:
            return ToolsSpec()
        if not isinstance(value, dict):
            raise _ConfigFormatError(
                "tools must be a mapping with allow/deny lists", "invalid_tools"
            )
        unknown = sorted(set(value) - _ALLOWED_TOOLS_FIELDS)
        if unknown:
            raise _ConfigFormatError(
                f"unknown tools field(s): {', '.join(unknown)}", "invalid_tools"
            )
        allow = self._parse_tool_list(
            value.get("allow", []), MAX_TOOLS_ALLOW, "tools.allow"
        )
        deny = self._parse_tool_list(
            value.get("deny", []), MAX_TOOLS_DENY, "tools.deny"
        )
        return ToolsSpec(allow=allow, deny=deny)

    def _parse_tool_list(self, value, limit: int, label: str) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise _ConfigFormatError(
                f"{label} must be a list of tool names", "invalid_tools"
            )
        if len(value) > limit:
            raise _ConfigFormatError(
                f"{label} exceeds {limit} entries", "tools_too_many"
            )
        names: list[str] = []
        for item in value:
            if not isinstance(item, str) or not NAME_RE.match(item):
                raise _ConfigFormatError(
                    f"{label} entries must be lowercase tool names matching {NAME_RE.pattern}",
                    "invalid_tool_name",
                )
            if item not in names:
                names.append(item)
        return tuple(names)

    def _parse_default_skills(self, value) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise _ConfigFormatError(
                "default_skills must be a list of skill names", "invalid_default_skills"
            )
        if len(value) > MAX_DEFAULT_SKILLS:
            raise _ConfigFormatError(
                f"default_skills exceeds {MAX_DEFAULT_SKILLS} entries",
                "default_skills_too_many",
            )
        names: list[str] = []
        for item in value:
            if not isinstance(item, str) or not NAME_RE.match(item):
                raise _ConfigFormatError(
                    f"default_skills entries must be skill names matching {NAME_RE.pattern}",
                    "invalid_skill_name",
                )
            if item not in names:
                names.append(item)
        return tuple(names)

    def _parse_model(self, value) -> ModelOverride | None:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise _ConfigFormatError(
                "model must be a mapping of overrides", "invalid_model"
            )
        unknown = sorted(set(value) - _ALLOWED_MODEL_FIELDS)
        if unknown:
            raise _ConfigFormatError(
                f"unknown model field(s): {', '.join(unknown)}", "invalid_model"
            )
        api_type = self._parse_model_string(
            value.get("api_type"), MAX_MODEL_FIELD_CHARS, "model.api_type"
        )
        model = self._parse_model_string(
            value.get("model"), MAX_MODEL_FIELD_CHARS, "model.model"
        )
        base_url = self._parse_model_string(
            value.get("base_url"), MAX_MODEL_BASE_URL_CHARS, "model.base_url"
        )
        api_key_env = self._parse_model_string(
            value.get("api_key_env"), MAX_MODEL_FIELD_CHARS, "model.api_key_env"
        )
        reasoning_mode = self._parse_reasoning_override(
            value, "reasoning_mode", MAX_MODEL_FIELD_CHARS, "model.reasoning_mode"
        )
        reasoning_effort = self._parse_reasoning_override(
            value,
            "reasoning_effort",
            MAX_MODEL_FIELD_CHARS,
            "model.reasoning_effort",
        )
        timeout = value.get("timeout")
        if timeout is not None:
            timeout = _as_finite_positive_timeout(timeout)
        if not any(
            item is not None for item in (api_type, model, base_url, timeout, api_key_env)
        ) and "reasoning_mode" not in value and "reasoning_effort" not in value:
            return None
        return ModelOverride(
            api_type=api_type,
            model=model,
            base_url=base_url,
            timeout=timeout,
            reasoning_mode=reasoning_mode,
            reasoning_effort=reasoning_effort,
            api_key_env=api_key_env,
        )

    @staticmethod
    def _parse_model_string(value, limit: int, label: str) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > limit:
            raise _ConfigFormatError(
                f"{label} must be a non-empty string of at most {limit} characters",
                "invalid_model",
            )
        return value

    @staticmethod
    def _parse_reasoning_override(
        mapping: dict, key: str, limit: int, label: str
    ):
        if key not in mapping:
            return INHERIT
        value = mapping[key]
        if value is None:
            return None
        if not isinstance(value, str) or not value or len(value) > limit:
            raise _ConfigFormatError(
                f"{label} must be null or a non-empty string of at most {limit} characters",
                "invalid_model",
            )
        return value

    # -- merge / limits ----------------------------------------------------
    def _resolve_ambiguity(
        self,
        candidates: dict[str, list[_ParsedEntry]],
        source: AgentSource,
        issues: list[AgentIssue],
    ) -> dict[str, AgentConfig]:
        valid: dict[str, AgentConfig] = {}
        for name, entries in candidates.items():
            if len(entries) > 1:
                paths = ", ".join(entry.display_path for entry in entries)
                issues.append(
                    AgentIssue(
                        "",
                        "ambiguous",
                        f"{source}: duplicate agent name {name!r} at {paths}",
                    )
                )
                continue
            valid[name] = entries[0].config
        return valid

    def _apply_catalog_limit(
        self, agents: list[AgentConfig], issues: list[AgentIssue]
    ) -> list[AgentConfig]:
        if len(agents) <= MAX_CATALOG_AGENTS:
            return agents
        issues.append(
            AgentIssue(
                "",
                "catalog_overflow",
                f"catalog exceeds {MAX_CATALOG_AGENTS} agents; further agents excluded",
            )
        )
        return agents[:MAX_CATALOG_AGENTS]


def resolve_child_model_config(
    parent: ModelConfig,
    override: ModelOverride | None,
    environ: Callable[[str], str | None],
) -> ModelConfig:
    """Resolve a child ModelConfig from the parent plus a per-field override (R8).

    Key rules:

    - provider and base_url unchanged -> inherit the parent API key;
    - provider or base_url changed -> ``api_key_env`` must be configured;
    - an explicit ``api_key_env`` always supplies the key from that variable
      (even when the parent already has a key), and a missing/empty variable
      fails the task;
    - error messages mention only variable *names*, never values.
    """
    api_type = parent.api_type
    model = parent.model
    base_url = parent.base_url
    timeout = parent.timeout
    reasoning_mode = parent.reasoning_mode
    reasoning_effort = parent.reasoning_effort
    api_key_env: str | None = None
    if override is not None:
        if override.api_type is not None:
            api_type = override.api_type
        if override.model is not None:
            model = override.model
        if override.base_url is not None:
            base_url = override.base_url
        if override.timeout is not None:
            timeout = override.timeout
        if override.reasoning_mode is not INHERIT:
            reasoning_mode = override.reasoning_mode
        if override.reasoning_effort is not INHERIT:
            reasoning_effort = override.reasoning_effort
        api_key_env = override.api_key_env
    provider_changed = api_type != parent.api_type or base_url != parent.base_url
    if api_key_env is not None:
        api_key = environ(api_key_env)
        if api_key is None or api_key == "":
            raise ModelAdapterError(
                f"environment variable {api_key_env!r} is required for the child "
                "model config but is not set"
            )
    elif provider_changed:
        raise ModelAdapterError(
            "child model config changes provider or base_url and must declare "
            "model.api_key_env; none is configured"
        )
    else:
        api_key = parent.api_key
    return ModelConfig(
        api_type=api_type,
        model=model,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        reasoning_mode=reasoning_mode,
        reasoning_effort=reasoning_effort,
    )


class AgentRegistry:
    """Process-local, refreshable view over the two subagent config roots.

    Mirrors the SkillRegistry lifecycle: ``refresh()`` rescans both roots and
    caches an immutable ``AgentConfigSet``; later steps fix a snapshot per
    parent run so a mid-run refresh cannot change what the run sees.
    """

    def __init__(
        self, workspace_root: str | Path, user_root: str | Path | None = None
    ) -> None:
        self._workspace_root = Path(workspace_root).resolve()
        # Trusted config root for user-level entries; injectable for tests.
        self._user_root = (
            Path(user_root).resolve()
            if user_root is not None
            else Path.home() / ".agents" / "subagents"
        )
        self._loader = AgentConfigLoader()
        self._current: AgentConfigSet | None = None

    @property
    def project_root(self) -> Path:
        return self._workspace_root / ".agents" / "subagents"

    def refresh(
        self,
        *,
        valid_tool_names: frozenset[str] | None = None,
        skills_snapshot: SkillSnapshot | None = None,
    ) -> AgentConfigSet:
        snapshot = self._loader.build_set(
            project_root=self.project_root, user_root=self._user_root
        )
        if valid_tool_names is not None:
            snapshot = validate_agent_snapshot(
                snapshot,
                valid_tool_names=valid_tool_names,
                skills_snapshot=skills_snapshot,
            )
        self._current = snapshot
        for issue in snapshot.issues:
            logger.warning("subagents: %s", issue.message)
        return snapshot

    def snapshot(self) -> AgentConfigSet | None:
        return self._current

    def get(self, name: str) -> AgentConfig | None:
        if self._current is None:
            return None
        return self._current.get(name)


def format_agent_catalog(agent_set: AgentConfigSet | None) -> str:
    """Return the complete, bounded catalog injected into the parent model."""
    if agent_set is None or not agent_set.agents:
        return "(no configured subagents)"
    return "\n".join(
        f"- {config.name}: {config.description}"
        for config in agent_set.agents.values()
    )


def validate_agent_snapshot(
    snapshot: AgentConfigSet,
    *,
    valid_tool_names: frozenset[str],
    skills_snapshot: SkillSnapshot | None,
) -> AgentConfigSet:
    """Exclude configs that cannot work in this fixed parent-run snapshot."""
    agents: dict[str, AgentConfig] = {}
    issues = list(snapshot.issues)
    catalog_chars = 0
    available_skills = (
        frozenset(descriptor.name for descriptor in skills_snapshot.descriptors)
        if skills_snapshot is not None
        else frozenset()
    )
    for name, config in snapshot.agents.items():
        unknown_tools = sorted(
            (set(config.tools.allow) | set(config.tools.deny)) - valid_tool_names
        )
        if unknown_tools:
            issues.append(
                AgentIssue(
                    name,
                    "unknown_tool",
                    f"agent {name!r}: unknown or unavailable tool(s): {', '.join(unknown_tools)}",
                )
            )
            continue
        missing_skills = sorted(set(config.default_skills) - available_skills)
        if missing_skills:
            issues.append(
                AgentIssue(
                    name,
                    "missing_default_skill",
                    f"agent {name!r}: default skill(s) unavailable in this run: {', '.join(missing_skills)}",
                )
            )
            continue
        line_chars = len(f"- {config.name}: {config.description}\n")
        if catalog_chars + line_chars > MAX_AGENT_CATALOG_CHARS:
            issues.append(
                AgentIssue(
                    name,
                    "catalog_overflow",
                    f"agent {name!r}: excluded because the model-visible catalog exceeds {MAX_AGENT_CATALOG_CHARS} characters",
                )
            )
            continue
        agents[name] = config
        catalog_chars += line_chars
    return AgentConfigSet(agents=agents, issues=tuple(issues))
