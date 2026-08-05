"""Subagent domain types and pure contracts (design §Configuration Contract, §Runtime Types).

Step 1 of the subagent runtime: config schema, task identity, the child task
state machine, permission profiles and visibility composition. No threads,
scheduling or child construction lives here — the coordinator (later step)
owns mutable task state and derives immutable ``SubagentTaskSnapshot`` values.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Literal

AgentSource = Literal["project", "user"]
PermissionProfile = Literal["research", "verify", "edit", "full"]
SubagentStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "cancelling",
    "cancelled",
]
# design §State machine: fixed cancellation/failure reasons. ``cancelling`` is
# non-terminal and carries the reason that will be finalized as ``cancelled``.
SubagentCancelReason = Literal[
    "requested",
    "parent_finished",
    "parent_failed",
    "parent_aborted",
    "timeout",
    "child_error",
]

# -- field bounds (design §Configuration Contract) -----------------------------
MAX_NAME_CHARS = 64
MAX_DESCRIPTION_CHARS = 2_000
MAX_INSTRUCTIONS_CHARS = 16_000
MAX_PROMPT_CHARS = 16_000
MAX_DEFAULT_SKILLS = 4
MAX_TOOLS_ALLOW = 64
MAX_TOOLS_DENY = 64
MIN_TASK_TIMEOUT_SECONDS = 1
MAX_TASK_TIMEOUT_SECONDS = 3_600
DEFAULT_TASK_TIMEOUT_SECONDS = 600
MAX_MODEL_FIELD_CHARS = 256
MAX_MODEL_BASE_URL_CHARS = 2_048
# Combined hard budget for agent instructions + parent prompt + pre-injected
# Skill bodies. Exceeding it fails the delegation before enqueue; never truncate.
MAX_COMBINED_CONTEXT_CHARS = 24_000
# The complete effective agent catalog is embedded in the parent tool
# description.  Discovery excludes entries beyond this aggregate budget so
# every agent exposed for a run is actually visible to the model.
MAX_AGENT_CATALOG_CHARS = 10_000

# Child delegation tools; structurally removed from every child catalog so a
# child can never create grandchildren (PRD R3).
SUBAGENT_TOOLS = frozenset(
    {"delegate_task", "wait_subagents", "list_subagents", "cancel_subagent"}
)


@dataclass(frozen=True, slots=True)
class ToolsSpec:
    """Nested ``tools.allow`` / ``tools.deny`` overlay (design §Configuration Contract)."""

    allow: tuple[str, ...] = ()
    deny: tuple[str, ...] = ()


class InheritValue(Enum):
    """Explicit marker for an omitted child model override field."""

    INHERIT = "inherit"


INHERIT = InheritValue.INHERIT


@dataclass(frozen=True, slots=True)
class ModelOverride:
    """Per-field model overrides.

    Reasoning fields are deliberately tri-state: ``INHERIT`` means the field
    was omitted, ``None`` means the config explicitly disables it, and a
    string replaces the parent value.

    ``api_key_env`` names an environment variable that supplies the raw API
    key; the key value itself must never appear in any config format (R8).
    """

    api_type: str | None = None
    model: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    reasoning_mode: str | None | InheritValue = INHERIT
    reasoning_effort: str | None | InheritValue = INHERIT
    api_key_env: str | None = None


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """Validated subagent configuration (one file under ``.agents/subagents/``).

    ``permission_profile`` plus ``tools`` only filter model-visible tools; the
    parent ToolCatalog, feature flags and PermissionManager stay the hard
    ceiling at execution time (R6).
    """

    name: str
    description: str
    instructions: str
    permission_profile: PermissionProfile = "research"
    tools: ToolsSpec = field(default_factory=ToolsSpec)
    default_skills: tuple[str, ...] = ()
    model: ModelOverride | None = None
    task_timeout_seconds: int = DEFAULT_TASK_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class AgentIssue:
    """Diagnostic for one skipped/degraded agent config.

    ``path`` is a safe relative display (relative to the source root), never an
    absolute user-home path.
    """

    path: str
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class AgentConfigSet:
    """Immutable view of one refresh (project wins over user on name collision)."""

    agents: Mapping[str, AgentConfig]
    issues: tuple[AgentIssue, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "agents", MappingProxyType(dict(self.agents)))

    def get(self, name: str) -> AgentConfig | None:
        return self.agents.get(name)


# -- task identity (design §Runtime Types) ------------------------------------
TASK_ID_PREFIX = "sub"
VALID_TASK_ID_RE = re.compile(r"^[a-z0-9_-]+$")


def generate_task_id() -> str:
    """Coordinator-owned task id: fixed prefix plus a random safe token.

    Only ``[a-z0-9_-]`` characters, so an id can never smuggle path separators
    or other unsafe text into child transcript filenames (R10).
    """
    return f"{TASK_ID_PREFIX}-{secrets.token_hex(8)}"


def is_valid_task_id(task_id: str) -> bool:
    return bool(VALID_TASK_ID_RE.match(task_id))


@dataclass(frozen=True, slots=True)
class SubagentRequest:
    """Immutable delegation request; ``task_id`` is coordinator-generated, never model-supplied."""

    task_id: str
    agent_name: str
    description: str
    prompt: str


@dataclass(frozen=True, slots=True)
class SubagentTaskSnapshot:
    """Immutable view of one child task for renderers / wait / list results.

    The coordinator keeps mutable task state under a lock and derives these
    snapshots; ``delivered`` marks whether the final body already entered the
    parent session (one-time handoff, R10).
    """

    task_id: str
    agent_name: str
    status: SubagentStatus
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    queue_position: int | None = None
    summary: str | None = None
    error: str | None = None
    cancel_reason: SubagentCancelReason | None = None
    delivered: bool = False


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """Final child outcome; ``delivered`` marks the one-time handoff to the parent history."""

    task_id: str
    agent_name: str
    final_text: str | None = None
    turn_count: int = 0
    tool_count: int = 0
    delivered: bool = False


# -- state machine (design §State Machine) -------------------------------------
TERMINAL_STATUSES: frozenset[SubagentStatus] = frozenset(
    {"completed", "failed", "cancelled"}
)

_TRANSITIONS: frozenset[tuple[SubagentStatus, SubagentStatus]] = frozenset(
    {
        ("queued", "running"),
        ("queued", "cancelled"),
        ("running", "waiting_approval"),
        ("running", "cancelling"),
        ("running", "completed"),
        ("running", "failed"),
        ("waiting_approval", "running"),
        ("waiting_approval", "cancelling"),
        ("cancelling", "cancelled"),
    }
)


def transition_allowed(from_status: SubagentStatus, to_status: SubagentStatus) -> bool:
    """Design §State machine: legal edges plus terminal idempotency.

    ``cancelling`` is non-terminal: it keeps occupying a worker slot until the
    worker actually exits, so a non-interruptible sync model request can never
    free the slot early.
    """
    if from_status == to_status:
        return from_status in TERMINAL_STATUSES
    return (from_status, to_status) in _TRANSITIONS


def is_terminal_status(status: SubagentStatus) -> bool:
    return status in TERMINAL_STATUSES


# -- permission composition (design §Permission Composition) -------------------
# Baseline sets are read-only tool visibility only. ``full`` intentionally
# resolves against the parent-visible catalog at composition time.
_RESEARCH_TOOLS = frozenset(
    {
        "list_files",
        "glob_files",
        "grep_text",
        "read_file",
        "get_file_info",
        "read_session_history",
        "load_skill",
    }
)

PERMISSION_PROFILES: dict[str, frozenset[str]] = {
    "research": _RESEARCH_TOOLS,
    "verify": _RESEARCH_TOOLS | frozenset({"run_command"}),
    "edit": _RESEARCH_TOOLS | frozenset({"edit_file", "write_file", "save_memory"}),
    # ``full`` is dynamic: it resolves to the parent-visible catalog at
    # composition time (see ``effective_tool_names``), so the static entry is
    # empty. It must adapt to parent capability flags, not a hardcoded list.
    "full": frozenset(),
}
VALID_PROFILES = frozenset(PERMISSION_PROFILES)


def effective_tool_names(
    profile: str,
    parent_visible: Iterable[str],
    allow: Iterable[str] = (),
    deny: Iterable[str] = (),
) -> frozenset[str]:
    """Design §Permission Composition: profile -> ∪ allow -> − deny -> − subagent tools.

    Composition order: the profile baseline (the parent-visible set for
    ``full``) is unioned with explicit ``allow`` (only names already present in
    the parent catalog), then ``deny`` is subtracted (always effective), then
    all subagent tools are removed, and the result is capped by the
    parent-visible set so a child can never see a tool the parent run cannot
    see (hard ceiling).

    This function decides visibility only; runtime allow/ask/deny still comes
    from the parent PermissionManager on the real path.
    """
    if profile not in PERMISSION_PROFILES:
        raise ValueError(f"unknown permission profile {profile!r}")
    parent = frozenset(parent_visible)
    base = parent if profile == "full" else PERMISSION_PROFILES[profile]
    allowed = frozenset(allow) & parent
    return (base | allowed) - frozenset(deny) - SUBAGENT_TOOLS & parent
