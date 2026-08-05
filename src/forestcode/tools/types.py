"""Shared types for tool definitions, permissions, and execution context."""

from __future__ import annotations

import types
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from forestcode.plan.types import PlanStoreProtocol

from ._path import canonical

if TYPE_CHECKING:
    # core.abort is a zero-dependency leaf module; importing it only for typing
    # avoids a tools<->core import cycle at runtime (plan §11-B3 decoupling).
    from forestcode.core.abort import AbortSignal


PathIntent = Literal["read", "list", "search", "stat", "write", "execute"]
RiskLevel = Literal["read_only", "write", "command", "external", "state"]
PermissionBehavior = Literal["allow", "ask", "deny"]
RuntimeInternalResponse = Literal["empty_ok", "file_not_found", "path_not_found"]


@dataclass(frozen=True, slots=True)
class PathAccess:
    path: str
    intent: PathIntent = "read"


@dataclass(frozen=True, slots=True)
class PermissionDecision:
    behavior: PermissionBehavior
    reason: str | None = None

    @classmethod
    def allow(cls) -> PermissionDecision:
        return cls("allow")

    @classmethod
    def ask(cls, reason: str) -> PermissionDecision:
        return cls("ask", reason)

    @classmethod
    def deny(cls, reason: str) -> PermissionDecision:
        return cls("deny", reason)


@dataclass(slots=True)
class ReadFileState:
    path: str
    content_hash: str
    mtime: float
    is_partial: bool
    offset: int | None
    limit: int | None


@dataclass(slots=True)
class PatchProposal:
    id: str
    path: str
    resolved_path: Path
    operation: Literal["edit", "write", "create"]
    base_hash: str | None
    diff: str
    updated_content: str
    status: Literal["proposed", "applied", "rejected", "failed"]
    tool_call_id: str
    created_at: str
    applied_at: str | None = None
    error: str | None = None


@dataclass(slots=True)
class CommandProposal:
    id: str
    command: str
    cwd: Path
    timeout: int
    shell_label: str
    is_dangerous: bool
    display: str
    status: Literal["proposed", "executed", "rejected", "failed", "timeout"]
    tool_call_id: str
    created_at: str
    stdout: str | None = None
    stderr: str | None = None
    exit_code: int | None = None
    executed_at: str | None = None
    error: str | None = None


class ReadStateStoreProtocol(Protocol):
    def record(
        self,
        path: Path,
        content: str,
        mtime: float,
        is_partial: bool,
        offset: int | None = None,
        limit: int | None = None,
    ) -> None: ...

    def get(self, path: Path) -> ReadFileState | None: ...

    def clear(self, path: Path) -> None: ...


class PatchServiceProtocol(Protocol):
    def propose(
        self,
        *,
        path: str,
        resolved_path: Path,
        operation: Literal["edit", "write", "create"],
        base_hash: str | None,
        diff: str,
        updated_content: str,
        tool_call_id: str,
    ) -> PatchProposal: ...

    def apply(self, proposal: PatchProposal) -> None: ...

    def reject(self, proposal: PatchProposal) -> None: ...


class CommandServiceProtocol(Protocol):
    def execute(
        self, proposal: CommandProposal, *, abort: AbortSignal | None = None
    ) -> None: ...

    def reject(self, proposal: CommandProposal) -> None: ...


class MutationGateProtocol(Protocol):
    """Serializes mutation sections across threads (design §Mutation Gate).

    Acquired only around the full patch/save-memory/command section
    (propose -> approval -> apply/execute) so a diff is computed from the file
    state visible at approval/apply time. Reads and other state tools never
    touch it. Implementations must be thread-safe context managers.
    """

    def __enter__(self) -> None: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None: ...


@dataclass(slots=True)
class ToolContext:
    workspace_root: Path
    max_output_chars: int = 20_000
    max_read_bytes: int = 200_000
    runtime_internal_dirs: frozenset[Path] = field(default_factory=frozenset)
    runtime_exception_dirs: frozenset[Path] = field(default_factory=frozenset)
    read_state_store: ReadStateStoreProtocol | None = None
    patch_service: PatchServiceProtocol | None = None
    plan_store: PlanStoreProtocol | None = None
    abort: AbortSignal | None = None

    def __post_init__(self) -> None:
        self.workspace_root = canonical(self.workspace_root)
        self.runtime_internal_dirs = frozenset(
            canonical(path) for path in self.runtime_internal_dirs
        )
        self.runtime_exception_dirs = frozenset(
            canonical(path) for path in self.runtime_exception_dirs
        )


Validator = Callable[[dict[str, Any]], dict[str, Any]]
PathGetter = Callable[[dict[str, Any]], list[PathAccess]]
ToolRunner = Callable[..., str]
Proposer = Callable[[ToolContext, dict[str, Any], str], PatchProposal]
CommandProposer = Callable[[ToolContext, dict[str, Any], str], CommandProposal]


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """Structured approval request for a patch or command (plan §11-B4).

    Carries only neutral facts. Allowlist granularity ("always allow <resource>")
    is a frontend policy: the frontend derives its key from these facts via
    ``allow_key`` (plan §6.2). The backend has no allowlist concept and must not
    learn one — keep this dataclass policy-free.
    """

    kind: Literal["patch", "command"]
    tool_name: str
    preview: str  # fallback text: the diff (patch) or command display (command)
    # —— neutral fact fields ——
    operation: str | None = None  # edit/write/create (patch)
    path: str | None = None  # patch target path
    added: int | None = None  # patch line stats
    removed: int | None = None
    command: str | None = None  # raw command (command)
    cwd: str | None = None
    is_dangerous: bool = False


@dataclass(slots=True)
class ToolRuntimeServices:
    read_state_store: ReadStateStoreProtocol | None = None
    patch_service: PatchServiceProtocol | None = None
    command_service: CommandServiceProtocol | None = None
    plan_store: PlanStoreProtocol | None = None
    confirm: Callable[[ApprovalRequest], bool] | None = None
    # Single-writer gate shared by a parent run and its children (design §Mutation
    # Gate): one mutation section at a time across all concurrent agents.
    mutation_gate: MutationGateProtocol | None = None


@dataclass(slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    runner: ToolRunner
    risk_level: RiskLevel = "read_only"
    is_read_only: bool = True
    validator: Validator | None = None
    path_getter: PathGetter | None = None
    proposer: Proposer | None = None
    command_proposer: CommandProposer | None = None
    on_runtime_internal_path: RuntimeInternalResponse | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Transient-result contract (design §Tool result persistence): when False the
    # ToolExecutor marks every result ``state_only``/``transient`` so the
    # SessionRecorder skips it. The content still enters the current run's
    # ``RunState.messages`` for later iterations. Default stays True: normal
    # tools keep recording.
    persist_result: bool = True

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.validator is None:
            return dict(arguments)
        return self.validator(arguments)

    def get_paths(self, arguments: dict[str, Any]) -> list[PathAccess]:
        if self.path_getter is None:
            return []
        return self.path_getter(arguments)

    def run(self, context: ToolContext, arguments: dict[str, Any]) -> str:
        return self.runner(context, **arguments)
