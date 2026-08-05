"""Tool system primitives and built-in registry helpers."""

from .builtin import create_builtin_tool_registry
from .command import CommandService, build_command_content
from .mutation_gate import MutationGate
from .patch import PatchService, compute_content_hash, compute_diff
from .permissions import PermissionManager
from .read_state import ReadStateStore
from .registry import ToolRegistry
from .sandbox import WorkspaceSandbox
from .types import (
    ApprovalRequest,
    CommandProposal,
    PatchProposal,
    PathAccess,
    PermissionDecision,
    ReadFileState,
    ToolContext,
    ToolDefinition,
    ToolRuntimeServices,
)

__all__ = [
    "ApprovalRequest",
    "CommandProposal",
    "CommandService",
    "MutationGate",
    "PatchProposal",
    "PatchService",
    "PathAccess",
    "PermissionDecision",
    "PermissionManager",
    "ReadFileState",
    "ReadStateStore",
    "ToolContext",
    "ToolDefinition",
    "ToolRegistry",
    "ToolRuntimeServices",
    "WorkspaceSandbox",
    "build_command_content",
    "compute_content_hash",
    "compute_diff",
    "create_builtin_tool_registry",
]
