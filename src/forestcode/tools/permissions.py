"""Permission decisions for tool execution."""

from __future__ import annotations

from collections.abc import Sequence

from .sandbox import PathCheck
from .types import PermissionDecision, ToolDefinition


class PermissionManager:
    def __init__(self, enable_command_tools: bool = False) -> None:
        self.enable_command_tools = enable_command_tools

    def decide(self, tool: ToolDefinition, path_checks: Sequence[PathCheck]) -> PermissionDecision:
        if tool.risk_level == "command" and not self.enable_command_tools:
            return PermissionDecision.deny(
                f"{tool.name} is disabled because command tools are not enabled."
            )

        for check in path_checks:
            if check.runtime_internal:
                return PermissionDecision.deny("Permission denied.")
            if not check.inside_workspace:
                if not tool.is_read_only:
                    return PermissionDecision.deny("Write outside workspace is not allowed.")
                return PermissionDecision.ask(f"Path is outside workspace: {check.resolved_path}")
            if check.sensitive:
                return PermissionDecision.ask(f"Sensitive path requires approval: {check.resolved_path}")

        if tool.risk_level == "state":
            if path_checks:
                return PermissionDecision.deny("State tools must not declare path access.")
            return PermissionDecision.allow()

        if not tool.is_read_only or tool.risk_level != "read_only":
            return PermissionDecision.ask(f"Tool requires elevated permission: {tool.name}")

        return PermissionDecision.allow()
