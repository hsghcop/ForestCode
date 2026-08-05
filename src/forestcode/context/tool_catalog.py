"""Context-facing catalog for model-visible tools."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from forestcode.tools import ToolDefinition, ToolRegistry

if TYPE_CHECKING:
    from forestcode.subagents.types import PermissionProfile


class ToolVisibilityPolicy:
    """Filters a child's catalog by permission profile + allow/deny overlay (R6).

    Composition (design §Permission Composition), applied to the *parent-visible*
    catalog as the hard ceiling:

        parent-visible set
          -> profile baseline (``full`` = the parent set itself)
          -> union explicit tools.allow (only names already parent-visible)
          -> subtract explicit tools.deny (always effective)
          -> remove all subagent delegation tools

    This policy only decides whether a child *sees* a tool. Runtime allow/ask/
    deny still comes from the parent's PermissionManager against the real path,
    tool risk and capability flags — a child can never relax the parent's
    behavior. ``parent_visible`` must be the names the parent run would expose
    (its ToolCatalog output), so the filter cannot introduce a tool the parent
    does not have.
    """

    def __init__(
        self,
        parent_visible: Iterable[str],
        profile: PermissionProfile = "research",
        allow: Iterable[str] = (),
        deny: Iterable[str] = (),
    ) -> None:
        from forestcode.subagents.types import effective_tool_names

        self._visible = effective_tool_names(profile, parent_visible, allow, deny)

    @property
    def visible_names(self) -> frozenset[str]:
        return self._visible

    def filter(self, tools: Iterable[ToolDefinition]) -> list[ToolDefinition]:
        return [tool for tool in tools if tool.name in self._visible]


class ToolCatalog:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
        read_only_only: bool = True,
        enable_command_tools: bool = False,
        visibility: ToolVisibilityPolicy | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.read_only_only = read_only_only
        self.enable_command_tools = enable_command_tools
        # Optional child visibility overlay (subagent permission profiles). When
        # set, ``list_visible_tools`` returns only tools the policy allows.
        self.visibility = visibility

    def list_visible_tools(self) -> list[ToolDefinition]:
        result: list[ToolDefinition] = []
        for tool in self.registry.list_tools():
            if tool.risk_level == "command":
                # Command tools never fall through to the write-tool bucket: an
                # explicit capability flag gates them, so a disabled command
                # must not reappear just because write tools are enabled.
                if self.enable_command_tools:
                    result.append(tool)
                continue
            if (
                tool.risk_level == "state"
                or (tool.is_read_only and tool.risk_level == "read_only")
                or not self.read_only_only
            ):
                result.append(tool)
        if self.visibility is not None:
            return self.visibility.filter(result)
        return result

    def list_model_schemas(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in self.list_visible_tools()
        ]

    def selected_tool_names(self) -> list[str]:
        return [tool.name for tool in self.list_visible_tools()]
