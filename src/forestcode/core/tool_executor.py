"""Execute tool calls and normalize tool results."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext as _nullcontext
from pathlib import Path
from typing import Any

from forestcode.tools import (
    ApprovalRequest,
    PermissionManager,
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    ToolRuntimeServices,
    WorkspaceSandbox,
    build_command_content,
)

from .abort import AbortSignal
from .run_state import RunState
from .types import ToolCall, ToolResult

ToolFunc = Callable[[dict[str, Any]], str]


class ToolExecutor:
    def __init__(
        self,
        tools: dict[str, ToolFunc] | ToolRegistry,
        workspace_root: str | Path = ".",
        permission_manager: PermissionManager | None = None,
        max_output_chars: int = 20_000,
        runtime_internal_dirs: frozenset[Path] = frozenset(),
        runtime_exception_dirs: frozenset[Path] = frozenset(),
        runtime: ToolRuntimeServices | None = None,
        enable_command_tools: bool = False,
        abort: AbortSignal | None = None,
        file_ref_threshold: int = 8_000,
        tool_results_dir: Path | None = None,
        session_id: str | None = None,
    ) -> None:
        # enable_command_tools: gate command-risk tools at execution time. IGNORED when a
        # custom permission_manager is supplied — that manager then owns command-gating policy.
        if isinstance(tools, ToolRegistry):
            self._registry = tools
        else:
            self._registry = ToolRegistry(
                [
                    ToolDefinition(
                        name=name,
                        description=f"Legacy tool: {name}",
                        input_schema={"type": "object"},
                        runner=self._wrap_legacy_tool(func),
                    )
                    for name, func in tools.items()
                ]
            )
        self._workspace_root = Path(workspace_root).resolve()
        self._permission_manager = permission_manager or PermissionManager(
            enable_command_tools=enable_command_tools
        )
        self._max_output_chars = max_output_chars
        self._runtime_internal_dirs = frozenset(runtime_internal_dirs)
        self._runtime_exception_dirs = frozenset(runtime_exception_dirs)
        self._runtime = runtime or ToolRuntimeServices()
        # Per-turn cancellation token (plan §11-B3); None => no cancellation.
        self._abort = abort
        self._file_ref_threshold = file_ref_threshold
        # Resolve the results dir the same way as workspace_root so that
        # dest.relative_to(workspace_root) works even when the two paths reach
        # the same location through different symlinks (e.g. /var -> /private/var
        # on macOS for TemporaryDirectory roots).
        self._tool_results_dir = (
            Path(tool_results_dir).resolve() if tool_results_dir is not None else None
        )
        self._session_id = session_id

    @staticmethod
    def _wrap_legacy_tool(func: ToolFunc):
        def run(_context: ToolContext, **arguments: Any) -> str:
            return func(arguments)

        return run

    def execute(self, call: ToolCall, state: RunState) -> ToolResult:
        tool = self._registry.get(call.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=False,
                content="",
                error=f"Tool not found: {call.name}",
            )
        result = self._execute_checked(call, state, tool)
        if not tool.persist_result:
            # Transient-result contract (design §Tool result persistence): mark
            # every outcome (success or failure) state_only/transient so the
            # SessionRecorder skips it, while the content stays visible for the
            # current run's later iterations.
            data = dict(result.data or {})
            data["state_only"] = True
            data["transient"] = True
            result.data = data
        return result

    def _execute_checked(
        self, call: ToolCall, state: RunState, tool: ToolDefinition
    ) -> ToolResult:
        try:
            context = ToolContext(
                workspace_root=self._workspace_root,
                max_output_chars=self._max_output_chars,
                runtime_internal_dirs=self._runtime_internal_dirs,
                runtime_exception_dirs=self._runtime_exception_dirs,
                read_state_store=self._runtime.read_state_store,
                patch_service=self._runtime.patch_service,
                plan_store=self._runtime.plan_store,
                abort=self._abort,
            )
            arguments = tool.validate_arguments(call.arguments)
            sandbox = WorkspaceSandbox(
                context.workspace_root,
                runtime_internal_dirs=self._runtime_internal_dirs,
                runtime_exception_dirs=self._runtime_exception_dirs,
            )
            path_checks = [
                sandbox.check_access(access) for access in tool.get_paths(arguments)
            ]
            hidden_result = self._hide_runtime_internal_path(
                call, path_checks, arguments, tool
            )
            if hidden_result is not None:
                return hidden_result
            decision = self._permission_manager.decide(tool, path_checks)
            if decision.behavior == "deny":
                return ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    content="",
                    error=decision.reason or "Permission denied",
                    data={"permission": "deny"},
                )
            if decision.behavior == "ask":
                gate = self._runtime.mutation_gate
                if (
                    tool.proposer is not None
                    and self._runtime.confirm is not None
                    and self._runtime.patch_service is not None
                ):
                    # Single-writer gate (design §Mutation Gate): propose, wait
                    # for approval and apply/reject are one atomic section so a
                    # diff is computed against the file state that will exist at
                    # apply time, even when other children are also editing.
                    with gate if gate is not None else _nullcontext():
                        proposal = tool.proposer(context, arguments, call.id)
                        added, removed = self._count_diff(proposal.diff)
                        approved = self._runtime.confirm(
                            ApprovalRequest(
                                kind="patch",
                                tool_name=tool.name,
                                preview=proposal.diff,
                                operation=proposal.operation,
                                path=proposal.path,
                                added=added,
                                removed=removed,
                            )
                        )
                        if context.abort is not None:
                            context.abort.raise_if_aborted()
                        if approved:
                            try:
                                self._runtime.patch_service.apply(proposal)
                            except Exception as exc:  # noqa: BLE001 - return tool error
                                return ToolResult(
                                    tool_call_id=call.id,
                                    tool_name=call.name,
                                    ok=False,
                                    content="",
                                    error=str(exc),
                                    data={"patch_id": proposal.id, "status": "failed"},
                                )
                            return ToolResult(
                                tool_call_id=call.id,
                                tool_name=call.name,
                                ok=True,
                                content=self._patch_summary(proposal),
                                summary=f"Applied patch {proposal.id}",
                                data={
                                    "patch_id": proposal.id,
                                    "diff": proposal.diff,
                                    "status": "applied",
                                },
                            )
                        self._runtime.patch_service.reject(proposal)
                        return ToolResult(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            ok=False,
                            content="",
                            error=f"User declined patch for {proposal.path}.",
                            data={"patch_id": proposal.id, "status": "rejected"},
                        )
                if (
                    tool.command_proposer is not None
                    and self._runtime.confirm is not None
                    and self._runtime.command_service is not None
                ):
                    with gate if gate is not None else _nullcontext():
                        proposal = tool.command_proposer(context, arguments, call.id)
                        approved = self._runtime.confirm(
                            ApprovalRequest(
                                kind="command",
                                tool_name=tool.name,
                                preview=proposal.display,
                                command=proposal.command,
                                cwd=str(proposal.cwd),
                                is_dangerous=proposal.is_dangerous,
                            )
                        )
                        if context.abort is not None:
                            context.abort.raise_if_aborted()
                        if approved:
                            try:
                                self._runtime.command_service.execute(
                                    proposal, abort=context.abort
                                )
                            except Exception as exc:  # noqa: BLE001 - return tool error
                                return ToolResult(
                                    tool_call_id=call.id,
                                    tool_name=call.name,
                                    ok=False,
                                    content=build_command_content(proposal),
                                    error=str(exc),
                                    data={
                                        "command_id": proposal.id,
                                        "status": proposal.status,
                                    },
                                )
                            return ToolResult(
                                tool_call_id=call.id,
                                tool_name=call.name,
                                ok=proposal.exit_code == 0,
                                content=build_command_content(proposal),
                                summary=self._command_summary(proposal),
                                data={
                                    "command_id": proposal.id,
                                    "exit_code": proposal.exit_code,
                                    "status": "executed",
                                    # B2: command output rides on `data` (already
                                    # truncated by CommandService) so the frontend
                                    # can render it (plan §5). Only commands carry an
                                    # output body; generic tools do not.
                                    "stdout": proposal.stdout,
                                    "stderr": proposal.stderr,
                                },
                            )
                        self._runtime.command_service.reject(proposal)
                        return ToolResult(
                            tool_call_id=call.id,
                            tool_name=call.name,
                            ok=False,
                            content="",
                            error="User declined command.",
                            data={"command_id": proposal.id, "status": "rejected"},
                        )
                missing_runtime_error = (
                    "Permission required"
                    if (tool.proposer is not None or tool.command_proposer is not None)
                    else decision.reason
                )
                return ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=False,
                    content="",
                    error=missing_runtime_error or "Permission required",
                    data={"permission": "ask"},
                )
            output = tool.run(context, arguments)
        except Exception as exc:  # noqa: BLE001 - tools must not break the loop
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=False,
                content="",
                error=str(exc),
            )

        # B2: generic success metric. `lines` is MERGED, never replaces
        # `state_only` — memory/recorder.py depends on `state_only` to skip
        # state tools, so overwriting it would regress recording.
        data: dict[str, Any] = {"lines": len(output.splitlines())}
        if tool.risk_level == "state":
            data["state_only"] = True
        # Transient tools (persist_result=False, e.g. load_skill) must never spill
        # their body to .forestcode/tool-results/ (PRD R7): always inline via
        # _truncate_plain so the content stays visible this run but nothing is
        # written to disk. Skill bodies are bounded (16,000 chars) below the
        # default max_output_chars, so inline never truncates.
        content = (
            self._truncate_plain(output)
            if not tool.persist_result
            else self._resolve_content(output, call.id, call.name)
        )
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=True,
            content=content,
            summary=self._summarize(output),
            data=data,
        )

    @staticmethod
    def _hide_runtime_internal_path(
        call: ToolCall,
        path_checks: list[Any],
        arguments: dict[str, Any],
        tool: ToolDefinition,
    ) -> ToolResult | None:
        if not any(check.runtime_internal for check in path_checks):
            return None

        response = tool.on_runtime_internal_path
        if response is None:
            return None

        path = str(arguments.get("path", "."))
        if response == "empty_ok":
            return ToolResult(
                tool_call_id=call.id,
                tool_name=call.name,
                ok=True,
                content="",
                summary="",
            )
        if response == "file_not_found":
            error = f"File not found: {path}"
        elif response == "path_not_found":
            error = f"Path not found: {path}"
        else:
            return None

        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            ok=False,
            content="",
            error=error,
        )

    def _truncate_plain(self, output: str) -> str:
        if len(output) <= self._max_output_chars:
            return output
        return (
            output[: self._max_output_chars]
            + f"\n...<truncated {len(output) - self._max_output_chars} chars>"
        )

    def _resolve_content(self, output: str, call_id: str, tool_name: str) -> str:
        """Route large tool output to a file reference instead of truncating inline."""
        if (
            len(output) <= self._file_ref_threshold
            or self._tool_results_dir is None
            or self._session_id is None
        ):
            return self._truncate_plain(output)

        dest_dir = self._tool_results_dir / self._session_id
        dest = dest_dir / f"{call_id}-{tool_name}.txt"

        # Verify the destination is inside the workspace before writing anything.
        try:
            rel = dest.relative_to(self._workspace_root)
        except ValueError:
            return self._truncate_plain(output)

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest.write_text(output, encoding="utf-8")

        lines = len(output.splitlines())
        chars = len(output)
        return (
            f"[output written to {rel}] ({lines} lines, {chars} chars)\n"
            f"Use read_file to view the full output."
        )

    @staticmethod
    def _summarize(output: str) -> str:
        stripped = output.strip()
        if not stripped:
            return ""
        return stripped.splitlines()[0][:200]

    @staticmethod
    def _count_diff(diff: str) -> tuple[int, int]:
        added = 0
        removed = 0
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                added += 1
            elif line.startswith("-") and not line.startswith("---"):
                removed += 1
        return added, removed

    @classmethod
    def _patch_summary(cls, proposal) -> str:
        added, removed = cls._count_diff(proposal.diff)
        return f"Applied patch {proposal.id}: {proposal.path} ({added}+/{removed}-)"

    @staticmethod
    def _command_summary(proposal) -> str:
        stdout = (proposal.stdout or "").strip()
        if stdout:
            return stdout.splitlines()[0][:200]
        if proposal.exit_code is not None:
            return f"Command exited with code {proposal.exit_code}"
        return f"Command status: {proposal.status}"
