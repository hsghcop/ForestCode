"""Patch-first file editing tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .patch import compute_content_hash, compute_diff
from .sandbox import WorkspaceSandbox
from .types import PathAccess, PatchProposal, ToolContext, ToolDefinition


_BLOCKED_DIRS = {".git", ".vscode", ".idea"}
_BLOCKED_NAMES = {"id_rsa", "id_ed25519"}


def _string_arg(arguments: dict[str, Any], name: str, *, allow_empty: bool = False) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _validate_safe_write_path(path: str) -> None:
    normalized = path.replace("\\", "/")
    if normalized.startswith("//"):
        raise ValueError("UNC paths are not allowed for write tools")
    parts = [part.lower() for part in normalized.split("/") if part]
    if any(part in _BLOCKED_DIRS for part in parts):
        raise ValueError("Writing to protected editor or VCS paths is not allowed")
    name = parts[-1] if parts else ""
    if name in _BLOCKED_NAMES or name == ".env" or name.startswith(".env.") or name.endswith(".env"):
        raise ValueError("Writing to sensitive files is not allowed")


def _validate_edit_file(arguments: dict[str, Any]) -> dict[str, Any]:
    path = _string_arg(arguments, "path")
    old_text = _string_arg(arguments, "old_text")
    new_text = _string_arg(arguments, "new_text", allow_empty=True)
    _validate_safe_write_path(path)
    return {"path": path, "old_text": old_text, "new_text": new_text}


def _validate_write_file(arguments: dict[str, Any]) -> dict[str, Any]:
    path = _string_arg(arguments, "path")
    content = _string_arg(arguments, "content", allow_empty=True)
    _validate_safe_write_path(path)
    return {"path": path, "content": content}


def _write_path_getter(arguments: dict[str, Any]) -> list[PathAccess]:
    return [PathAccess(arguments["path"], "write")]


def _propose_edit_file(context: ToolContext, arguments: dict[str, Any], tool_call_id: str) -> PatchProposal:
    path = arguments["path"]
    old_text = arguments["old_text"]
    new_text = arguments["new_text"]
    target = _resolve_existing_text_file(context, path)
    current = _current_text_matching_read_state(context, target, path)
    count = current.count(old_text)
    if count == 0:
        raise ValueError("old_text was not found")
    if count > 1:
        raise ValueError("old_text must match exactly once")
    updated = current.replace(old_text, new_text, 1)
    display_path = _display_path(context, target)
    diff = compute_diff(current, updated, display_path)
    if context.patch_service is None:
        raise ValueError("PatchService is required for edit_file")
    return context.patch_service.propose(
        path=display_path,
        resolved_path=target,
        operation="edit",
        base_hash=compute_content_hash(current),
        diff=diff,
        updated_content=updated,
        tool_call_id=tool_call_id,
    )


def _propose_write_file(context: ToolContext, arguments: dict[str, Any], tool_call_id: str) -> PatchProposal:
    path = arguments["path"]
    content = arguments["content"]
    sandbox = WorkspaceSandbox(
        context.workspace_root,
        runtime_internal_dirs=context.runtime_internal_dirs,
    )
    target = sandbox.resolve_path(path)
    display_path = _display_path(context, target)
    if context.patch_service is None:
        raise ValueError("PatchService is required for write_file")

    if not target.exists():
        return context.patch_service.propose(
            path=display_path,
            resolved_path=target,
            operation="create",
            base_hash=None,
            diff=_content_preview(content),
            updated_content=content,
            tool_call_id=tool_call_id,
        )

    if not target.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    current = _current_text_matching_read_state(context, target, path)
    return context.patch_service.propose(
        path=display_path,
        resolved_path=target,
        operation="write",
        base_hash=compute_content_hash(current),
        diff=compute_diff(current, content, display_path),
        updated_content=content,
        tool_call_id=tool_call_id,
    )


def _resolve_existing_text_file(context: ToolContext, path: str) -> Path:
    sandbox = WorkspaceSandbox(
        context.workspace_root,
        runtime_internal_dirs=context.runtime_internal_dirs,
    )
    target = sandbox.resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not target.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    return target


def _current_text_matching_read_state(context: ToolContext, target: Path, path: str) -> str:
    if context.read_state_store is None:
        raise ValueError("ReadStateStore is required")
    state = context.read_state_store.get(target)
    if state is None:
        raise ValueError(f"Must read file before editing: {path}")
    if state.is_partial:
        raise ValueError(f"Must read the full file before editing: {path}")
    current = target.read_text(encoding="utf-8", errors="replace")
    current_hash = compute_content_hash(current)
    if current_hash != state.content_hash:
        raise ValueError(f"File changed since read: {path}")
    return current


def _display_path(context: ToolContext, target: Path) -> str:
    try:
        return target.relative_to(context.workspace_root).as_posix()
    except ValueError:
        return str(target)


def _content_preview(content: str) -> str:
    preview = content[:800]
    if len(content) > 800:
        preview += f"\n...<truncated {len(content) - 800} chars>"
    return f"(new file content preview)\n{preview}"


def _write_runner(_context: ToolContext, **_arguments: Any) -> str:
    raise RuntimeError("Write tools must be executed through proposer/confirm/apply")


def create_edit_file_tool() -> ToolDefinition:
    return ToolDefinition(
        name="edit_file",
        description="Propose a single exact text replacement in an existing workspace file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
        runner=_write_runner,
        risk_level="write",
        is_read_only=False,
        validator=_validate_edit_file,
        path_getter=_write_path_getter,
        proposer=_propose_edit_file,
    )


def create_write_file_tool() -> ToolDefinition:
    return ToolDefinition(
        name="write_file",
        description="Propose creating or fully rewriting a workspace file.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        runner=_write_runner,
        risk_level="write",
        is_read_only=False,
        validator=_validate_write_file,
        path_getter=_write_path_getter,
        proposer=_propose_write_file,
    )
