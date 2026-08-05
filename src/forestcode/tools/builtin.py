"""Built-in read-only workspace tools."""

from __future__ import annotations

import fnmatch
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forestcode.memory.session_store import SessionStore

from .command import create_run_command_tool
from .editing import create_edit_file_tool, create_write_file_tool
from .patch import compute_content_hash, compute_diff
from .registry import ToolRegistry
from .todos import create_write_todos_tool
from .sandbox import WorkspaceSandbox
from .types import PathAccess, ToolContext, ToolDefinition


def _string_arg(arguments: dict[str, Any], name: str, default: str | None = None) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _int_arg(
    arguments: dict[str, Any],
    name: str,
    default: int,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    value = arguments.get(name, default)
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...<truncated {len(text) - max_chars} chars>"


def _looks_binary(path: Path, max_probe: int = 4096) -> bool:
    try:
        with path.open("rb") as file:
            return b"\x00" in file.read(max_probe)
    except OSError:
        return True


def _iter_files(root: Path, sandbox: WorkspaceSandbox):
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            children = sorted(current.iterdir(), key=lambda path: path.name.lower())
        except OSError:
            continue
        for child in children:
            if child.is_dir():
                if sandbox.should_skip_dir(child) or sandbox.is_runtime_internal_path(child):
                    continue
                stack.append(child)
                continue
            if not sandbox.is_inside_workspace(child) or sandbox.is_sensitive_path(child):
                continue
            yield child


def _validate_list_files(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _string_arg(arguments, "path", "."),
        "max_entries": _int_arg(arguments, "max_entries", 200, minimum=1, maximum=1000),
    }


def _run_list_files(context: ToolContext, path: str, max_entries: int) -> str:
    sandbox = WorkspaceSandbox(
        context.workspace_root,
        runtime_internal_dirs=context.runtime_internal_dirs,
        runtime_exception_dirs=context.runtime_exception_dirs,
    )
    target = sandbox.resolve_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if sandbox.is_runtime_internal_path(target):
        raise FileNotFoundError(f"Path not found: {path}")
    if target.is_file():
        return f"file {target.relative_to(context.workspace_root).as_posix()}"

    entries: list[str] = []
    for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if sandbox.is_runtime_internal_path(child):
            continue
        if child.is_dir() and sandbox.should_skip_dir(child):
            continue
        if sandbox.is_sensitive_path(child):
            continue
        kind = "dir " if child.is_dir() else "file"
        entries.append(f"{kind} {child.relative_to(context.workspace_root).as_posix()}")
        if len(entries) >= max_entries:
            entries.append("...<truncated entries>")
            break
    return "\n".join(entries)


def _validate_glob_files(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "pattern": _string_arg(arguments, "pattern"),
        "path": _string_arg(arguments, "path", "."),
        "max_results": _int_arg(arguments, "max_results", 100, minimum=1, maximum=1000),
    }


def _run_glob_files(context: ToolContext, pattern: str, path: str, max_results: int) -> str:
    sandbox = WorkspaceSandbox(
        context.workspace_root,
        runtime_internal_dirs=context.runtime_internal_dirs,
        runtime_exception_dirs=context.runtime_exception_dirs,
    )
    root = sandbox.resolve_path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if sandbox.is_runtime_internal_path(root):
        return ""

    candidates = [root] if root.is_file() else _iter_files(root, sandbox)
    matches: list[str] = []
    for candidate in candidates:
        rel = candidate.relative_to(context.workspace_root).as_posix()
        if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(candidate.name, pattern):
            matches.append(rel)
            if len(matches) >= max_results:
                matches.append("...<truncated results>")
                break
    return "\n".join(matches)


def _validate_grep_text(arguments: dict[str, Any]) -> dict[str, Any]:
    include = arguments.get("include")
    if include is not None and not isinstance(include, str):
        raise ValueError("include must be a string when provided")
    return {
        "pattern": _string_arg(arguments, "pattern"),
        "path": _string_arg(arguments, "path", "."),
        "include": include,
        "max_results": _int_arg(arguments, "max_results", 100, minimum=1, maximum=500),
    }


def _run_grep_text(
    context: ToolContext,
    pattern: str,
    path: str,
    include: str | None,
    max_results: int,
) -> str:
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern: {exc}") from exc

    sandbox = WorkspaceSandbox(
        context.workspace_root,
        runtime_internal_dirs=context.runtime_internal_dirs,
        runtime_exception_dirs=context.runtime_exception_dirs,
    )
    root = sandbox.resolve_path(path)
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if sandbox.is_runtime_internal_path(root):
        return ""

    candidates = [root] if root.is_file() else _iter_files(root, sandbox)
    results: list[str] = []
    for candidate in candidates:
        if not candidate.is_file():
            continue
        rel = candidate.relative_to(context.workspace_root).as_posix()
        if include and not fnmatch.fnmatch(rel, include) and not fnmatch.fnmatch(candidate.name, include):
            continue
        if _looks_binary(candidate):
            continue
        try:
            lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            if regex.search(line):
                results.append(f"{rel}:{line_no}:{line}")
                if len(results) >= max_results:
                    results.append("...<truncated results>")
                    return "\n".join(results)
    return "\n".join(results)


def _validate_read_file(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _string_arg(arguments, "path"),
        "offset": _int_arg(arguments, "offset", 0, minimum=0),
        "limit": _int_arg(arguments, "limit", 20000, minimum=1, maximum=200000),
    }


def _run_read_file(context: ToolContext, path: str, offset: int, limit: int) -> str:
    sandbox = WorkspaceSandbox(
        context.workspace_root,
        runtime_internal_dirs=context.runtime_internal_dirs,
        runtime_exception_dirs=context.runtime_exception_dirs,
    )
    target = sandbox.resolve_path(path)
    if sandbox.is_runtime_internal_path(target):
        raise FileNotFoundError(f"File not found: {path}")
    if not target.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not target.is_file():
        raise IsADirectoryError(f"Path is not a file: {path}")
    stat = target.stat()
    if stat.st_size > context.max_read_bytes:
        raise ValueError(f"File is too large: {path}")
    if _looks_binary(target):
        raise ValueError(f"File appears to be binary: {path}")

    text = target.read_text(encoding="utf-8", errors="replace")
    chunk = text[offset : offset + limit]
    rel = target.relative_to(context.workspace_root).as_posix()
    header = f"FILE {rel} OFFSET {offset} END {offset + len(chunk)} TOTAL_CHARS {len(text)}"
    if context.read_state_store is not None:
        is_partial = (
            offset > 0
            or (offset + limit) < len(text)
            or len(header + "\n" + chunk) > context.max_output_chars
        )
        context.read_state_store.record(
            path=target,
            content=text,
            mtime=stat.st_mtime,
            is_partial=is_partial,
            offset=offset,
            limit=limit,
        )
    return _truncate(header + "\n" + chunk, context.max_output_chars)


def _validate_get_file_info(arguments: dict[str, Any]) -> dict[str, Any]:
    return {"path": _string_arg(arguments, "path")}


def _run_get_file_info(context: ToolContext, path: str) -> str:
    sandbox = WorkspaceSandbox(
        context.workspace_root,
        runtime_internal_dirs=context.runtime_internal_dirs,
        runtime_exception_dirs=context.runtime_exception_dirs,
    )
    target = sandbox.resolve_path(path)
    if sandbox.is_runtime_internal_path(target):
        raise FileNotFoundError(f"Path not found: {path}")
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    stat = target.stat()
    rel = target.relative_to(context.workspace_root).as_posix()
    kind = "directory" if target.is_dir() else "file"
    return "\n".join(
        [
            f"path: {rel}",
            f"type: {kind}",
            f"size_bytes: {stat.st_size}",
            f"modified_time: {stat.st_mtime}",
        ]
    )


def _validate_read_session_history(arguments: dict[str, Any]) -> dict[str, Any]:
    allowed = {"offset", "limit"}
    extra = set(arguments) - allowed
    if extra:
        raise ValueError(f"unsupported arguments: {', '.join(sorted(extra))}")
    return {
        "offset": _int_arg(arguments, "offset", 0, minimum=0),
        "limit": _int_arg(arguments, "limit", 20, minimum=1, maximum=200),
    }


def _make_run_read_session_history(store: SessionStore, session_id: str):
    def run(context: ToolContext, offset: int, limit: int) -> str:
        memory = store.load(session_id)
        total = len(memory.entries)
        entries = memory.entries[offset : offset + limit]
        budget = int(context.max_output_chars * 0.85)
        return _fit_session_history_to_budget(entries, total, offset, limit, memory.session_id, budget)

    return run


def _fit_session_history_to_budget(
    entries,
    total: int,
    offset: int,
    limit: int,
    session_id: str,
    budget: int,
) -> str:
    for per_entry_cap in [4000, 2000, 800, 200]:
        payload = _build_session_history_payload(entries, total, offset, limit, session_id, per_entry_cap)
        text = json.dumps(payload, ensure_ascii=False)
        if len(text) <= budget:
            return text

    fallback = {
        "session_id": session_id,
        "offset": offset,
        "limit": limit,
        "total_entries": total,
        "has_more": offset + limit < total,
        "warning": "entries omitted: payload too large even at minimum truncation",
        "entries": [],
    }
    return json.dumps(fallback, ensure_ascii=False)


def _build_session_history_payload(
    entries,
    total: int,
    offset: int,
    limit: int,
    session_id: str,
    per_entry_cap: int,
) -> dict[str, Any]:
    serialized = []
    for entry in entries:
        record = asdict(entry)
        content = record.get("content", "") or ""
        if len(content) > per_entry_cap:
            record["content"] = content[:per_entry_cap] + f"...<{len(content) - per_entry_cap} chars>"
            record["content_truncated"] = True
        serialized.append(record)

    return {
        "session_id": session_id,
        "offset": offset,
        "limit": limit,
        "total_entries": total,
        "has_more": offset + limit < total,
        "entry_content_chars": per_entry_cap,
        "entries": serialized,
    }


_SAVE_MEMORY_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9\-]*$")
_SAVE_MEMORY_TYPES = {"user", "feedback", "project", "reference"}


def _validate_save_memory(arguments: dict[str, Any]) -> dict[str, Any]:
    name = arguments.get("name")
    if not isinstance(name, str) or not _SAVE_MEMORY_NAME_RE.match(name) or len(name) > 64:
        raise ValueError("name must be a kebab-case slug up to 64 characters")

    memory_type = arguments.get("memory_type")
    if memory_type not in _SAVE_MEMORY_TYPES:
        raise ValueError("memory_type must be one of user, feedback, project, reference")

    content = arguments.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")
    if len(content) > 4000:
        raise ValueError("content must be <= 4000 characters")
    if any(line.startswith("## ") for line in content.splitlines()):
        raise ValueError("content must not contain markdown section headers")

    return {"name": name, "memory_type": memory_type, "content": content}


def _propose_save_memory(context: ToolContext, arguments: dict[str, Any], tool_call_id: str):
    from forestcode.memory.markdown_memory import render_upserted_content

    sandbox = WorkspaceSandbox(
        context.workspace_root,
        runtime_internal_dirs=context.runtime_internal_dirs,
        runtime_exception_dirs=context.runtime_exception_dirs,
    )
    target = sandbox.resolve_path("MEMORY.md")
    exists = target.exists()
    old_content = target.read_text(encoding="utf-8", errors="replace") if exists else ""
    new_content = render_upserted_content(
        old_content,
        arguments["name"],
        arguments["memory_type"],
        arguments["content"],
    )
    diff = compute_diff(old_content, new_content, "MEMORY.md")
    if context.patch_service is None:
        raise ValueError("PatchService is required for save_memory")
    return context.patch_service.propose(
        path="MEMORY.md",
        resolved_path=target,
        operation="edit" if exists else "create",
        base_hash=compute_content_hash(old_content) if exists else None,
        diff=diff,
        updated_content=new_content,
        tool_call_id=tool_call_id,
    )


def _save_memory_runner(_context: ToolContext, **_arguments: Any) -> str:
    raise RuntimeError("save_memory must be executed through proposer/confirm/apply")


def create_builtin_tool_registry(
    session_store: SessionStore | None = None,
    session_id: str | None = None,
    enable_memory_write: bool = True,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="list_files",
            description="List files and directories under a workspace path.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "default": "."},
                    "max_entries": {"type": "integer", "default": 200},
                },
            },
            validator=_validate_list_files,
            path_getter=lambda args: [PathAccess(args["path"], "list")],
            on_runtime_internal_path="path_not_found",
            runner=_run_list_files,
        )
    )
    registry.register(
        ToolDefinition(
            name="glob_files",
            description="Find files by path or filename glob pattern.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "max_results": {"type": "integer", "default": 100},
                },
                "required": ["pattern"],
            },
            validator=_validate_glob_files,
            path_getter=lambda args: [PathAccess(args["path"], "search")],
            on_runtime_internal_path="empty_ok",
            runner=_run_glob_files,
        )
    )
    registry.register(
        ToolDefinition(
            name="grep_text",
            description="Search workspace files using a regex pattern.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "default": "."},
                    "include": {"type": "string"},
                    "max_results": {"type": "integer", "default": 100},
                },
                "required": ["pattern"],
            },
            validator=_validate_grep_text,
            path_getter=lambda args: [PathAccess(args["path"], "search")],
            on_runtime_internal_path="empty_ok",
            runner=_run_grep_text,
        )
    )
    registry.register(
        ToolDefinition(
            name="read_file",
            description="Read a UTF-8 text file from the workspace.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer", "default": 0},
                    "limit": {"type": "integer", "default": 20000},
                },
                "required": ["path"],
            },
            validator=_validate_read_file,
            path_getter=lambda args: [PathAccess(args["path"], "read")],
            on_runtime_internal_path="file_not_found",
            runner=_run_read_file,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_file_info",
            description="Return basic metadata for a workspace file or directory.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            validator=_validate_get_file_info,
            path_getter=lambda args: [PathAccess(args["path"], "stat")],
            on_runtime_internal_path="path_not_found",
            runner=_run_get_file_info,
        )
    )
    registry.register(create_edit_file_tool())
    registry.register(create_write_file_tool())
    if enable_memory_write:
        registry.register(
            ToolDefinition(
                name="save_memory",
                description="Save or update a named memory entry in MEMORY.md.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "memory_type": {
                            "type": "string",
                            "enum": ["user", "feedback", "project", "reference"],
                        },
                        "content": {"type": "string"},
                    },
                    "required": ["name", "memory_type", "content"],
                },
                runner=_save_memory_runner,
                risk_level="write",
                is_read_only=False,
                validator=_validate_save_memory,
                path_getter=lambda _args: [PathAccess("MEMORY.md", "write")],
                proposer=_propose_save_memory,
            )
        )
    registry.register(create_run_command_tool())
    registry.register(create_write_todos_tool())
    if session_store is not None and session_id is not None:
        registry.register(
            ToolDefinition(
                name="read_session_history",
                description="Read paginated history entries from the current ForestCode session.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "offset": {"type": "integer", "default": 0},
                        "limit": {"type": "integer", "default": 20},
                    },
                },
                validator=_validate_read_session_history,
                runner=_make_run_read_session_history(session_store, session_id),
            )
        )
    return registry
