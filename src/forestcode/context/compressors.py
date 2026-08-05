"""Context compression helpers."""

from __future__ import annotations

from forestcode.core.types import ToolResult

from .budget import truncate_text


def summarize_tool_result(result: ToolResult, max_chars: int) -> str:
    if result.ok:
        content = result.summary if result.summary is not None else result.content
        prefix = f"ok:{result.tool_name}:{result.tool_call_id}:"
    else:
        content = result.error or result.summary or result.content
        prefix = f"error:{result.tool_name}:{result.tool_call_id}:"

    compacted = truncate_text(content, max_chars)
    return prefix + compacted.text
