"""Child EventSink wrapper: bounded, task-tagged tool summaries (design §Terminal).

The coordinator's ``subagent_status_changed`` events already cover lifecycle
(queued/running/waiting_approval/completed/failed/cancelling/cancelled), so the
wrapper forwards only the two bounded tool events:

- ``subagent_tool_call_started``  — tool name/id + arguments truncated to ~200 chars;
- ``subagent_tool_call_finished`` — ok flag + bounded summary/data.

Payloads are neutral: no keys, no absolute paths, no full instructions, no
reasoning and no large tool output. Per-child ordering is preserved by
construction (the wrapper is called synchronously from one child's loop); events
from different children interleave freely and are distinguished by ``task_id``.
Lifecycle events (``run_started``, ``assistant_text``, ...) are intentionally not
forwarded.
"""

from __future__ import annotations

import json

from forestcode.core.events import EventSink
from forestcode.core.types import RunEvent

# design §Terminal: bounded excerpts, never full content.
MAX_ARGUMENTS_CHARS = 200
MAX_SUMMARY_CHARS = 200
MAX_DATA_CHARS = 500

EVENT_SUBAGENT_TOOL_CALL_STARTED = "subagent_tool_call_started"
EVENT_SUBAGENT_TOOL_CALL_FINISHED = "subagent_tool_call_finished"


def _bounded_text(value, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...<truncated {len(text) - limit} chars>"


def _bounded_json(value, limit: int) -> str | None:
    if value is None:
        return None
    try:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError):
        return _bounded_text(value, limit)
    return _bounded_text(text, limit)


class SubagentEventSink:
    """Forwards bounded child tool events to the parent sink with task identity.

    Implements the ``EventSink`` protocol so it can be passed directly to a
    child ``AgentLoop``/``ToolExecutor`` as the child's event sink.
    """

    def __init__(
        self,
        sink: EventSink,
        *,
        task_id: str,
        agent_name: str,
    ) -> None:
        self._sink = sink
        self._task_id = task_id
        self._agent_name = agent_name

    def emit(self, event: RunEvent) -> None:
        if event.type == "tool_call_started":
            self._forward_started(event)
        elif event.type == "tool_call_finished":
            self._forward_finished(event)
        # Everything else is covered by the coordinator's status events; never
        # forward full reasoning, prompts or large tool content.

    def _forward_started(self, event: RunEvent) -> None:
        payload = event.payload
        self._sink.emit(
            RunEvent(
                type=EVENT_SUBAGENT_TOOL_CALL_STARTED,
                payload={
                    "task_id": self._task_id,
                    "agent_name": self._agent_name,
                    "tool_call_id": payload.get("tool_call_id"),
                    "tool_name": payload.get("tool_name"),
                    "arguments": _bounded_json(
                        payload.get("arguments"), MAX_ARGUMENTS_CHARS
                    ),
                },
            )
        )

    def _forward_finished(self, event: RunEvent) -> None:
        payload = event.payload
        self._sink.emit(
            RunEvent(
                type=EVENT_SUBAGENT_TOOL_CALL_FINISHED,
                payload={
                    "task_id": self._task_id,
                    "agent_name": self._agent_name,
                    "tool_call_id": payload.get("tool_call_id"),
                    "tool_name": payload.get("tool_name"),
                    "ok": bool(payload.get("ok", False)),
                    "summary": _bounded_text(payload.get("summary"), MAX_SUMMARY_CHARS),
                    "data": _bounded_json(payload.get("data"), MAX_DATA_CHARS),
                },
            )
        )
