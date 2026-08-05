"""Subagent terminal observability (design §Terminal and Observability / R11).

Verifies the renderer's handling of the three subagent event types and that
plain/fallback output stays readable and neutral (no keys, no absolute paths,
no full instructions). Offline only.
"""

from __future__ import annotations

import io
import unittest

from forestcode.core.types import RunEvent
from forestcode.subagents.coordinator import EVENT_SUBAGENT_STATUS_CHANGED
from forestcode.subagents.events import (
    EVENT_SUBAGENT_TOOL_CALL_FINISHED,
    EVENT_SUBAGENT_TOOL_CALL_STARTED,
)
from forestcode.terminal.renderer import PlainRenderer


class SubagentRendererTest(unittest.TestCase):
    def _renderer(self) -> tuple[PlainRenderer, io.StringIO, io.StringIO]:
        out = io.StringIO()
        err = io.StringIO()
        return PlainRenderer(out, err), out, err

    def test_status_event_renders_task_and_agent(self) -> None:
        renderer, out, _err = self._renderer()
        renderer.on_event(
            RunEvent(
                EVENT_SUBAGENT_STATUS_CHANGED,
                {
                    "task_id": "sub-0001",
                    "agent_name": "helper",
                    "status": "running",
                },
            )
        )
        text = out.getvalue()
        self.assertIn("sub-0001", text)
        self.assertIn("helper", text)
        self.assertIn("started", text)

    def test_tool_events_render_bounded_and_tagged(self) -> None:
        renderer, out, _err = self._renderer()
        renderer.on_event(
            RunEvent(
                EVENT_SUBAGENT_TOOL_CALL_STARTED,
                {
                    "task_id": "sub-0001",
                    "agent_name": "helper",
                    "tool_name": "grep_text",
                },
            )
        )
        renderer.on_event(
            RunEvent(
                EVENT_SUBAGENT_TOOL_CALL_FINISHED,
                {
                    "task_id": "sub-0001",
                    "agent_name": "helper",
                    "tool_name": "grep_text",
                    "ok": True,
                },
            )
        )
        text = out.getvalue()
        self.assertIn("sub-0001", text)
        self.assertIn("grep_text", text)
        self.assertIn("ok", text)

    def test_interleaved_children_are_distinguishable(self) -> None:
        renderer, out, _err = self._renderer()
        # Two children interleave; every line carries its task id so the user
        # can tell them apart (design: per-child order, cross-child interleave).
        for task_id, agent in (("sub-0001", "a"), ("sub-0002", "b")):
            renderer.on_event(
                RunEvent(
                    EVENT_SUBAGENT_STATUS_CHANGED,
                    {"task_id": task_id, "agent_name": agent, "status": "running"},
                )
            )
        renderer.on_event(
            RunEvent(
                EVENT_SUBAGENT_TOOL_CALL_STARTED,
                {"task_id": "sub-0001", "agent_name": "a", "tool_name": "read_file"},
            )
        )
        renderer.on_event(
            RunEvent(
                EVENT_SUBAGENT_TOOL_CALL_STARTED,
                {"task_id": "sub-0002", "agent_name": "b", "tool_name": "grep_text"},
            )
        )
        text = out.getvalue()
        self.assertIn("sub-0001", text)
        self.assertIn("sub-0002", text)
        # Neutral payload: no absolute paths, no key material, no instructions.
        self.assertNotIn("/Users", text)
        self.assertNotIn("api_key", text)
        self.assertNotIn("instructions", text)

    def test_plain_events_do_not_interfere_with_tool_events(self) -> None:
        renderer, out, _err = self._renderer()
        renderer.on_event(
            RunEvent(
                "tool_call_started", {"tool_name": "read_file", "tool_call_id": "c1"}
            )
        )
        renderer.on_event(RunEvent("assistant_text_received", {"text": "hi"}))
        text = out.getvalue()
        self.assertIn("Tool> read_file", text)
        self.assertIn("Assistant> hi", text)

    def test_skill_activation_event_is_visible(self) -> None:
        renderer, out, _err = self._renderer()

        renderer.on_event(RunEvent("skill_activated", {"name": "python-review"}))

        self.assertEqual(out.getvalue(), "Skill> 已加载 python-review\n")


if __name__ == "__main__":
    unittest.main()
