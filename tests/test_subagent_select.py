"""Tests for the configured-subagent selector and pending marker."""

from __future__ import annotations

import io
import unittest

from forestcode.subagents import (
    AgentConfig,
    AgentConfigSet,
    PendingSubagentSelection,
)
from forestcode.terminal.subagent_select import make_numbered_subagent_selector


def _snapshot() -> AgentConfigSet:
    return AgentConfigSet(
        agents={
            "reviewer": AgentConfig(
                name="reviewer",
                description="Review code",
                instructions="Review carefully",
                permission_profile="research",
            ),
            "editor": AgentConfig(
                name="editor",
                description="Edit code",
                instructions="Make focused edits",
                permission_profile="edit",
            ),
        }
    )


class SubagentSelectorTest(unittest.TestCase):
    def test_numbered_selector_shows_description_and_permission(self) -> None:
        output = io.StringIO()
        selector = make_numbered_subagent_selector(
            input_func=lambda _prompt: "2", stdout=output
        )

        selected = selector(_snapshot())

        self.assertEqual(selected, "reviewer")
        text = output.getvalue()
        self.assertIn("editor — Edit code · permission: edit", text)
        self.assertIn("reviewer — Review code · permission: research", text)

    def test_empty_input_cancels(self) -> None:
        selector = make_numbered_subagent_selector(
            input_func=lambda _prompt: "", stdout=io.StringIO()
        )
        self.assertIsNone(selector(_snapshot()))

    def test_pending_marker_contains_only_selected_name(self) -> None:
        pending = PendingSubagentSelection()
        pending.replace("reviewer")
        self.assertEqual(pending.marker_text(), "[Subagent: reviewer]")
        pending.clear()
        self.assertIsNone(pending.marker_text())


if __name__ == "__main__":
    unittest.main()
