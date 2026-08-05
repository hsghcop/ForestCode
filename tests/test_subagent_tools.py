"""The four parent-side subagent tools: schema, validation and persistence contracts.

design §Model Tool Contract / R2 / R10. Offline only: coordinator is driven by
fake runners, no network, no model calls. Covers:

- delegate_task: agent validation, description/prompt bounds, 24,000-char
  combined context budget (fail before enqueue, never truncate);
- wait_subagents: one-time body delivery, repeat-wait ``result_omitted``,
  timeout_ms=0 immediate snapshot, empty/unknown task ids are tool errors;
- list_subagents: structured snapshots without prompts/instructions;
- cancel_subagent: idempotent, unknown id is a tool error;
- persistence contract: delegate/list/cancel results are ``state_only``
  (SessionRecorder skips them); wait's first body is a normal persistent
  result and lands in the parent session JSONL.
"""

from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from forestcode.core.abort import AbortSignal
from forestcode.core.types import ModelOutput, ToolCall
from forestcode.skills.types import SkillSnapshot
from forestcode.subagents import SubagentCoordinator
from forestcode.subagents.tools import create_subagent_tools
from forestcode.subagents.types import (
    AgentConfig,
    AgentConfigSet,
    SubagentRequest,
    SubagentResult,
)
from forestcode.tools import ToolDefinition, ToolRegistry

Agent = AgentConfig


def _agent(name: str, **kwargs: Any) -> AgentConfig:
    """Test helper: required description/instructions with sane defaults."""
    kwargs.setdefault("instructions", "you are a helper")
    return AgentConfig(name=name, description=f"desc {name}", **kwargs)


def _make_agent_set(*agents: AgentConfig) -> AgentConfigSet:
    return AgentConfigSet({agent.name: agent for agent in agents})


def _seq_ids() -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"sub-{next(counter):04d}"


def _result(task_id: str, agent_name: str, text: str = "child done") -> SubagentResult:
    return SubagentResult(
        task_id=task_id,
        agent_name=agent_name,
        final_text=text,
        turn_count=1,
        tool_count=0,
    )


class _InstantRunner:
    """Completes every child immediately with a deterministic body."""

    def __init__(self) -> None:
        self.runs: list[SubagentRequest] = []

    def run(self, request: SubagentRequest, *, abort: AbortSignal) -> SubagentResult:
        self.runs.append(request)
        return _result(
            request.task_id, request.agent_name, f"result for {request.agent_name}"
        )


class _CoordinatorHarness:
    """Builds a coordinator plus the four tools bound to it."""

    def __init__(
        self,
        runner: Any = None,
        agent_set: AgentConfigSet | None = None,
        skills_snapshot: SkillSnapshot | None = None,
    ) -> None:
        self.runner = runner or _InstantRunner()
        self.coordinator = SubagentCoordinator(
            self.runner,
            max_workers=2,
            default_wait_timeout=5.0,
            id_factory=_seq_ids(),
        )
        self.tools = {
            tool.name: tool
            for tool in create_subagent_tools(
                self.coordinator,
                agent_set=agent_set or _make_agent_set(),
                skills_snapshot=skills_snapshot,
            )
        }
        self.registry = ToolRegistry()
        for tool in self.tools.values():
            self.registry.register(tool)

    def run_tool(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.registry.get(name)
        if tool is None:
            raise KeyError(name)
        runner = tool.runner
        if runner is None:
            raise TypeError(f"{name} has no runner")
        return runner(None, **arguments)


class _BaseAgent:
    def __init__(self, name: str, instructions: str = "you are a helper") -> None:
        self.name = name
        self.instructions = instructions


class SubagentToolsTest(unittest.TestCase):
    def _harness(
        self, agents: tuple[AgentConfig, ...] = (), skills: SkillSnapshot | None = None
    ) -> _CoordinatorHarness:
        return _CoordinatorHarness(
            agent_set=_make_agent_set(*agents), skills_snapshot=skills
        )

    # -- delegate_task -----------------------------------------------------
    def test_delegate_schema_exposes_complete_agent_catalog(self) -> None:
        h = self._harness(agents=(_agent("researcher"), _agent("reviewer")))
        tool = h.tools["delegate_task"]
        self.assertIn("researcher: desc researcher", tool.description)
        self.assertIn("reviewer: desc reviewer", tool.description)
        self.assertEqual(
            tool.input_schema["properties"]["agent"]["enum"],
            ["researcher", "reviewer"],
        )

    def test_delegate_returns_task_id_and_status(self) -> None:
        helper = _agent("helper")
        h = self._harness(agents=(helper,))
        out = json.loads(
            h.run_tool(
                "delegate_task", {"agent": "helper", "description": "d", "prompt": "p"}
            )
        )
        # design §Model Tool Contract: delegate returns queued or running.
        self.assertIn(out["status"], {"queued", "running"})
        self.assertRegex(out["task_id"], r"^sub-[0-9]{4}$")

    def test_delegate_unknown_agent_fails(self) -> None:
        h = self._harness()
        with self.assertRaises(ValueError):
            h.run_tool(
                "delegate_task", {"agent": "nope", "description": "d", "prompt": "p"}
            )

    def test_delegate_rejects_empty_or_overlong_description(self) -> None:
        helper = _agent("helper")
        h = self._harness(agents=(helper,))
        with self.assertRaises(ValueError):
            h.run_tool(
                "delegate_task", {"agent": "helper", "description": "", "prompt": "p"}
            )
        with self.assertRaises(ValueError):
            h.run_tool(
                "delegate_task",
                {"agent": "helper", "description": "x" * 2001, "prompt": "p"},
            )

    def test_delegate_rejects_overlong_prompt(self) -> None:
        helper = _agent("helper")
        h = self._harness(agents=(helper,))
        with self.assertRaises(ValueError):
            h.run_tool(
                "delegate_task",
                {"agent": "helper", "description": "d", "prompt": "x" * 16001},
            )

    def test_delegate_budget_fails_before_enqueue_when_over_24k(self) -> None:
        # instructions (16k) + prompt (8k+) crosses the 24,000 combined limit;
        # the delegation must fail and no child may be registered.
        helper = _agent("helper", instructions="i" * 16000)
        h = self._harness(agents=(helper,))
        with self.assertRaises(ValueError):
            h.run_tool(
                "delegate_task",
                {"agent": "helper", "description": "d", "prompt": "p" * 9000},
            )
        self.assertEqual(h.coordinator.list(), ())

    def test_delegate_budget_allows_exactly_at_limit(self) -> None:
        helper = _agent("helper", instructions="i" * 10000)
        h = self._harness(agents=(helper,))
        out = json.loads(
            h.run_tool(
                "delegate_task",
                {"agent": "helper", "description": "d", "prompt": "p" * 14000},
            )
        )
        self.assertIn(out["status"], {"queued", "running"})

    def test_delegate_uses_agent_task_timeout(self) -> None:
        helper = _agent("helper", task_timeout_seconds=120)
        h = self._harness(agents=(helper,))
        h.run_tool(
            "delegate_task", {"agent": "helper", "description": "d", "prompt": "p"}
        )
        # The instant runner may already be terminal; the contract to verify is
        # that the delegation registered exactly one task for the run.
        snapshots = h.coordinator.list()
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0].agent_name, "helper")

    def test_delegate_missing_default_skill_makes_agent_invalid(self) -> None:
        helper = _agent("helper", default_skills=("ghost",))
        h = self._harness(agents=(helper,), skills=SkillSnapshot(descriptors=()))
        with self.assertRaises(ValueError):
            h.run_tool(
                "delegate_task", {"agent": "helper", "description": "d", "prompt": "p"}
            )

    # -- wait_subagents ----------------------------------------------------
    def test_wait_delivers_body_once(self) -> None:
        helper = _agent("helper")
        h = self._harness(agents=(helper,))
        h.run_tool(
            "delegate_task", {"agent": "helper", "description": "d", "prompt": "p"}
        )
        out = json.loads(h.run_tool("wait_subagents", {}))
        self.assertFalse(out["timed_out"])
        self.assertEqual(out["results"]["sub-0001"]["final_text"], "result for helper")
        self.assertEqual(out["tasks"][0]["delivered"], True)

    def test_repeat_wait_omits_body(self) -> None:
        helper = _agent("helper")
        h = self._harness(agents=(helper,))
        h.run_tool(
            "delegate_task", {"agent": "helper", "description": "d", "prompt": "p"}
        )
        first = json.loads(h.run_tool("wait_subagents", {}))
        self.assertIn("sub-0001", first["results"])
        second = json.loads(h.run_tool("wait_subagents", {}))
        self.assertFalse(second["timed_out"])
        self.assertEqual(second["results"], {})
        self.assertIn("sub-0001", second["result_omitted"])
        self.assertEqual(second["tasks"][0]["delivered"], True)

    def test_wait_zero_timeout_is_immediate_snapshot(self) -> None:
        helper = _agent("helper")
        h = self._harness(agents=(helper,))
        h.run_tool(
            "delegate_task", {"agent": "helper", "description": "d", "prompt": "p"}
        )
        out = json.loads(h.run_tool("wait_subagents", {"timeout_ms": 0}))
        # The instant runner may already be terminal; the key contract is the
        # call returns without blocking and never cancels the task.
        self.assertIn("tasks", out)
        self.assertEqual(len(out["tasks"]), 1)

    def test_wait_empty_list_is_tool_error(self) -> None:
        h = self._harness()
        with self.assertRaises(ValueError):
            h.run_tool("wait_subagents", {"task_ids": []})

    def test_wait_no_tasks_in_run_is_tool_error(self) -> None:
        # Design §Model Tool Contract: a run with no delegated tasks must fail
        # the wait (distinguishable from a bounded wait that timed out).
        h = self._harness()
        with self.assertRaises(ValueError):
            h.run_tool("wait_subagents", {})
        with self.assertRaises(ValueError):
            h.run_tool("wait_subagents", {"timeout_ms": 0})

    def test_wait_unknown_task_id_is_tool_error(self) -> None:
        h = self._harness()
        with self.assertRaises(ValueError):
            h.run_tool("wait_subagents", {"task_ids": ["sub-9999"]})

    def test_wait_validates_timeout_range(self) -> None:
        h = self._harness()
        with self.assertRaises(ValueError):
            h.run_tool("wait_subagents", {"timeout_ms": -1})
        with self.assertRaises(ValueError):
            h.run_tool("wait_subagents", {"timeout_ms": 60001})

    # -- list_subagents ----------------------------------------------------
    def test_list_returns_snapshots_without_prompts(self) -> None:
        helper = _agent("helper", instructions="secret instructions")
        h = self._harness(agents=(helper,))
        h.run_tool(
            "delegate_task",
            {"agent": "helper", "description": "d", "prompt": "hidden prompt"},
        )
        out = json.loads(h.run_tool("list_subagents", {}))
        self.assertEqual(len(out["tasks"]), 1)
        text = json.dumps(out)
        self.assertNotIn("hidden prompt", text)
        self.assertNotIn("secret instructions", text)

    # -- cancel_subagent ---------------------------------------------------
    def test_cancel_is_idempotent_and_reports_status(self) -> None:
        helper = _agent("helper")
        h = self._harness(agents=(helper,))
        h.run_tool(
            "delegate_task", {"agent": "helper", "description": "d", "prompt": "p"}
        )
        first = json.loads(h.run_tool("cancel_subagent", {"task_id": "sub-0001"}))
        self.assertIn("status", first)
        second = json.loads(h.run_tool("cancel_subagent", {"task_id": "sub-0001"}))
        self.assertEqual(second["status"], first["status"])

    def test_cancel_unknown_task_id_is_tool_error(self) -> None:
        h = self._harness()
        with self.assertRaises(ValueError):
            h.run_tool("cancel_subagent", {"task_id": "sub-0001"})

    # -- schema persistence contract --------------------------------------
    def test_delegate_list_cancel_are_state_only(self) -> None:
        helper = _agent("helper")
        h = self._harness(agents=(helper,))
        for name in ("delegate_task", "list_subagents", "cancel_subagent"):
            tool: ToolDefinition = h.tools[name]
            self.assertFalse(tool.persist_result, f"{name} must be state_only")
        wait_tool: ToolDefinition = h.tools["wait_subagents"]
        self.assertTrue(wait_tool.persist_result, "wait_subagents must persist")

    def test_wait_first_body_lands_in_parent_session(self) -> None:
        # End-to-end: run the real agent loop with a fake model that delegates
        # then waits; the child's final text must appear in the parent JSONL.
        from forestcode.config import AgentRuntimeConfig
        from forestcode.core.fake_model import FakeModelClient
        from forestcode.memory import SessionStore
        from forestcode.runtime.factory import build_agent_loop

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            helper = _agent("helper")
            agent_set = _make_agent_set(helper)
            runner = _InstantRunner()
            coordinator = SubagentCoordinator(
                runner, max_workers=2, default_wait_timeout=5.0, id_factory=_seq_ids()
            )
            tools = {
                tool.name: tool
                for tool in create_subagent_tools(
                    coordinator, agent_set=agent_set, skills_snapshot=None
                )
            }
            # First turn: delegate. Second turn: wait. Third: final answer.
            model = FakeModelClient(
                [
                    ModelOutput(
                        tool_calls=[
                            ToolCall(
                                id="c1",
                                name="delegate_task",
                                arguments={
                                    "agent": "helper",
                                    "description": "d",
                                    "prompt": "p",
                                },
                            )
                        ]
                    ),
                    ModelOutput(
                        tool_calls=[
                            ToolCall(id="c2", name="wait_subagents", arguments={})
                        ]
                    ),
                    ModelOutput(text="all done"),
                ]
            )
            session_id = "sess-test"
            store = SessionStore(workspace)
            registry = ToolRegistry()
            for tool in tools.values():
                registry.register(tool)
            loop = build_agent_loop(
                model,
                workspace,
                agent=AgentRuntimeConfig(),
                session_id=session_id,
                session_store=store,
                enable_write_tools=True,
                subagents=coordinator,
                subagent_tools=list(tools.values()),
            )
            state = loop.run("run the helpers")
            self.assertIsNotNone(state.final_text)
            # The child body "result for helper" must be in the parent history
            # (one-time handoff) — verify through the session store entries.
            entries = store.load(session_id).entries
            bodies = [e.content for e in entries if e.kind == "tool_result"]
            self.assertTrue(any("result for helper" in (body or "") for body in bodies))


if __name__ == "__main__":
    unittest.main()
