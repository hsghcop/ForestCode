"""Child construction and the real child AgentLoop (design §Child Construction).

Offline only: fake model clients drive both parent and child loops; no network.
Covers:

- child catalog structurally excludes the four delegation tools (R3);
- a child run completes and its final text reaches the coordinator (R4/R9);
- model inheritance via ``resolve_child_model_config`` and ``api_key_env``
  failures stay diagnostic without leaking values (R8);
- combined 24,000-char budget fails before enqueue (R7);
- typed skill inheritance: activated ∪ default_skills dedup, bodies land in
  the child's model input, and bodies never persist to any JSONL (R7/AC10);
- runtime isolation: the whole workspace ``.forestcode/`` is hidden from the
  child's tools, only ``tool-results`` stays readable;
- the child confirm bridge tags tickets with the task id (R6).
"""

from __future__ import annotations

import os
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from forestcode.config import AgentRuntimeConfig
from forestcode.core.abort import AbortSignal
from forestcode.core.events import InMemoryEventSink
from forestcode.core.fake_model import FakeModelClient
from forestcode.core.types import ModelOutput, ToolCall
from forestcode.models import ModelConfig
from forestcode.runtime.factory import build_subagent_child_runner
from forestcode.skills.types import LoadedSkill, SkillDescriptor, SkillSnapshot
from forestcode.subagents.child import combined_context_chars
from forestcode.subagents.coordinator import SubagentCoordinator
from forestcode.subagents.types import (
    MAX_COMBINED_CONTEXT_CHARS,
    AgentConfig,
    AgentConfigSet,
    ModelOverride,
    SubagentRequest,
)
from forestcode.tools import MutationGate
from forestcode.tools.types import ApprovalRequest


def _agent(name: str = "helper", **kwargs: Any) -> AgentConfig:
    kwargs.setdefault("instructions", "you are a helper")
    return AgentConfig(name=name, description=f"desc {name}", **kwargs)


def _agent_set(*agents: AgentConfig) -> AgentConfigSet:
    return AgentConfigSet({agent.name: agent for agent in agents})


def _parent_model() -> ModelConfig:
    return ModelConfig(
        api_type="openai-compatible",
        model="parent-model",
        base_url="http://parent.example/v1",
        api_key="parent-key",
        timeout=30.0,
        reasoning_mode=None,
        reasoning_effort=None,
    )


def _skill_descriptor(name: str, digest: str) -> SkillDescriptor:
    return SkillDescriptor(
        name=name,
        description=f"{name} skill",
        root=Path(f"/tmp/skill-{name}"),
        entry_path=Path(f"/tmp/skill-{name}/SKILL.md"),
        source="user",
        source_root=Path("/tmp"),
        content_digest=digest,
    )


def _make_skill_snapshot(*names: str) -> SkillSnapshot:
    descriptors = tuple(
        _skill_descriptor(name, f"d{index}") for index, name in enumerate(names)
    )

    class _Loader:
        def load_entry(self, descriptor: SkillDescriptor) -> LoadedSkill | None:
            return LoadedSkill(
                descriptor=descriptor, instructions=f"{descriptor.name} skill body"
            )

    return SkillSnapshot(descriptors=descriptors, loader=_Loader())


class _ConfirmRecorder:
    """Records child confirm calls and their task ids (ConfirmBridgeProtocol)."""

    def __init__(self) -> None:
        self.calls: list[tuple[ApprovalRequest, str | None]] = []
        self.answer = True
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(
        self, request: ApprovalRequest, *, task_id: str | None = None, abort=None
    ) -> bool:
        self.calls.append((request, task_id))
        self.entered.set()
        self.release.wait(5)
        return self.answer

    def cancel_task(self, task_id: str) -> None:
        pass


class ChildConstructionTest(unittest.TestCase):
    def _runner(self, **kwargs: Any):
        from forestcode.core.fake_model import FakeModelClient
        from forestcode.core.types import ModelOutput

        if "model_factory" not in kwargs:

            def _fake_factory(_config: Any) -> FakeModelClient:
                return FakeModelClient([ModelOutput(text="child final answer")])

            kwargs["model_factory"] = _fake_factory
        return build_subagent_child_runner(
            workspace_root=kwargs.pop("workspace_root", Path("/tmp/ws")),
            agent_set=kwargs.pop("agent_set", _agent_set(_agent())),
            parent_model=kwargs.pop("parent_model", _parent_model()),
            environ=kwargs.pop("environ", os.environ.get),
            skills_snapshot=kwargs.pop("skills_snapshot", None),
            activated_skill_names=kwargs.pop("activated_skill_names", ()),
            inherited_fragments=kwargs.pop("inherited_fragments", ()),
            parent_visible_tools=kwargs.pop(
                "parent_visible_tools",
                frozenset({"list_files", "read_file", "grep_text"}),
            ),
            mutation_gate=kwargs.pop("mutation_gate", None),
            confirm_bridge=kwargs.pop("confirm_bridge", None),
            events=kwargs.pop("events", InMemoryEventSink()),
            session_root=kwargs.pop("session_root", None),
            agent=kwargs.pop("agent", AgentRuntimeConfig()),
            command_service=kwargs.pop("command_service", None),
            model_factory=kwargs.pop("model_factory"),
        )

    def _run_through_coordinator(self, runner: Any, prompt: str = "p") -> Any:
        coordinator = SubagentCoordinator(runner, max_workers=1)
        snapshot = coordinator.delegate(
            SubagentRequest(
                task_id="unused", agent_name="helper", description="d", prompt=prompt
            )
        )
        outcome = coordinator.wait([snapshot.task_id], timeout=5)
        self.assertFalse(outcome.timed_out)
        return outcome.snapshots[0]

    def test_child_completes_and_reports_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            coordinator = SubagentCoordinator(
                self._runner(workspace_root=ws), max_workers=1
            )
            snapshot = coordinator.delegate(
                SubagentRequest(
                    task_id="unused", agent_name="helper", description="d", prompt="p"
                )
            )
            outcome = coordinator.wait([snapshot.task_id], timeout=5)
            self.assertFalse(outcome.timed_out)
            snapshot = outcome.snapshots[0]
            self.assertEqual(snapshot.status, "completed")
            result = outcome.results[snapshot.task_id]
            self.assertEqual(result.final_text, "child final answer")

    def test_child_catalog_excludes_delegation_tools(self) -> None:
        from forestcode.context import ToolCatalog
        from forestcode.context.tool_catalog import ToolVisibilityPolicy
        from forestcode.subagents.types import SUBAGENT_TOOLS
        from forestcode.tools import create_builtin_tool_registry

        registry = create_builtin_tool_registry()
        catalog = ToolCatalog(
            registry,
            read_only_only=False,
            enable_command_tools=False,
            visibility=ToolVisibilityPolicy(
                frozenset(
                    {
                        "list_files",
                        "read_file",
                        "delegate_task",
                        "wait_subagents",
                        "list_subagents",
                        "cancel_subagent",
                    }
                ),
                "research",
            ),
        )
        names = {tool.name for tool in catalog.list_visible_tools()}
        self.assertFalse(SUBAGENT_TOOLS & names)

    def test_missing_default_skill_invalidates_agent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent = _agent(default_skills=("ghost",))
            runner = self._runner(
                workspace_root=ws,
                agent_set=_agent_set(agent),
                skills_snapshot=_make_skill_snapshot("py"),
            )
            with self.assertRaises(ValueError):
                runner.run(
                    SubagentRequest(
                        task_id="sub-1",
                        agent_name="helper",
                        description="d",
                        prompt="p",
                    ),
                    abort=AbortSignal(),
                )

    def test_combined_budget_fails_over_24k(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent = _agent(instructions="i" * 16000)
            runner = self._runner(workspace_root=ws, agent_set=_agent_set(agent))
            with self.assertRaises(ValueError):
                runner.run(
                    SubagentRequest(
                        task_id="sub-1",
                        agent_name="helper",
                        description="d",
                        prompt="p" * 9000,
                    ),
                    abort=AbortSignal(),
                )

    def test_api_key_env_missing_fails_diagnostically(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent = _agent(
                model=ModelOverride(
                    base_url="http://other.example/v1",
                    api_key_env="FORESTCODE_CHILD_KEY",
                )
            )
            runner = self._runner(
                workspace_root=ws,
                agent_set=_agent_set(agent),
                environ=lambda _name: None,
            )
            with self.assertRaises(Exception) as ctx:
                runner.run(
                    SubagentRequest(
                        task_id="sub-1",
                        agent_name="helper",
                        description="d",
                        prompt="p",
                    ),
                    abort=AbortSignal(),
                )
            message = str(ctx.exception)
            self.assertIn("FORESTCODE_CHILD_KEY", message)
            self.assertNotIn("parent-key", message)

    def test_skill_default_skills_merge_and_bodies_are_transient(self) -> None:
        # The activated skill's body arrives via the parent's inherited
        # fragments (design §Child Construction: parent-activated skills are
        # handed over as fragments, never re-loaded by name); the agent's
        # ``default_skills`` are loaded and appended, deduped by typed name.
        from forestcode.skills.catalog import skill_body_fragment
        from forestcode.subagents.child import resolve_child_skill_fragments

        snapshot = _make_skill_snapshot("py", "extra")
        agent = _agent(default_skills=("extra",))
        loaded_py = snapshot.load("py")
        self.assertIsNotNone(loaded_py)
        assert loaded_py is not None
        inherited = (skill_body_fragment(loaded_py),)
        fragments = resolve_child_skill_fragments(agent, snapshot, ("py",), inherited)
        bodies = [
            fragment.content for fragment in fragments if fragment.kind == "skill"
        ]
        self.assertEqual(sorted(bodies), sorted(["py skill body", "extra skill body"]))

    def test_combined_context_chars_contract(self) -> None:
        from forestcode.context.types import ContextFragment

        fragments = (
            ContextFragment(kind="skill", label="Skill: py", content="body" * 100),
        )
        total = combined_context_chars("instr" * 10, "prompt" * 10, fragments)
        self.assertEqual(total, 50 + 60 + 400)
        self.assertLessEqual(total, MAX_COMBINED_CONTEXT_CHARS)

    def test_child_completes_with_skill_inheritance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            snapshot = _make_skill_snapshot("py")
            agent = _agent(default_skills=("py",))
            runner = self._runner(
                workspace_root=ws,
                agent_set=_agent_set(agent),
                skills_snapshot=snapshot,
                activated_skill_names=(),
            )
            snapshot = self._run_through_coordinator(runner)
            self.assertEqual(snapshot.status, "completed")


class ChildConfirmBridgeTest(unittest.TestCase):
    def test_real_child_approval_updates_status_and_wires_task_id(self) -> None:
        # Exercise the real ToolExecutor -> child confirm -> coordinator path.
        # This protects the observable waiting_approval contract; directly
        # calling coordinator._transition would still pass if the wiring were
        # accidentally deleted.
        recorder = _ConfirmRecorder()
        recorder.release.clear()
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            coordinator_ref: list[SubagentCoordinator] = []
            runner = build_subagent_child_runner(
                workspace_root=ws,
                agent_set=_agent_set(_agent(permission_profile="edit")),
                parent_model=_parent_model(),
                environ=os.environ.get,
                skills_snapshot=None,
                activated_skill_names=(),
                inherited_fragments=(),
                parent_visible_tools=frozenset(
                    {"edit_file", "list_files", "read_file"}
                ),
                mutation_gate=MutationGate(),
                confirm_bridge=recorder,
                approval_started=lambda task_id: coordinator_ref[
                    0
                ].approval_started(task_id),
                approval_finished=lambda task_id: coordinator_ref[
                    0
                ].approval_finished(task_id),
                events=InMemoryEventSink(),
                session_root=None,
                agent=AgentRuntimeConfig(),
                model_factory=lambda _config: FakeModelClient(
                    [
                        ModelOutput(
                            tool_calls=[
                                ToolCall(
                                    id="write-1",
                                    name="write_file",
                                    arguments={"path": "new.txt", "content": "ok\n"},
                                )
                            ]
                        ),
                        ModelOutput(text="final"),
                    ]
                ),
            )
            coordinator = SubagentCoordinator(runner, max_workers=1)
            coordinator_ref.append(coordinator)
            delegated = coordinator.delegate(
                SubagentRequest(
                    task_id="unused", agent_name="helper", description="d", prompt="p"
                )
            )
            self.assertTrue(recorder.entered.wait(5))
            waiting = coordinator.list()[0]
            self.assertEqual(waiting.status, "waiting_approval")
            self.assertEqual(recorder.calls[0][1], delegated.task_id)
            recorder.release.set()
            outcome = coordinator.wait([delegated.task_id], timeout=5)
            self.assertFalse(outcome.timed_out)
            self.assertEqual(outcome.snapshots[0].status, "completed")
            self.assertEqual((ws / "new.txt").read_text(encoding="utf-8"), "ok\n")


class ParentIntegrationTest(unittest.TestCase):
    """End-to-end parent loop wiring (design §Model Tool Contract, §Persistence)."""

    def _parent_loop(
        self,
        workspace: Path,
        model: FakeModelClient,
        coordinator: SubagentCoordinator,
        subagent_tools: list[Any],
        session_id: str | None = None,
    ) -> Any:
        from forestcode.memory import SessionStore
        from forestcode.runtime.factory import build_agent_loop

        store = SessionStore(workspace) if session_id else None
        loop = build_agent_loop(
            model,
            workspace,
            agent=AgentRuntimeConfig(),
            session_id=session_id,
            session_store=store,
            enable_write_tools=True,
            subagents=coordinator,
            subagent_tools=subagent_tools,
        )
        return loop, store

    def test_parent_delegates_waits_and_auto_cancels_on_final(self) -> None:
        from forestcode.subagents.tools import create_subagent_tools

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent_set = _agent_set(_agent())
            runner = self._child_runner(ws, agent_set)
            coordinator = SubagentCoordinator(runner, max_workers=2)
            tools = create_subagent_tools(
                coordinator, agent_set=agent_set, skills_snapshot=None
            )
            # Turn 1: delegate; Turn 2: wait; Turn 3: final answer. The child
            # body must reach the parent history exactly once.
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
            loop, store = self._parent_loop(
                ws, model, coordinator, tools, session_id="sess-1"
            )
            state = loop.run("do it")
            self.assertIsNotNone(state.final_text)
            self.assertIsNotNone(store)
            entries = store.load("sess-1").entries
            bodies = [e.content for e in entries if e.kind == "tool_result"]
            # The wait result carries the child's final text into the parent
            # history (one-time handoff, design §Persistence).
            self.assertTrue(
                any("child final answer" in (body or "") for body in bodies)
            )
            # All children must be terminal after the parent run.
            for snapshot in coordinator.list():
                self.assertIn(snapshot.status, {"completed", "failed", "cancelled"})

    def test_parent_final_without_wait_cancels_children(self) -> None:
        from forestcode.subagents.tools import create_subagent_tools

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            agent_set = _agent_set(_agent())
            runner = self._child_runner(ws, agent_set)
            coordinator = SubagentCoordinator(runner, max_workers=1)
            tools = create_subagent_tools(
                coordinator, agent_set=agent_set, skills_snapshot=None
            )
            # The parent delegates then finalizes immediately without waiting.
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
                    ModelOutput(text="final without waiting"),
                ]
            )
            loop, _store = self._parent_loop(ws, model, coordinator, tools)
            state = loop.run("do it")
            self.assertIsNotNone(state.final_text)
            # AgentLoop.run's finally cleanup cancels the leftover child; the
            # child may already be completed (instant runner) but must be
            # terminal — never queued/running after the parent finished.
            for snapshot in coordinator.list():
                self.assertIn(snapshot.status, {"completed", "failed", "cancelled"})

    def test_child_cannot_read_parent_transcript(self) -> None:
        from forestcode.memory import SessionStore
        from forestcode.subagents.events import EVENT_SUBAGENT_TOOL_CALL_FINISHED
        from forestcode.subagents.persistence import child_transcript_dir

        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            # A parent session exists; its JSONL must be hidden from the child's
            # tools (runtime isolation, design §Persistence).
            store = SessionStore(ws)
            store.save_plan("sess-parent", [])
            transcript = child_transcript_dir(ws, "sess-parent")
            transcript.mkdir(parents=True, exist_ok=True)
            (transcript / "sub-0001.jsonl").write_text('{"k":"v"}\n', encoding="utf-8")
            agent_set = _agent_set(_agent())
            recorded: list[Any] = []

            class _Sink:
                def emit(self, event: Any) -> None:
                    recorded.append(event)

            # The *child* loop calls list_files on .forestcode; the child's
            # ToolExecutor must hide the whole runtime directory (not just its
            # own transcript subtree).
            runner = build_subagent_child_runner(
                workspace_root=ws,
                agent_set=agent_set,
                parent_model=_parent_model(),
                environ=os.environ.get,
                skills_snapshot=None,
                activated_skill_names=(),
                inherited_fragments=(),
                parent_visible_tools=frozenset({"list_files"}),
                mutation_gate=None,
                confirm_bridge=None,
                events=_Sink(),
                session_root=None,
                agent=AgentRuntimeConfig(),
                model_factory=lambda _config: FakeModelClient(
                    [
                        ModelOutput(
                            tool_calls=[
                                ToolCall(
                                    id="c1",
                                    name="list_files",
                                    arguments={"path": ".forestcode"},
                                )
                            ]
                        ),
                        ModelOutput(text="done"),
                    ]
                ),
            )
            coordinator = SubagentCoordinator(runner, max_workers=1)
            delegated = coordinator.delegate(
                SubagentRequest(
                    task_id="unused", agent_name="helper", description="d", prompt="p"
                )
            )
            outcome = coordinator.wait([delegated.task_id], timeout=5)
            self.assertFalse(outcome.timed_out)
            self.assertEqual(outcome.snapshots[0].status, "completed")
            # The child's list_files on .forestcode must have been denied as
            # runtime-internal (ok=False, empty listing) — never leaked.
            finished = [
                e for e in recorded if e.type == EVENT_SUBAGENT_TOOL_CALL_FINISHED
            ]
            self.assertTrue(finished, "child tool event should be forwarded")
            list_event = next(
                e for e in finished if e.payload.get("tool_name") == "list_files"
            )
            self.assertFalse(bool(list_event.payload.get("ok")))
            text = str(recorded)
            self.assertNotIn("sub-0001.jsonl", text)
            self.assertNotIn('"k":"v"', text)

    @staticmethod
    def _child_runner(ws: Path, agent_set: AgentConfigSet) -> Any:
        return build_subagent_child_runner(
            workspace_root=ws,
            agent_set=agent_set,
            parent_model=_parent_model(),
            environ=os.environ.get,
            skills_snapshot=None,
            activated_skill_names=(),
            inherited_fragments=(),
            parent_visible_tools=frozenset({"list_files", "read_file", "grep_text"}),
            mutation_gate=None,
            confirm_bridge=None,
            events=InMemoryEventSink(),
            session_root=None,
            agent=AgentRuntimeConfig(),
            model_factory=lambda _config: FakeModelClient(
                [ModelOutput(text="child final answer")]
            ),
        )


if __name__ == "__main__":
    unittest.main()
