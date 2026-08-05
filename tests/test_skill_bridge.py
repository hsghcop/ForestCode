"""Bridge-level skills behavior tests: activation, pending lifecycle, cleanup (PRD R4-R6)."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forestcode.config import AgentRuntimeConfig
from forestcode.core.abort import Aborted
from forestcode.core.fake_model import FakeModelClient
from forestcode.core.types import ModelOutput
from forestcode.models.types import ModelAdapterError
from forestcode.terminal.bridge import BackendBridge, BackendBridgeConfig
from forestcode.terminal.confirm import ConfirmationController
from forestcode.terminal.input import StdinInputController
from forestcode.terminal.renderer import PlainRenderer


def _make_skill(
    workspace: Path, name: str, description: str = "desc", body: str = "BODY"
) -> None:
    skill_dir = workspace / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}", encoding="utf-8"
    )


def _build_bridge(
    workspace: Path,
    model,
    *,
    selector=None,
    pending=None,
    subagent_selector=None,
    subagent_pending=None,
):
    out, err = io.StringIO(), io.StringIO()
    renderer = PlainRenderer(out, err)
    input_controller = StdinInputController(lambda _prompt: "")
    confirm = ConfirmationController(renderer, input_controller)
    bridge = BackendBridge(
        BackendBridgeConfig(
            workspace_root=workspace,
            session_id="default",
            agent=AgentRuntimeConfig(),
            model=model,
            renderer=renderer,
            input_controller=input_controller,
            confirmation_controller=confirm,
            stdout=out,
            stderr=err,
            pending_skill_selection=pending,
            skill_selector=selector,
            pending_subagent_selection=subagent_pending,
            subagent_selector=subagent_selector,
        )
    )
    return bridge, out, err


class SkillBridgeClassifyTest(unittest.TestCase):
    def test_explicit_skill_token_activates_and_cleans_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "refactor", body="REFACTOR-BODY")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model)

            decision = bridge.classify("$refactor 帮我重构这个文件")
            self.assertEqual(decision.kind, "run")
            self.assertEqual(decision.task, "帮我重构这个文件")
            self.assertIsNotNone(decision.skills_snapshot)
            kinds = [f.kind for f in decision.transient_fragments]
            self.assertIn("skills_catalog", kinds)
            self.assertIn("skill", kinds)
            catalog = next(
                f for f in decision.transient_fragments if f.kind == "skills_catalog"
            )
            self.assertIn("- refactor: desc", catalog.content)
            body = next(f for f in decision.transient_fragments if f.kind == "skill")
            self.assertEqual(body.content, "REFACTOR-BODY")

    def test_unknown_explicit_skill_is_user_error_no_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "known")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, err = _build_bridge(workspace, model)

            decision = bridge.classify("$missing do it")
            self.assertEqual(decision.kind, "noop")
            self.assertIn("Unknown skill: missing", err.getvalue())
            self.assertEqual(model.inputs, [])  # model never called

    def test_explicit_skill_load_failure_is_user_error_no_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "known")
            model = FakeModelClient([ModelOutput(text="must not run")])
            bridge, _out, err = _build_bridge(workspace, model)

            with patch(
                "forestcode.skills.loader.SkillLoader.load_entry", return_value=None
            ):
                decision = bridge.classify("$known do it")

            self.assertEqual(decision.kind, "noop")
            self.assertIn("Skill could not be loaded: known", err.getvalue())
            self.assertEqual(model.inputs, [])

    def test_token_without_task_is_user_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "known")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, err = _build_bridge(workspace, model)

            decision = bridge.classify("$known")
            self.assertEqual(decision.kind, "noop")
            self.assertIn("requires a task", err.getvalue())
            self.assertEqual(model.inputs, [])

    def test_plain_task_gets_catalog_fragment_when_skills_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model)

            decision = bridge.classify("hello")
            self.assertEqual(decision.kind, "run")
            self.assertEqual(decision.task, "hello")
            self.assertEqual(
                [f.kind for f in decision.transient_fragments], ["skills_catalog"]
            )
            self.assertIsNotNone(decision.skills_snapshot)

    def test_no_skills_keeps_behavior_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model)

            decision = bridge.classify("hello")
            self.assertEqual(decision.kind, "run")
            self.assertEqual(decision.transient_fragments, ())
            self.assertIsNone(decision.skills_snapshot)


class SkillBridgePendingTest(unittest.TestCase):
    def test_pending_selection_consumed_on_next_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo")
            from forestcode.skills import PendingSkillSelection

            pending = PendingSkillSelection()
            pending.replace("demo")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model, pending=pending)

            self.assertEqual(bridge.pending_skill_marker(), "[Skill: demo]")
            decision = bridge.classify("question")
            self.assertEqual(decision.kind, "run")
            self.assertIn("skill", [f.kind for f in decision.transient_fragments])

    def test_run_one_turn_clears_pending_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo")
            from forestcode.skills import PendingSkillSelection

            pending = PendingSkillSelection()
            pending.replace("demo")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model, pending=pending)
            snapshot = bridge.classify("q").skills_snapshot

            outcome = bridge.run_one_turn(
                "q",
                sink=lambda _event: None,
                confirm=lambda _request: True,
                skills_snapshot=snapshot,
            )
            self.assertEqual(outcome.outcome.action, "continue")
            self.assertIsNone(pending.name)  # consumed

    def test_run_one_turn_clears_pending_on_model_error(self):
        class _RaisingModel:
            def complete(self, model_input, *, abort=None):
                raise ModelAdapterError("boom")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            from forestcode.skills import PendingSkillSelection

            pending = PendingSkillSelection()
            pending.replace("demo")
            bridge, _out, _err = _build_bridge(
                workspace, _RaisingModel(), pending=pending
            )

            outcome = bridge.run_one_turn(
                "q", sink=lambda _e: None, confirm=lambda _r: True
            )
            self.assertEqual(outcome.outcome.action, "error")
            self.assertIsNone(pending.name)

    def test_run_one_turn_clears_pending_on_abort(self):
        class _AbortingModel:
            def complete(self, model_input, *, abort=None):
                raise Aborted()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            from forestcode.skills import PendingSkillSelection

            pending = PendingSkillSelection()
            pending.replace("demo")
            bridge, _out, _err = _build_bridge(
                workspace, _AbortingModel(), pending=pending
            )

            with self.assertRaises(Aborted):
                bridge.run_one_turn("q", sink=lambda _e: None, confirm=lambda _r: True)
            self.assertIsNone(pending.name)

    def test_empty_input_and_plain_slash_keep_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo")
            from forestcode.skills import PendingSkillSelection

            pending = PendingSkillSelection()
            pending.replace("demo")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model, pending=pending)

            bridge.classify("")
            self.assertEqual(pending.name, "demo")
            bridge.classify("/memory")
            self.assertEqual(pending.name, "demo")

    def test_session_switch_clears_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo")
            from forestcode.skills import PendingSkillSelection

            pending = PendingSkillSelection()
            pending.replace("demo")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model, pending=pending)

            bridge.switch_session("other")
            self.assertIsNone(pending.name)


class SkillSlashTest(unittest.TestCase):
    def test_skills_command_sets_pending_via_selector(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo")
            model = FakeModelClient([ModelOutput(text="ok")])
            selected: list[list[str]] = []

            def selector(snapshot):
                selected.append([d.name for d in snapshot.descriptors])
                return "demo"

            bridge, _out, _err = _build_bridge(workspace, model, selector=selector)
            decision = bridge.classify("/skills")
            self.assertEqual(decision.kind, "noop")
            self.assertEqual(selected, [["demo"]])
            self.assertEqual(bridge.pending_skill_marker(), "[Skill: demo]")

    def test_skills_cancel_keeps_pending_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo")
            from forestcode.skills import PendingSkillSelection

            pending = PendingSkillSelection()
            pending.replace("other")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(
                workspace, model, selector=lambda _snapshot: None, pending=pending
            )
            bridge.classify("/skills")
            self.assertEqual(bridge.pending_skill_marker(), "[Skill: other]")

    def test_skills_no_skills_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, out, _err = _build_bridge(
                workspace, model, selector=lambda _snapshot: "x"
            )
            decision = bridge.classify("/skills")
            self.assertEqual(decision.kind, "noop")
            self.assertIn("no skills found", out.getvalue())

    def test_explicit_token_overrides_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "alpha")
            _make_skill(workspace, "beta")
            from forestcode.skills import PendingSkillSelection

            pending = PendingSkillSelection()
            pending.replace("alpha")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model, pending=pending)

            decision = bridge.classify("$beta do it")
            self.assertEqual(decision.kind, "run")
            self.assertEqual(decision.task, "do it")
            body = next(f for f in decision.transient_fragments if f.kind == "skill")
            self.assertEqual(body.label, "Skill: beta")


class SkillFrontendStateTest(unittest.TestCase):
    def test_plain_input_controller_renders_marker_prefix(self):
        from forestcode.skills import PendingSkillSelection

        pending = PendingSkillSelection()
        pending.replace("demo")
        captured: list[str] = []

        def input_func(prompt: str) -> str:
            captured.append(prompt)
            return "hi"

        controller = StdinInputController(
            input_func, marker_provider=pending.marker_text
        )
        controller.read_user_input("ForestCode> ")
        self.assertIn("[Skill: demo]", captured[0])

    def test_plain_input_controller_without_marker(self):
        captured: list[str] = []

        def input_func(prompt: str) -> str:
            captured.append(prompt)
            return "hi"

        controller = StdinInputController(input_func)
        controller.read_user_input("ForestCode> ")
        self.assertEqual(captured, ["ForestCode> "])


class ManualDelegationBridgeTest(unittest.TestCase):
    """/subagents selection -> next task -> direct child response."""

    def _make_agent(self, workspace: Path, name: str = "helper") -> None:
        agent_dir = workspace / ".agents" / "subagents" / f"{name}.md"
        agent_dir.parent.mkdir(parents=True, exist_ok=True)
        agent_dir.write_text(
            f"---\nname: {name}\ndescription: desc\n---\nBODY", encoding="utf-8"
        )

    def test_singular_subagent_command_is_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge, _out, err = _build_bridge(
                Path(tmp), FakeModelClient([ModelOutput(text="must not run")])
            )

            decision = bridge.classify("/subagent helper do it")

            self.assertEqual(decision.kind, "noop")
            self.assertIn("Unknown command: /subagent", err.getvalue())

    def test_selection_is_pending_until_next_ordinary_task(self):
        from forestcode.subagents import PendingSubagentSelection

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_agent(workspace)
            pending = PendingSubagentSelection()
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(
                workspace,
                model,
                subagent_pending=pending,
                subagent_selector=lambda _snapshot: "helper",
            )

            selected = bridge.classify("/subagents")
            self.assertEqual(selected.kind, "noop")
            self.assertEqual(bridge.pending_subagent_marker(), "[Subagent: helper]")
            self.assertEqual(bridge.classify("").kind, "noop")
            self.assertEqual(bridge.pending_subagent_marker(), "[Subagent: helper]")
            self.assertEqual(bridge.classify("/memory").kind, "noop")
            self.assertEqual(bridge.pending_subagent_marker(), "[Subagent: helper]")

            decision = bridge.classify("do the thing")
            self.assertEqual(decision.kind, "run")
            self.assertEqual(decision.task, "do the thing")
            launch = decision.launch_context
            self.assertIsNotNone(launch)
            assert launch is not None
            manual = launch.manual_delegation
            self.assertIsNotNone(manual)
            assert manual is not None
            self.assertEqual(manual.agent_name, "helper")
            self.assertEqual(manual.task, "do the thing")

    def test_session_switch_clears_pending_subagent(self):
        from forestcode.subagents import PendingSubagentSelection

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            pending = PendingSubagentSelection("helper")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(
                workspace, model, subagent_pending=pending
            )
            bridge.switch_session("other")
            self.assertIsNone(pending.name)

    def test_deleted_pending_subagent_is_user_error_without_model_call(self):
        from forestcode.subagents import PendingSubagentSelection

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_agent(workspace)
            pending = PendingSubagentSelection()
            model = FakeModelClient([ModelOutput(text="must not run")])
            bridge, _out, err = _build_bridge(
                workspace,
                model,
                subagent_pending=pending,
                subagent_selector=lambda _snapshot: "helper",
            )
            bridge.classify("/subagents")
            (workspace / ".agents" / "subagents" / "helper.md").unlink()

            decision = bridge.classify("do it")

            self.assertEqual(decision.kind, "noop")
            self.assertIn("no longer available", err.getvalue())
            self.assertEqual(model.inputs, [])
            self.assertIsNone(pending.name)

    def test_exit_clears_pending_subagent(self):
        from forestcode.subagents import PendingSubagentSelection

        with tempfile.TemporaryDirectory() as tmp:
            pending = PendingSubagentSelection("helper")
            bridge, _out, _err = _build_bridge(
                Path(tmp),
                FakeModelClient([ModelOutput(text="must not run")]),
                subagent_pending=pending,
            )

            decision = bridge.classify("/exit")

            self.assertEqual(decision.kind, "exit")
            self.assertIsNone(pending.name)

    def test_manual_delegation_returns_child_text_without_parent_model_call(self):
        from forestcode.subagents import PendingSubagentSelection
        from forestcode.subagents.types import SubagentResult

        class _FakeRouter:
            """ModelClient stand-in exposing ``config`` like the real ModelRouter."""

            def __init__(self) -> None:
                from forestcode.models import ModelConfig

                self.config = ModelConfig(
                    api_type="openai-compatible",
                    model="parent",
                    base_url="http://parent/v1",
                    api_key="k",
                    timeout=30.0,
                    reasoning_mode=None,
                    reasoning_effort=None,
                )
                self.inputs = []

            def complete(self, model_input, *, abort=None) -> ModelOutput:
                self.inputs.append(model_input)
                raise AssertionError("manual delegation must not call parent model")

        def _fake_child_runner(
            *, workspace_root, agent_set, parent_model, environ, **kwargs
        ):
            class _InstantRunner:
                def run(self, request, *, abort) -> SubagentResult:
                    return SubagentResult(
                        task_id=request.task_id,
                        agent_name=request.agent_name,
                        final_text="child done",
                        turn_count=1,
                        tool_count=0,
                    )

            return _InstantRunner()

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_agent(workspace)
            pending = PendingSubagentSelection()
            router = _FakeRouter()
            bridge, _out, _err = _build_bridge(
                workspace,
                router,
                subagent_pending=pending,
                subagent_selector=lambda _snapshot: "helper",
            )

            bridge.classify("/subagents")
            decision = bridge.classify("do it")
            self.assertEqual(decision.kind, "run")
            launch = decision.launch_context
            self.assertIsNotNone(launch)
            assert launch is not None
            self.assertIsNotNone(launch.manual_delegation)

            events = []
            with patch(
                "forestcode.terminal.bridge.build_subagent_child_runner",
                side_effect=_fake_child_runner,
            ):
                outcome = bridge.run_one_turn(
                    decision.task or "",
                    sink=events.append,
                    confirm=lambda _request: True,
                    transient_fragments=decision.transient_fragments,
                    skills_snapshot=decision.skills_snapshot,
                    launch_context=launch,
                )
            self.assertEqual(outcome.outcome.action, "continue")
            self.assertEqual(outcome.outcome.run_state.final_text, "child done")
            self.assertEqual(router.inputs, [])
            assistant_events = [
                event for event in events if event.type == "assistant_text_received"
            ]
            self.assertEqual(
                [event.payload["text"] for event in assistant_events], ["child done"]
            )
            self.assertIsNone(pending.name)
            messages = [
                (entry.role, entry.content)
                for entry in bridge._session_store.load("default").entries
                if entry.kind == "message"
            ]
            self.assertEqual(messages, [("user", "do it"), ("assistant", "child done")])

    def test_manual_delegation_validation_failure_clears_pending(self):
        from forestcode.subagents import PendingSubagentSelection

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_agent(workspace)
            pending = PendingSubagentSelection()
            bridge, _out, _err = _build_bridge(
                workspace,
                FakeModelClient([ModelOutput(text="parent must not run")]),
                subagent_pending=pending,
                subagent_selector=lambda _snapshot: "helper",
            )
            bridge.classify("/subagents")
            decision = bridge.classify("do it")

            execution = bridge.run_one_turn(
                decision.task or "",
                sink=lambda _event: None,
                confirm=lambda _request: True,
                transient_fragments=decision.transient_fragments,
                skills_snapshot=decision.skills_snapshot,
                launch_context=decision.launch_context,
            )

            self.assertEqual(execution.outcome.action, "error")
            self.assertIn("parent model configuration", execution.outcome.error)
            self.assertIsNone(pending.name)


class SkillActivationFeedbackTest(unittest.TestCase):
    def test_skill_loaded_event_is_emitted_once_with_name_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo", body="SECRET-BODY")
            model = FakeModelClient([ModelOutput(text="ok")])
            bridge, _out, _err = _build_bridge(workspace, model)
            decision = bridge.classify("$demo inspect")
            events = []

            bridge.run_one_turn(
                decision.task or "",
                sink=events.append,
                confirm=lambda _request: True,
                transient_fragments=decision.transient_fragments,
                skills_snapshot=decision.skills_snapshot,
                launch_context=decision.launch_context,
            )

            loaded = [event for event in events if event.type == "skill_activated"]
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].payload, {"name": "demo"})
            self.assertNotIn("SECRET-BODY", repr(loaded[0]))


if __name__ == "__main__":
    unittest.main()
