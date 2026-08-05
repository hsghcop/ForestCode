import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forestcode.cli import main, print_agent_event, print_agent_events, run_chat
from forestcode.core import (
    AssistantTurn,
    FakeModelClient,
    ModelOutput,
    ReasoningArtifact,
    ToolCall,
)
from forestcode.core.types import RunEvent
from forestcode.models import ModelAdapterError

_MODEL_ENV = {
    "FORESTCODE_MODEL": "deepseek-chat",
    "FORESTCODE_BASE_URL": "https://api.deepseek.com/v1",
    "FORESTCODE_API_KEY": "secret",
}


class FailingModel:
    def complete(self, model_input, *, abort=None):
        raise ModelAdapterError("bad config")


class ChatCliTest(unittest.TestCase):
    def test_run_chat_selects_subagent_and_prints_child_response_directly(self):
        from forestcode.models import ModelConfig
        from forestcode.subagents.types import SubagentResult

        class Router:
            def __init__(self) -> None:
                self.config = ModelConfig(
                    api_type="openai-compatible",
                    model="parent",
                    base_url="http://parent/v1",
                    api_key="key",
                    timeout=30,
                )
                self.inputs = []

            def complete(self, model_input, *, abort=None):
                self.inputs.append(model_input)
                raise AssertionError("manual subagent flow must not call parent model")

        def fake_child_runner(**_kwargs):
            class Runner:
                def run(self, request, *, abort):
                    return SubagentResult(
                        task_id=request.task_id,
                        agent_name=request.agent_name,
                        final_text="child result",
                    )

            return Runner()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_file = root / ".agents" / "subagents" / "helper.md"
            agent_file.parent.mkdir(parents=True)
            agent_file.write_text(
                "---\nname: helper\ndescription: checks code\n---\nReview carefully",
                encoding="utf-8",
            )
            router = Router()
            stdout = io.StringIO()
            stderr = io.StringIO()
            inputs = iter(["/subagents", "1", "inspect this", "/exit"])
            with patch(
                "forestcode.terminal.bridge.build_subagent_child_runner",
                side_effect=fake_child_runner,
            ):
                code = run_chat(
                    model=router,
                    input_func=lambda _prompt: next(inputs),
                    stdout=stdout,
                    stderr=stderr,
                    workspace_root=root,
                )

        self.assertEqual(code, 0)
        self.assertIn("1. helper — checks code · permission: research", stdout.getvalue())
        self.assertIn("Assistant> child result", stdout.getvalue())
        self.assertEqual(router.inputs, [])
        self.assertEqual(stderr.getvalue(), "")

    def test_run_chat_runs_agent_loop_and_prints_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient([ModelOutput(text="你好")])
            inputs = iter(["hello", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=Path(tmp),
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("ForestCode chat", stdout.getvalue())
            self.assertIn("Assistant> 你好", stdout.getvalue())
            self.assertEqual(stdout.getvalue().count("Assistant> 你好"), 1)
            self.assertEqual(model.inputs[0].messages[0].content, "hello")
            self.assertEqual(stderr.getvalue(), "")

    def test_run_chat_executes_tool_calls_through_agent_loop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            model = FakeModelClient(
                [
                    ModelOutput(
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="list_files",
                                arguments={"path": ".", "max_entries": 10},
                            )
                        ]
                    ),
                    ModelOutput(text="看完了"),
                ]
            )
            inputs = iter(["list files", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=root,
            )

        self.assertEqual(exit_code, 0)
        self.assertIn("Tool> list_files", stdout.getvalue())
        self.assertIn("Tool> list_files ok", stdout.getvalue())
        self.assertIn("Assistant> 看完了", stdout.getvalue())
        self.assertEqual(stdout.getvalue().count("Tool> list_files"), 2)
        self.assertEqual(stdout.getvalue().count("Assistant> 看完了"), 1)
        self.assertEqual(stderr.getvalue(), "")

    def test_run_chat_reports_model_error(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = run_chat(
            model=FailingModel(),
            input_func=lambda prompt: "hello",
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 1)
        self.assertIn("Model error: bad config", stderr.getvalue())

    def test_run_chat_handles_unknown_slash_without_model_call(self):
        model = FakeModelClient([])
        inputs = iter(["/cmpact", "/exit"])
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = run_chat(
            model=model,
            input_func=lambda prompt: next(inputs),
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("Unknown command: /cmpact", stderr.getvalue())
        self.assertIn("Available:", stderr.getvalue())
        self.assertEqual(model.inputs, [])

    def test_run_chat_handles_uppercase_slash_exit(self):
        model = FakeModelClient([])
        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = run_chat(
            model=model,
            input_func=lambda prompt: "/Exit",
            stdout=stdout,
            stderr=stderr,
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(model.inputs, [])
        self.assertEqual(stderr.getvalue(), "")

    def test_run_chat_allows_path_like_slash_message_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient([ModelOutput(text="done")])
            inputs = iter(["/path/to/file", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=Path(tmp),
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(model.inputs[0].messages[0].content, "/path/to/file")
            self.assertEqual(stderr.getvalue(), "")

    def test_run_chat_allows_question_slash_message_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient([ModelOutput(text="done")])
            inputs = iter(["/?", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=Path(tmp),
            )

            self.assertEqual(exit_code, 0)
            self.assertEqual(model.inputs[0].messages[0].content, "/?")
            self.assertEqual(stderr.getvalue(), "")

    def test_run_chat_switches_session_for_following_turns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = FakeModelClient([ModelOutput(text="done")])
            inputs = iter(["/switch s2", "hello", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=root,
                session_id="s1",
            )

            self.assertEqual(exit_code, 0)
            self.assertIn("Session> s2", stdout.getvalue())
            self.assertTrue((root / ".forestcode" / "sessions" / "s2.jsonl").exists())
            self.assertIn("Memory> recorded .forestcode/sessions/s2.jsonl", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_run_chat_switch_from_no_session_enables_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = FakeModelClient([ModelOutput(text="done")])
            inputs = iter(["/switch s2", "hello", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=root,
                session_id=None,
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue((root / ".forestcode" / "sessions" / "s2.jsonl").exists())
            self.assertIn("Memory> recorded .forestcode/sessions/s2.jsonl", stdout.getvalue())
            self.assertEqual(stderr.getvalue(), "")

    def test_run_chat_empty_stop_is_success_without_memory_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient([ModelOutput(AssistantTurn(text="", finish_reason="stop"))])
            inputs = iter(["empty stop", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=Path(tmp),
                session_id="s1",
            )

            self.assertEqual(exit_code, 0)
            self.assertNotIn("Memory> recorded", stdout.getvalue())
            self.assertNotIn("Agent error", stderr.getvalue())

    def test_run_chat_hides_reasoning_by_default(self):
        model = FakeModelClient(
            [
                ModelOutput(
                    AssistantTurn(
                        text="done",
                        reasoning_artifacts=[
                            ReasoningArtifact(
                                provider="deepseek",
                                kind="reasoning_content",
                                visible=True,
                                display_text="thinking",
                            )
                        ],
                    )
                )
            ]
        )
        inputs = iter(["hello", "/exit"])
        stdout = io.StringIO()

        exit_code = run_chat(model=model, input_func=lambda prompt: next(inputs), stdout=stdout)

        self.assertEqual(exit_code, 0)
        self.assertNotIn("Reasoning>", stdout.getvalue())
        self.assertIn("Assistant> done", stdout.getvalue())

    def test_run_chat_prints_reasoning_when_enabled(self):
        model = FakeModelClient(
            [
                ModelOutput(
                    AssistantTurn(
                        text="done",
                        reasoning_artifacts=[
                            ReasoningArtifact(
                                provider="deepseek",
                                kind="reasoning_content",
                                visible=True,
                                display_text="thinking",
                            )
                        ],
                    )
                )
            ]
        )
        inputs = iter(["hello", "/exit"])
        stdout = io.StringIO()

        exit_code = run_chat(
            model=model,
            input_func=lambda prompt: next(inputs),
            stdout=stdout,
            show_reasoning=True,
        )

        self.assertEqual(exit_code, 0)
        self.assertIn("Reasoning> thinking", stdout.getvalue())
        self.assertIn("Assistant> done", stdout.getvalue())

    def test_print_agent_event_prints_assistant_text(self):
        stdout = io.StringIO()

        print_agent_event(RunEvent("assistant_text_received", {"text": "done"}), stdout)

        self.assertEqual(stdout.getvalue(), "Assistant> done\n")

    def test_print_agent_event_prints_compaction_events(self):
        stdout = io.StringIO()

        print_agent_event(RunEvent("session_compaction_finished", {"kind": "normal"}), stdout)
        print_agent_event(RunEvent("session_compaction_failed", {"kind": "major", "error": "model unavailable"}), stdout)

        self.assertIn("Memory> compacted normal", stdout.getvalue())
        self.assertIn("Memory> compaction failed (major): model unavailable", stdout.getvalue())

    def test_print_agent_events_uses_single_event_renderer(self):
        stdout = io.StringIO()

        print_agent_events(
            [
                RunEvent("tool_call_started", {"tool_name": "list_files"}),
                RunEvent("tool_call_finished", {"tool_name": "list_files", "ok": True}),
                RunEvent("assistant_text_received", {"text": "done"}),
            ],
            stdout,
        )

        self.assertIn("Tool> list_files\n", stdout.getvalue())
        self.assertIn("Tool> list_files ok\n", stdout.getvalue())
        self.assertIn("Assistant> done\n", stdout.getvalue())


class MainConfigWiringTest(unittest.TestCase):
    def _run_main(self, argv, env):
        captured = {}

        def fake_run_chat(model, **kwargs):
            captured["agent"] = kwargs.get("agent")
            return 0

        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / ".forestcode" / "settings.json"
            with patch("forestcode.config.settings_file.default_settings_path", return_value=settings_path):
                with patch.dict(os.environ, env, clear=True):
                    with patch("forestcode.cli.run_chat", side_effect=fake_run_chat):
                        exit_code = main(argv)
        return exit_code, captured["agent"]

    def test_cli_flags_thread_into_agent_config(self):
        exit_code, agent = self._run_main(
            ["chat", "--max-turns", "5", "--no-command-tools"], _MODEL_ENV
        )
        self.assertEqual(exit_code, 0)
        self.assertEqual(agent.runtime.max_turns, 5)
        self.assertFalse(agent.features.enable_command_tools)

    def test_cli_overrides_process_env(self):
        # process env enables command tools; CLI --no-command-tools turns it back off
        env = {**_MODEL_ENV, "FORESTCODE_MAX_TURNS": "9", "FORESTCODE_ENABLE_COMMAND_TOOLS": "1"}
        exit_code, agent = self._run_main(["chat", "--no-command-tools"], env)
        self.assertEqual(exit_code, 0)
        self.assertEqual(agent.runtime.max_turns, 9)  # from process env
        self.assertFalse(agent.features.enable_command_tools)  # CLI beats env

    def test_command_tools_flag_enables(self):
        env = {**_MODEL_ENV}
        exit_code, agent = self._run_main(["chat", "--command-tools"], env)
        self.assertEqual(exit_code, 0)
        self.assertTrue(agent.features.enable_command_tools)

    def test_invalid_config_returns_exit_code_2(self):
        # Keep settings present with api_key so first-run guidance does not
        # intercept; main should still report the missing model/base_url error.
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings_path = root / ".forestcode" / "settings.json"
            settings_path.parent.mkdir(parents=True)
            settings_path.write_text('{"api_key": "secret"}', encoding="utf-8")
            os.chdir(root)
            try:
                with patch("forestcode.config.settings_file.default_settings_path", return_value=settings_path):
                    with patch.dict(os.environ, {}, clear=True):
                        self.assertEqual(main(["chat"]), 2)  # missing FORESTCODE_MODEL
            finally:
                os.chdir(original)

    def test_first_run_creates_template_when_config_missing(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings_path = root / ".forestcode" / "settings.json"
            os.chdir(root)
            try:
                with patch("forestcode.config.settings_file.default_settings_path", return_value=settings_path):
                    with patch.dict(os.environ, {}, clear=True):
                        self.assertEqual(main(["chat"]), 0)
            finally:
                os.chdir(original)
            self.assertTrue(settings_path.exists())

    def test_empty_settings_api_key_does_not_block_complete_dotenv(self):
        original = os.getcwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            settings_path = root / ".forestcode" / "settings.json"
            settings_path.parent.mkdir(parents=True)
            settings_path.write_text('{"api_key": ""}', encoding="utf-8")
            (root / ".env").write_text(
                "\n".join(
                    [
                        "FORESTCODE_MODEL=deepseek-chat",
                        "FORESTCODE_BASE_URL=https://api.deepseek.com/v1",
                        "FORESTCODE_API_KEY=secret",
                    ]
                ),
                encoding="utf-8",
            )
            captured = {}

            def fake_run_chat(model, **kwargs):
                captured["agent"] = kwargs.get("agent")
                return 0

            os.chdir(root)
            try:
                with patch("forestcode.config.settings_file.default_settings_path", return_value=settings_path):
                    with patch.dict(os.environ, {}, clear=True):
                        with patch("forestcode.cli.run_chat", side_effect=fake_run_chat):
                            self.assertEqual(main(["chat"]), 0)
            finally:
                os.chdir(original)
        self.assertIsNotNone(captured["agent"])


class InputControllerTierTest(unittest.TestCase):
    def _tty(self):
        class TTY(io.StringIO):
            def isatty(self):
                return True

        return TTY()

    def test_non_tty_is_plain(self):
        from forestcode.cli import build_input_controller
        from forestcode.terminal.input import StdinInputController

        c = build_input_controller(
            no_color=False, no_history=False, session_id="s",
            workspace_root=Path("."), model_name="m", stdout=io.StringIO(),
        )
        self.assertIsInstance(c, StdinInputController)

    def test_no_color_forces_plain(self):
        from forestcode.cli import build_input_controller
        from forestcode.terminal.input import StdinInputController

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NO_COLOR", None)
            c = build_input_controller(
                no_color=True, no_history=False, session_id="s",
                workspace_root=Path("."), model_name="m", stdout=self._tty(),
            )
        self.assertIsInstance(c, StdinInputController)


class LegacyChatSkillsTest(unittest.TestCase):
    """PRD R4/R6 regression: legacy ``run_chat`` must consume a ``/skills``
    selection on the next real task, apply it only to that run, and clear it
    afterwards — without consuming it on empty input or plain slash commands.
    """

    def _make_skill(self, workspace: Path, name: str, body: str = "BODY") -> None:
        skill_dir = workspace / ".agents" / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: desc {name}\n---\n{body}", encoding="utf-8"
        )

    def test_skills_selection_applies_to_next_task_and_is_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_skill(workspace, "refactor", body="REFACTOR-BODY")
            model = FakeModelClient([ModelOutput(text="ok"), ModelOutput(text="ok2")])
            # /skills -> pick 1 -> real task -> next task without selection.
            inputs = iter(["/skills", "1", "重构一下", "再问一次", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=workspace,
            )

            self.assertEqual(exit_code, 0)
            # First real task saw the manually activated skill body as a
            # transient fragment (PRD R6: visible in this run's iterations).
            self.assertGreaterEqual(len(model.inputs), 2)
            first_messages = model.inputs[0].messages
            joined = "\n".join(m.content or "" for m in first_messages)
            self.assertIn("REFACTOR-BODY", joined)
            # Second task must NOT see the skill body (selection was consumed).
            second_messages = model.inputs[1].messages
            joined2 = "\n".join(m.content or "" for m in second_messages)
            self.assertNotIn("REFACTOR-BODY", joined2)

    def test_explicit_token_wins_and_pending_is_cleared(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_skill(workspace, "refactor", body="REFACTOR-BODY")
            self._make_skill(workspace, "testing", body="TEST-BODY")
            model = FakeModelClient([ModelOutput(text="ok")])
            # Pick 'testing' via /skills, then override with an explicit
            # $refactor token on the next input (R4: explicit wins).
            inputs = iter(["/skills", "2", "$refactor 重构", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=workspace,
            )

            self.assertEqual(exit_code, 0)
            messages = model.inputs[0].messages
            joined = "\n".join(m.content or "" for m in messages)
            self.assertIn("REFACTOR-BODY", joined)
            self.assertNotIn("TEST-BODY", joined)

    def test_empty_input_and_plain_slash_do_not_consume_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_skill(workspace, "refactor", body="REFACTOR-BODY")
            model = FakeModelClient([ModelOutput(text="ok")])
            # /skills -> pick 1 -> empty input (noop) -> plain slash (noop)
            # -> real task must still see the skill.
            inputs = iter(["/skills", "1", "", "/sessions", "真正的问题", "/exit"])
            stdout = io.StringIO()
            stderr = io.StringIO()

            exit_code = run_chat(
                model=model,
                input_func=lambda prompt: next(inputs),
                stdout=stdout,
                stderr=stderr,
                workspace_root=workspace,
            )

            self.assertEqual(exit_code, 0)
            messages = model.inputs[0].messages
            joined = "\n".join(m.content or "" for m in messages)
            self.assertIn("REFACTOR-BODY", joined)

    def test_unknown_pending_skill_is_cleared_with_warning(self):
        # PRD R4: a pending selection that no longer resolves (skill deleted
        # after /skills) is cleared with a warning; the task still runs.
        # Simulated by making refresh return an empty snapshot between turns.
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_skill(workspace, "refactor", body="REFACTOR-BODY")
            model = FakeModelClient([ModelOutput(text="ok")])
            stdout = io.StringIO()
            stderr = io.StringIO()

            real_refresh = None
            calls = {"n": 0}

            def patched_refresh(reg):
                calls["n"] += 1
                if calls["n"] == 1:
                    return real_refresh(reg)
                # After the /skills selection, the skill disappears.
                return type(real_refresh(reg))(descriptors=(), issues=(), loader=None)

            from forestcode.skills.registry import SkillRegistry

            real_refresh = SkillRegistry.refresh
            with patch.object(SkillRegistry, "refresh", patched_refresh):
                inputs = iter(["/skills", "1", "任务", "/exit"])
                exit_code = run_chat(
                    model=model,
                    input_func=lambda prompt: next(inputs),
                    stdout=stdout,
                    stderr=stderr,
                    workspace_root=workspace,
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("no longer available", stdout.getvalue() + stderr.getvalue())
            # The task ran without the (vanished) skill body.
            self.assertGreaterEqual(len(model.inputs), 1)
            joined = "\n".join(m.content or "" for m in model.inputs[0].messages)
            self.assertNotIn("REFACTOR-BODY", joined)

    def test_explicit_skill_load_failure_does_not_call_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            self._make_skill(workspace, "refactor", body="REFACTOR-BODY")
            model = FakeModelClient([ModelOutput(text="must not run")])
            stdout = io.StringIO()
            stderr = io.StringIO()

            with patch("forestcode.skills.loader.SkillLoader.load_entry", return_value=None):
                inputs = iter(["$refactor 重构", "/exit"])
                exit_code = run_chat(
                    model=model,
                    input_func=lambda prompt: next(inputs),
                    stdout=stdout,
                    stderr=stderr,
                    workspace_root=workspace,
                )

            self.assertEqual(exit_code, 0)
            self.assertIn("Skill could not be loaded: refactor", stderr.getvalue())
            self.assertEqual(model.inputs, [])


if __name__ == "__main__":
    unittest.main()
