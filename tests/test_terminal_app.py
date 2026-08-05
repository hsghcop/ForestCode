"""Tests for the new ForestCode interactive frontend (terminal app stack).

Covers the §19.2 scenarios that exercise the renderer / bridge / app wiring:
normal reply, tool display, empty stop, slash dispatch, session switch,
patch confirm/reject, command gating, recorded dedup, and the EOF / Ctrl+C
confirmation semantics (§8.2).
"""

import io
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from forestcode.config import AgentRuntimeConfig
from forestcode.core import AssistantTurn, FakeModelClient, ModelOutput, ToolCall
from forestcode.models import ModelAdapterError
from forestcode.terminal.app import ForestCodeCliApp
from forestcode.terminal.bridge import BackendBridge, BackendBridgeConfig
from forestcode.terminal.confirm import ConfirmationController
from forestcode.terminal.input import StdinInputController
from forestcode.terminal.renderer import FrontendState, PlainRenderer, build_renderer

_EOF = object()
_INT = object()


def _input_func(inputs):
    it = iter(inputs)

    def read(_prompt):
        value = next(it)
        if value is _EOF:
            raise EOFError
        if value is _INT:
            raise KeyboardInterrupt
        return value

    return read


def _build_app(root, model, inputs, *, session_id="default", agent=None):
    out, err = io.StringIO(), io.StringIO()
    renderer = PlainRenderer(out, err)
    input_controller = StdinInputController(_input_func(inputs))
    confirm = ConfirmationController(renderer, input_controller)
    agent = agent or AgentRuntimeConfig()
    bridge = BackendBridge(
        BackendBridgeConfig(
            workspace_root=root,
            session_id=session_id,
            agent=agent,
            model=model,
            renderer=renderer,
            input_controller=input_controller,
            confirmation_controller=confirm,
            stdout=out,
            stderr=err,
        )
    )
    state = FrontendState(workspace_root=root, session_id=session_id, model_name="fake")
    app = ForestCodeCliApp(bridge, renderer, input_controller, state)
    return app, out, err, model


class TerminalAppTest(unittest.TestCase):
    def test_normal_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, out, err, model = _build_app(
                Path(tmp), FakeModelClient([ModelOutput(text="你好")]), ["hello", "/exit"]
            )
            rc = app.run()
        self.assertEqual(rc, 0)
        self.assertIn("Assistant> 你好", out.getvalue())
        self.assertEqual(out.getvalue().count("Assistant> 你好"), 1)
        self.assertEqual(model.inputs[0].messages[0].content, "hello")
        self.assertEqual(err.getvalue(), "")

    def test_welcome_shows_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, out, _err, _model = _build_app(Path(tmp), FakeModelClient([]), ["/exit"])
            app.run()
        value = out.getvalue()
        self.assertIn("ForestCode", value)
        self.assertIn("Session> default", value)
        self.assertIn("Model> fake", value)

    def test_tool_display(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("hello", encoding="utf-8")
            model = FakeModelClient(
                [
                    ModelOutput(tool_calls=[ToolCall(id="c1", name="list_files", arguments={"path": ".", "max_entries": 10})]),
                    ModelOutput(text="看完了"),
                ]
            )
            app, out, err, _ = _build_app(root, model, ["list files", "/exit"])
            rc = app.run()
        self.assertEqual(rc, 0)
        self.assertIn("Tool> list_files", out.getvalue())
        self.assertIn("Tool> list_files ok", out.getvalue())
        self.assertIn("Assistant> 看完了", out.getvalue())
        self.assertEqual(err.getvalue(), "")

    def test_empty_stop_no_blank_assistant_no_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient([ModelOutput(AssistantTurn(text="", finish_reason="stop"))])
            app, out, err, _ = _build_app(Path(tmp), model, ["go", "/exit"], session_id="s1")
            rc = app.run()
        self.assertEqual(rc, 0)
        self.assertNotIn("Assistant>", out.getvalue())  # no blank Assistant> line
        self.assertNotIn("Memory> recorded", out.getvalue())
        self.assertNotIn("Agent error", err.getvalue())

    def test_recorded_message_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient([ModelOutput(text="done")])
            app, out, _err, _ = _build_app(Path(tmp), model, ["hi", "/exit"], session_id="s1")
            app.run()
        self.assertIn("Memory> recorded .forestcode/sessions/s1.jsonl", out.getvalue())

    def test_unknown_slash_no_model_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient([])
            app, _out, err, _ = _build_app(Path(tmp), model, ["/cmpact", "/exit"])
            rc = app.run()
        self.assertEqual(rc, 0)
        self.assertIn("Unknown command: /cmpact", err.getvalue())
        self.assertIn("Available:", err.getvalue())
        self.assertEqual(model.inputs, [])

    def test_path_like_slash_falls_through_to_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient([ModelOutput(text="done")])
            app, _out, err, _ = _build_app(Path(tmp), model, ["/path/to/file", "/exit"])
            rc = app.run()
        self.assertEqual(rc, 0)
        self.assertEqual(model.inputs[0].messages[0].content, "/path/to/file")
        self.assertEqual(err.getvalue(), "")

    def test_switch_from_no_session_enables_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = FakeModelClient([ModelOutput(text="done")])
            app, out, err, _ = _build_app(root, model, ["/switch s2", "hello", "/exit"], session_id=None)
            rc = app.run()
            self.assertEqual(rc, 0)
            self.assertIn("Session> s2", out.getvalue())
            self.assertTrue((root / ".forestcode" / "sessions" / "s2.jsonl").exists())
            self.assertIn("Memory> recorded .forestcode/sessions/s2.jsonl", out.getvalue())
            self.assertEqual(err.getvalue(), "")

    def test_model_error_renders_once_and_exits(self):
        class Failing:
            def complete(self, _model_input, *, abort=None):
                raise ModelAdapterError("bad config")

        with tempfile.TemporaryDirectory() as tmp:
            app, _out, err, _ = _build_app(Path(tmp), Failing(), ["hello"])
            rc = app.run()
        self.assertEqual(rc, 1)
        self.assertEqual(err.getvalue().count("Model error: bad config"), 1)

    def test_edit_file_confirm_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("old\n", encoding="utf-8")
            model = FakeModelClient(
                [
                    ModelOutput(tool_calls=[ToolCall(id="r", name="read_file", arguments={"path": "a.txt", "offset": 0, "limit": 20000})]),
                    ModelOutput(tool_calls=[ToolCall(id="e", name="edit_file", arguments={"path": "a.txt", "old_text": "old", "new_text": "new"})]),
                    ModelOutput(text="done"),
                ]
            )
            app, out, _err, _ = _build_app(root, model, ["edit it", "y", "/exit"], session_id="s1")
            rc = app.run()
            self.assertEqual(rc, 0)
            self.assertIn("Patch> edit_file requires approval", out.getvalue())
            self.assertIn("Tool> edit_file ok", out.getvalue())
            self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "new\n")

    def test_edit_file_reject_keeps_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("old\n", encoding="utf-8")
            model = FakeModelClient(
                [
                    ModelOutput(tool_calls=[ToolCall(id="r", name="read_file", arguments={"path": "a.txt", "offset": 0, "limit": 20000})]),
                    ModelOutput(tool_calls=[ToolCall(id="e", name="edit_file", arguments={"path": "a.txt", "old_text": "old", "new_text": "new"})]),
                    ModelOutput(text="ok"),
                ]
            )
            app, out, _err, _ = _build_app(root, model, ["edit it", "n", "/exit"], session_id="s1")
            rc = app.run()
            self.assertEqual(rc, 0)
            self.assertIn("Tool> edit_file error", out.getvalue())
            self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "old\n")

    def test_edit_file_eof_confirmation_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("old\n", encoding="utf-8")
            model = FakeModelClient(
                [
                    ModelOutput(tool_calls=[ToolCall(id="r", name="read_file", arguments={"path": "a.txt", "offset": 0, "limit": 20000})]),
                    ModelOutput(tool_calls=[ToolCall(id="e", name="edit_file", arguments={"path": "a.txt", "old_text": "old", "new_text": "new"})]),
                    ModelOutput(text="ok"),
                ]
            )
            app, out, _err, _ = _build_app(root, model, ["edit it", _EOF, "/exit"], session_id="s1")
            rc = app.run()
            self.assertEqual(rc, 0)
            self.assertIn("Tool> edit_file error", out.getvalue())
            self.assertEqual((root / "a.txt").read_text(encoding="utf-8"), "old\n")

    def test_run_command_denied_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient(
                [
                    ModelOutput(tool_calls=[ToolCall(id="c", name="run_command", arguments={"command": "echo hi", "timeout": 30})]),
                    ModelOutput(text="done"),
                ]
            )
            app, out, _err, _ = _build_app(Path(tmp), model, ["run", "/exit"], session_id="s1")
            rc = app.run()
        self.assertEqual(rc, 0)
        self.assertIn("Tool> run_command error", out.getvalue())
        self.assertNotIn("Command> requires approval", out.getvalue())

    def test_run_command_preview_and_reject_when_enabled(self):
        agent = AgentRuntimeConfig()
        agent = replace(agent, features=replace(agent.features, enable_command_tools=True))
        with tempfile.TemporaryDirectory() as tmp:
            model = FakeModelClient(
                [
                    ModelOutput(tool_calls=[ToolCall(id="c", name="run_command", arguments={"command": "echo hi", "timeout": 30})]),
                    ModelOutput(text="done"),
                ]
            )
            app, out, _err, _ = _build_app(Path(tmp), model, ["run", "n", "/exit"], session_id="s1", agent=agent)
            rc = app.run()
        self.assertEqual(rc, 0)
        self.assertIn("Command> requires approval", out.getvalue())
        self.assertIn("Tool> run_command error", out.getvalue())

    def test_ctrl_c_at_prompt_returns_130(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _out, err, _ = _build_app(Path(tmp), FakeModelClient([]), [_INT])
            rc = app.run()
        self.assertEqual(rc, 130)
        self.assertIn("Interrupted.", err.getvalue())

    def test_eof_at_prompt_exits_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            app, _out, _err, _ = _build_app(Path(tmp), FakeModelClient([]), [_EOF])
            rc = app.run()
        self.assertEqual(rc, 0)


class BuildRendererTest(unittest.TestCase):
    def test_non_tty_is_plain(self):
        renderer = build_renderer(io.StringIO(), io.StringIO())
        self.assertIsInstance(renderer, PlainRenderer)

    def test_no_color_env_forces_plain(self):
        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            renderer = build_renderer(FakeTTY(), FakeTTY())
        self.assertIsInstance(renderer, PlainRenderer)


if __name__ == "__main__":
    unittest.main()
