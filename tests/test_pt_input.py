import unittest
from pathlib import Path

from forestcode.terminal.pt_input import (
    SlashCompleter,
    _input_body_children,
    echo_fragments,
    status_text,
)
from forestcode.terminal.renderer import FrontendState

try:
    import prompt_toolkit  # noqa: F401

    HAS_PT = True
except ImportError:
    HAS_PT = False


class StatusTextTest(unittest.TestCase):
    def test_none_state_shows_hints_only(self):
        out = status_text(None)
        self.assertIn("Enter", out)
        self.assertNotIn("session:", out)

    def test_full_state(self):
        out = status_text(
            FrontendState(
                workspace_root=Path("/x/rt_project2"),
                session_id="default",
                model_name="deepseek-chat",
            )
        )
        self.assertIn("session:default", out)
        self.assertIn("deepseek-chat", out)
        self.assertIn("rt_project2", out)

    def test_disabled_session(self):
        out = status_text(
            FrontendState(workspace_root=Path("/x/y"), session_id=None, model_name="m")
        )
        self.assertIn("session:disabled", out)


class EchoFragmentsTest(unittest.TestCase):
    def test_single_line_gets_prompt_marker(self):
        frags = echo_fragments("hello")
        self.assertEqual(frags, [("class:prompt", "› "), ("", "hello\n")])

    def test_multiline_only_first_line_has_marker(self):
        frags = echo_fragments("a\nb")
        markers = [f[1] for f in frags if f[0] == "class:prompt"]
        self.assertEqual(markers, ["› ", "  "])
        bodies = [f[1] for f in frags if f[0] == ""]
        self.assertEqual(bodies, ["a\n", "b\n"])


class SlashCompleterTest(unittest.TestCase):
    def _completer(self):
        return SlashCompleter(
            lambda: [("exit", "Exit"), ("compact", "Compact"), ("switch", "Switch")],
            lambda: [
                ("python-review", "Review Python code"),
                ("release-notes", "Write release notes"),
            ],
        )

    def test_prefix_match(self):
        self.assertEqual(self._completer().matches("/co"), [("compact", "Compact")])

    def test_bare_slash_lists_all(self):
        names = [n for n, _ in self._completer().matches("/")]
        self.assertEqual(names, ["exit", "compact", "switch"])

    def test_no_match_for_non_slash(self):
        self.assertEqual(self._completer().matches("hello"), [])

    def test_no_match_after_space(self):
        self.assertEqual(self._completer().matches("/switch s2"), [])

    def test_bare_dollar_lists_skills(self):
        self.assertEqual(
            self._completer().matches("$"),
            [
                ("python-review", "Review Python code"),
                ("release-notes", "Write release notes"),
            ],
        )

    def test_dollar_prefix_filters_skills(self):
        self.assertEqual(
            self._completer().matches("$py"),
            [("python-review", "Review Python code")],
        )

    def test_no_skill_match_after_space(self):
        self.assertEqual(self._completer().matches("$python-review check"), [])

    def test_inline_dollar_has_no_matches(self):
        self.assertEqual(self._completer().matches("use $python-review"), [])


class InputLayoutTest(unittest.TestCase):
    def test_completion_menu_renders_above_input_box(self):
        count = 0

        def rule():
            nonlocal count
            count += 1
            return f"rule-{count}"

        self.assertEqual(
            _input_body_children("menu", rule, "input", "status"),
            ["menu", "rule-1", "input", "rule-2", "status"],
        )


# Note: PromptToolkitInputController._build_application() is intentionally NOT
# constructed here — instantiating a prompt_toolkit Application eagerly opens a
# console output, which raises NoConsoleScreenBufferError under a non-Windows
# console (e.g. git-bash xterm) even though it works in a real terminal. The
# completer's get_completions and the history factory do not need a console.
@unittest.skipUnless(HAS_PT, "prompt_toolkit not installed")
class PtGlueTest(unittest.TestCase):
    def test_get_completions_yields(self):
        from prompt_toolkit.document import Document

        from forestcode.terminal.pt_input import PromptToolkitInputController

        controller = PromptToolkitInputController(
            slash_commands=lambda: [("compact", "Compact")],
            skill_candidates=lambda: [("python-review", "Review Python code")],
        )
        completions = list(controller._completer.get_completions(Document("/co"), None))
        self.assertEqual([c.text for c in completions], ["compact"])

        skill_completions = list(
            controller._completer.get_completions(Document("$py"), None)
        )
        self.assertEqual([c.text for c in skill_completions], ["python-review"])
        self.assertEqual(skill_completions[0].display_text, "$python-review")

    def test_pt_completer_adapter_is_a_real_completer(self):
        # Regression: Buffer calls get_completions_async, which only the
        # Completer base class provides. A bare SlashCompleter lacks it and
        # crashes the event loop at completion time.
        from prompt_toolkit.completion import Completer
        from prompt_toolkit.document import Document

        from forestcode.terminal.pt_input import SlashCompleter, _pt_completer

        adapted = _pt_completer(SlashCompleter(lambda: [("compact", "Compact")]))
        self.assertIsInstance(adapted, Completer)
        self.assertTrue(hasattr(adapted, "get_completions_async"))
        self.assertEqual(
            [c.text for c in adapted.get_completions(Document("/co"), None)],
            ["compact"],
        )

    def test_pt_completer_none_passthrough(self):
        from forestcode.terminal.pt_input import _pt_completer

        self.assertIsNone(_pt_completer(None))

    def test_build_history_in_memory_when_no_session(self):
        from forestcode.terminal.pt_input import build_history

        h = build_history(session_enabled=False, workspace_root=Path("."))
        self.assertEqual(type(h).__name__, "InMemoryHistory")

    def test_build_history_file_when_session(self):
        import tempfile

        from forestcode.terminal.pt_input import build_history

        with tempfile.TemporaryDirectory() as tmp:
            h = build_history(session_enabled=True, workspace_root=Path(tmp))
            self.assertEqual(type(h).__name__, "FileHistory")
            self.assertTrue((Path(tmp) / ".forestcode").exists())

    def test_no_history_forces_in_memory(self):
        import tempfile

        from forestcode.terminal.pt_input import build_history

        with tempfile.TemporaryDirectory() as tmp:
            h = build_history(session_enabled=True, workspace_root=Path(tmp), no_history=True)
            self.assertEqual(type(h).__name__, "InMemoryHistory")


@unittest.skipUnless(HAS_PT, "prompt_toolkit not installed")
class ApprovalOptionsTest(unittest.TestCase):
    """Tests for _build_approval_options — pure, no Application instantiation."""

    def _req(self, *, is_dangerous=False):
        from forestcode.tools import ApprovalRequest
        return ApprovalRequest(
            kind="patch", tool_name="edit_file", preview="diff",
            operation="edit", path="a.txt", is_dangerous=is_dangerous,
        )

    def test_approval_app_shows_always_when_not_dangerous(self):
        from forestcode.terminal.pt_input import _build_approval_options
        opts = _build_approval_options(self._req(is_dangerous=False), allow_always=True)
        choices = [c for c, _ in opts]
        self.assertIn("always", choices)

    def test_approval_app_hides_always_when_dangerous(self):
        from forestcode.terminal.pt_input import _build_approval_options
        opts = _build_approval_options(self._req(is_dangerous=True), allow_always=True)
        choices = [c for c, _ in opts]
        self.assertNotIn("always", choices)

    def test_approval_app_hides_always_when_allow_always_false(self):
        from forestcode.terminal.pt_input import _build_approval_options
        opts = _build_approval_options(self._req(is_dangerous=False), allow_always=False)
        choices = [c for c, _ in opts]
        self.assertNotIn("always", choices)


if __name__ == "__main__":
    unittest.main()
