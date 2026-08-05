import io
import unittest

from forestcode.terminal.activity import (
    BRAILLE_FRAMES,
    FOREST_VERBS,
    elapsed_label,
    shimmer_index,
    spinner_frame,
    spinner_verb,
)

try:
    import rich  # noqa: F401

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class SpinnerMathTest(unittest.TestCase):
    def test_spinner_frame_cycles(self):
        self.assertEqual(spinner_frame(0.0), BRAILLE_FRAMES[0])
        self.assertEqual(spinner_frame(0.08), BRAILLE_FRAMES[1])
        # wraps around
        self.assertEqual(spinner_frame(0.08 * len(BRAILLE_FRAMES)), BRAILLE_FRAMES[0])

    def test_spinner_verb_rotates(self):
        self.assertEqual(spinner_verb(0.0), FOREST_VERBS[0])
        self.assertEqual(spinner_verb(2.0), FOREST_VERBS[1])
        self.assertEqual(spinner_verb(2.0 * len(FOREST_VERBS)), FOREST_VERBS[0])

    def test_elapsed_label(self):
        self.assertEqual(elapsed_label(0.4), "(0s)")
        self.assertEqual(elapsed_label(4.9), "(4s)")

    def test_shimmer_index(self):
        self.assertEqual(shimmer_index(0, 5.0), -1)
        self.assertEqual(shimmer_index(4, 0.0), 0)
        self.assertTrue(0 <= shimmer_index(4, 1.7) < 4)


@unittest.skipUnless(HAS_RICH, "rich not installed")
class TurnActivityTest(unittest.TestCase):
    def _console(self):
        from rich.console import Console

        buf = io.StringIO()
        return Console(file=buf, force_terminal=True, width=80), buf

    def test_spinner_renderable_produces_output(self):
        from forestcode.terminal.activity import SpinnerRenderable
        from forestcode.terminal.renderer import FOREST_THEME

        console, buf = self._console()
        clock = iter([0.0, 0.0]).__next__  # start, then render (elapsed 0.0)
        spinner = SpinnerRenderable(FOREST_THEME, clock=clock)
        console.print(spinner)
        out = buf.getvalue()
        # The elapsed label is ASCII so it survives any console encoding; the
        # verb itself renders as styled spans (CJK may be substituted by the
        # console encoding, which is acceptable for the flourish).
        self.assertIn("(0s)", out)

    def test_print_above_without_live_prints_directly(self):
        from forestcode.terminal.activity import TurnActivity
        from forestcode.terminal.renderer import FOREST_THEME

        console, buf = self._console()
        activity = TurnActivity(console, FOREST_THEME)
        self.assertFalse(activity.active)
        from rich.text import Text

        activity.print_above(Text("committed line"))
        self.assertIn("committed line", buf.getvalue())

    def test_tool_lines(self):
        from forestcode.terminal.activity import tool_active_line, tool_finished_line
        from forestcode.terminal.renderer import FOREST_THEME

        console, buf = self._console()
        console.print(tool_active_line("read_file", {"path": "src/cli.py"}, FOREST_THEME))
        console.print(
            tool_finished_line("read_file", True, {"lines": 42}, FOREST_THEME, args="src/cli.py")
        )
        out = buf.getvalue()
        self.assertIn("●", out)
        self.assertIn("read_file", out)
        self.assertIn("src/cli.py", out)
        self.assertIn("✓", out)
        self.assertIn("42 lines", out)


@unittest.skipUnless(HAS_RICH, "rich not installed")
class RichRendererToolEventTest(unittest.TestCase):
    def _renderer(self):
        from forestcode.terminal.rich_renderer import RichRenderer

        buf = io.StringIO()
        return RichRenderer(buf, buf), buf

    def test_tool_started_and_finished_via_on_event(self):
        from forestcode.core.types import RunEvent

        renderer, buf = self._renderer()
        renderer.on_event(
            RunEvent(
                "tool_call_started",
                {"tool_call_id": "c1", "tool_name": "read_file", "arguments": {"path": "a.py"}},
            )
        )
        renderer.on_event(
            RunEvent(
                "tool_call_finished",
                {"tool_call_id": "c1", "tool_name": "read_file", "ok": True, "data": {"lines": 7}},
            )
        )
        out = buf.getvalue()
        self.assertIn("read_file", out)
        self.assertIn("a.py", out)
        self.assertIn("✓", out)
        self.assertIn("7 lines", out)

    def test_command_output_rendered(self):
        from forestcode.core.types import RunEvent

        renderer, buf = self._renderer()
        renderer.on_event(
            RunEvent(
                "tool_call_started",
                {"tool_call_id": "c1", "tool_name": "run_command", "arguments": {"command": "ls"}},
            )
        )
        renderer.on_event(
            RunEvent(
                "tool_call_finished",
                {
                    "tool_call_id": "c1",
                    "tool_name": "run_command",
                    "ok": True,
                    "data": {"exit_code": 0, "stdout": "file_a\nfile_b", "stderr": ""},
                },
            )
        )
        out = buf.getvalue()
        self.assertIn("exit 0", out)
        self.assertIn("file_a", out)
        self.assertIn("│", out)


if __name__ == "__main__":
    unittest.main()
