import io
import unittest
from pathlib import Path

try:
    import rich  # noqa: F401

    HAS_RICH = True
except ImportError:
    HAS_RICH = False

from forestcode.terminal.renderer import FrontendState


@unittest.skipUnless(HAS_RICH, "rich not installed")
class RichRendererTest(unittest.TestCase):
    def _renderer(self):
        from forestcode.terminal.rich_renderer import RichRenderer

        buf = io.StringIO()
        return RichRenderer(buf, buf), buf

    def test_skill_activation_uses_live_safe_rich_output(self):
        from forestcode.core.types import RunEvent

        renderer, output = self._renderer()
        renderer.on_event(RunEvent("skill_activated", {"name": "python-review"}))
        self.assertIn("Skill> 已加载 python-review", output.getvalue())

    def test_welcome_renders_logo_and_info(self):
        renderer, buf = self._renderer()
        renderer.render_welcome(
            FrontendState(workspace_root=Path("."), session_id="default", model_name="fake")
        )
        out = buf.getvalue()
        self.assertIn("Session", out)
        self.assertIn("fake", out)
        self.assertTrue(out.strip())

    def test_assistant_markdown_emits_badge(self):
        renderer, buf = self._renderer()
        renderer.render_assistant_text("# Title\n\n- item")
        out = buf.getvalue()
        self.assertIn("●", out)  # the AI voice badge ●
        self.assertIn("Title", out)

    def test_reminder_emits_glyph_and_message(self):
        renderer, buf = self._renderer()
        renderer.render_memory_status("saved 1 fact")
        out = buf.getvalue()
        self.assertIn("✎", out)  # memory glyph ✎
        self.assertIn("memory", out)
        self.assertIn("saved 1 fact", out)

    def test_empty_assistant_text_skipped(self):
        renderer, buf = self._renderer()
        renderer.render_assistant_text("")
        self.assertEqual(buf.getvalue(), "")

    def test_gradient_helpers(self):
        from forestcode.terminal.rich_renderer import _hex_rgb, _lerp_hex

        self.assertEqual(_hex_rgb("bold #2E8B57"), (46, 139, 87))
        self.assertEqual(_hex_rgb("#7FB069"), (127, 176, 105))
        self.assertEqual(_hex_rgb("not-a-color"), (255, 255, 255))
        self.assertEqual(_lerp_hex((0, 0, 0), (10, 20, 30), 0.0), "#000000")
        self.assertEqual(_lerp_hex((0, 0, 0), (10, 20, 30), 1.0), "#0A141E")


if __name__ == "__main__":
    unittest.main()
