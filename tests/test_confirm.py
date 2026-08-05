import io
import unittest

from forestcode.terminal.confirm import CONFIRM_DIFF_MAX_LINES, ConfirmationController, allow_key
from forestcode.terminal.input import StdinInputController
from forestcode.terminal.renderer import PlainRenderer
from forestcode.tools import ApprovalRequest


def _patch(path="a.txt", added=1, removed=1, preview="diff"):
    return ApprovalRequest(
        kind="patch", tool_name="edit_file", preview=preview,
        operation="edit", path=path, added=added, removed=removed,
    )


def _command(command="npm test", cwd="/repo", preview="npm test"):
    return ApprovalRequest(
        kind="command", tool_name="run_command", preview=preview,
        command=command, cwd=cwd,
    )


class AllowKeyTest(unittest.TestCase):
    def test_patch_key_is_path(self):
        self.assertEqual(allow_key(_patch(path="x.py")), ("patch", "x.py"))

    def test_command_key_includes_cwd_and_normalized_command(self):
        self.assertEqual(
            allow_key(_command(command="npm   test", cwd="/repo")),
            ("command", "/repo", "npm test"),
        )

    def test_command_same_command_different_cwd_differs(self):
        self.assertNotEqual(
            allow_key(_command(cwd="/a")), allow_key(_command(cwd="/b"))
        )


class ConfirmTest(unittest.TestCase):
    def _controller(self, choices, *, allow_always=True):
        buf = io.StringIO()
        renderer = PlainRenderer(buf, buf)
        it = iter(choices)
        controller = ConfirmationController(
            renderer,
            StdinInputController(lambda _p: ""),
            chooser=lambda _r: next(it),
            allow_always=allow_always,
        )
        return controller, buf

    def test_yes(self):
        c, _ = self._controller(["yes"])
        self.assertTrue(c.confirm(_patch()))

    def test_no(self):
        c, _ = self._controller(["no"])
        self.assertFalse(c.confirm(_patch()))

    def test_always_adds_to_allowlist_and_skips_second_prompt(self):
        # chooser yields "always" once; a second confirm of the same resource
        # must NOT consult the chooser (StopIteration would fire if it did).
        c, _ = self._controller(["always"])
        self.assertTrue(c.confirm(_patch(path="a.txt")))
        self.assertTrue(c.confirm(_patch(path="a.txt")))

    def test_always_is_resource_scoped(self):
        c, _ = self._controller(["always", "no"])
        self.assertTrue(c.confirm(_patch(path="a.txt")))
        # different resource still prompts -> uses the second chooser value
        self.assertFalse(c.confirm(_patch(path="b.txt")))

    def test_eof_rejects(self):
        buf = io.StringIO()
        renderer = PlainRenderer(buf, buf)

        def boom(_r):
            raise EOFError

        c = ConfirmationController(renderer, StdinInputController(lambda _p: ""), chooser=boom)
        self.assertFalse(c.confirm(_command()))

    def test_diff_truncation(self):
        diff = "\n".join(f"+line{i}" for i in range(CONFIRM_DIFF_MAX_LINES + 10))
        c, buf = self._controller(["no"])
        c.confirm(_patch(preview=diff))
        out = buf.getvalue()
        self.assertIn("more lines", out)


class TextChooserTest(unittest.TestCase):
    def _controller(self, answer, *, allow_always=True):
        buf = io.StringIO()
        renderer = PlainRenderer(buf, buf)
        controller = ConfirmationController(
            renderer, StdinInputController(lambda _p: answer), allow_always=allow_always
        )
        return controller

    def test_y_yes(self):
        self.assertTrue(self._controller("y").confirm(_patch()))
        self.assertTrue(self._controller("yes").confirm(_patch()))

    def test_n_blank_garbage(self):
        for ans in ("n", "", "garbage"):
            self.assertFalse(self._controller(ans).confirm(_patch()))

    def test_a_means_always(self):
        c = self._controller("a")
        self.assertTrue(c.confirm(_patch(path="a.txt")))
        # allowlisted now
        self.assertIn(("patch", "a.txt"), c._allowlist)

    def test_a_disabled_when_allow_always_false(self):
        c = self._controller("a", allow_always=False)
        self.assertFalse(c.confirm(_patch()))

    def test_dangerous_command_no_always_in_text_chooser(self):
        # is_dangerous=True: even with allow_always=True, 'a' must not grant always
        dangerous = ApprovalRequest(
            kind="command", tool_name="run_command", preview="rm -rf /",
            command="rm -rf /", cwd="/", is_dangerous=True,
        )
        buf = io.StringIO()
        renderer = PlainRenderer(buf, buf)
        controller = ConfirmationController(
            renderer, StdinInputController(lambda _p: "a"), allow_always=True
        )
        # 'a' input on a dangerous command must fall through to "no"
        self.assertFalse(controller.confirm(dangerous))
        # and the prompt must not contain [a]
        self.assertNotIn("[a]", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
