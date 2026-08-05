import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forestcode.tools.command import (
    CommandService,
    _build_display,
    _is_dangerous,
    _truncate_tail,
    build_command_content,
)
from forestcode.tools.types import CommandProposal


def _proposal(command: str, cwd: Path, timeout: int = 30) -> CommandProposal:
    return CommandProposal(
        id="cmd_1",
        command=command,
        cwd=cwd,
        timeout=timeout,
        shell_label="test-shell",
        is_dangerous=False,
        display="display",
        status="proposed",
        tool_call_id="call_1",
        created_at="now",
    )


class CommandServiceTest(unittest.TestCase):
    def test_truncate_tail_preserves_short_text(self):
        self.assertEqual(_truncate_tail("a\nb", 10, 100), "a\nb")

    def test_truncate_tail_by_lines(self):
        text = "\n".join(str(index) for index in range(5))

        result = _truncate_tail(text, 2, 100)

        self.assertIn("[... 3 lines omitted ...]", result)
        self.assertTrue(result.endswith("3\n4"))

    def test_truncate_tail_by_bytes(self):
        result = _truncate_tail("abcdef", 10, 3)

        self.assertIn("[... 3 bytes omitted ...]", result)
        self.assertTrue(result.endswith("def"))

    def test_is_dangerous_detects_known_patterns(self):
        for command in ["rm -rf tmp", "sudo make install", "mkfs /dev/sda", "Remove-Item foo -Recurse"]:
            with self.subTest(command=command):
                self.assertTrue(_is_dangerous(command))

    def test_is_dangerous_allows_common_safe_commands(self):
        for command in ["git status", "pytest"]:
            with self.subTest(command=command):
                self.assertFalse(_is_dangerous(command))

    def test_build_display_marks_dangerous_commands(self):
        dangerous = _build_display("rm -rf tmp", Path("."), "bash", 30, True)
        safe = _build_display("git status", Path("."), "bash", 30, False)

        self.assertIn("⚠️", dangerous)
        self.assertNotIn("⚠️", safe)

    def test_execute_success_records_stdout_and_status(self):
        proposal = _proposal("noop", Path("."))
        completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"done", stderr=b"")

        with patch("forestcode.tools.command.subprocess.run", return_value=completed) as run_mock:
            CommandService().execute(proposal)

        run_mock.assert_called_once()
        self.assertEqual(proposal.status, "executed")
        self.assertEqual(proposal.exit_code, 0)
        self.assertEqual(proposal.stdout, "done")
        self.assertIsNotNone(proposal.executed_at)

    def test_execute_nonzero_exit_records_code_without_raising(self):
        proposal = _proposal("noop", Path("."))
        completed = subprocess.CompletedProcess(args=[], returncode=3, stdout=b"", stderr=b"boom")

        with patch("forestcode.tools.command.subprocess.run", return_value=completed):
            CommandService().execute(proposal)

        self.assertEqual(proposal.status, "executed")
        self.assertEqual(proposal.exit_code, 3)
        self.assertEqual(proposal.stderr, "boom")

    def test_execute_timeout_marks_timeout_and_raises(self):
        proposal = _proposal("noop", Path("."), timeout=1)
        timeout_exc = subprocess.TimeoutExpired(cmd="noop", timeout=1, output=b"partial", stderr=b"")

        with patch("forestcode.tools.command.subprocess.run", side_effect=timeout_exc):
            with self.assertRaises(subprocess.TimeoutExpired):
                CommandService().execute(proposal)

        self.assertEqual(proposal.status, "timeout")
        self.assertIn("timed out", proposal.error or "")
        self.assertEqual(proposal.stdout, "partial")

    def test_execute_failure_marks_failed_and_raises(self):
        proposal = _proposal("noop", Path("."))

        with patch("forestcode.tools.command.subprocess.run", side_effect=FileNotFoundError("no shell")):
            with self.assertRaises(FileNotFoundError):
                CommandService().execute(proposal)

        self.assertEqual(proposal.status, "failed")
        self.assertIn("no shell", proposal.error or "")

    def test_execute_integration_through_real_shell(self):
        # 唯一的真实 subprocess 测试：echo 在 PowerShell/cmd/bash/sh/zsh 下都可输出，
        # 不依赖 PATH 上的 python；解析失败时跳过而非误报。
        with tempfile.TemporaryDirectory() as tmp:
            proposal = _proposal("echo forestcode_ok", Path(tmp))
            try:
                CommandService().execute(proposal)
            except (FileNotFoundError, OSError) as exc:
                self.skipTest(f"shell unavailable: {exc}")

        self.assertEqual(proposal.status, "executed")
        self.assertEqual(proposal.exit_code, 0)
        self.assertIn("forestcode_ok", proposal.stdout or "")

    def test_reject_sets_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            proposal = _proposal("git status", Path(tmp))

            CommandService().reject(proposal)

        self.assertEqual(proposal.status, "rejected")

    def test_build_command_content_formats_exit_code_and_empty_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            proposal = _proposal("git status", Path(tmp))
            proposal.status = "executed"
            proposal.stdout = ""
            proposal.exit_code = 3

            content = build_command_content(proposal)

        self.assertIn("(no output)", content)
        self.assertIn("[exit code: 3]", content)

    def test_build_command_content_omits_zero_exit_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            proposal = _proposal("git status", Path(tmp))
            proposal.status = "executed"
            proposal.stdout = "ok"
            proposal.exit_code = 0

            content = build_command_content(proposal)

        self.assertEqual(content, "ok")


if __name__ == "__main__":
    unittest.main()
