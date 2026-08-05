import os
import unittest
from unittest.mock import patch

from forestcode.tools.command import decode_output
from forestcode.tools.shell import build_argv, describe_shell, resolve_shell


class ShellResolutionTest(unittest.TestCase):
    def test_default_windows_uses_powershell(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.platform", "win32"):
            self.assertEqual(
                resolve_shell(),
                ("PowerShell", "powershell.exe", ["-NoProfile", "-NonInteractive", "-Command"]),
            )
            self.assertEqual(
                build_argv("Write-Output ok"),
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", "Write-Output ok"],
            )

    def test_default_non_windows_uses_shell_env(self):
        with patch.dict(os.environ, {"SHELL": "/usr/local/bin/zsh"}, clear=True), patch("sys.platform", "linux"):
            self.assertEqual(resolve_shell(), ("zsh", "/usr/local/bin/zsh", ["-c"]))
            self.assertEqual(build_argv("echo ok"), ["/usr/local/bin/zsh", "-c", "echo ok"])

    def test_default_non_windows_falls_back_to_sh(self):
        with patch.dict(os.environ, {}, clear=True), patch("sys.platform", "linux"):
            self.assertEqual(resolve_shell(), ("sh", "/bin/sh", ["-c"]))

    def test_shell_override_cmd(self):
        with patch.dict(os.environ, {"FORESTCODE_SHELL": "cmd"}, clear=True):
            self.assertEqual(resolve_shell(), ("cmd", "cmd.exe", ["/c"]))
            self.assertEqual(build_argv("dir"), ["cmd.exe", "/c", "dir"])

    def test_shell_override_pwsh(self):
        with patch.dict(os.environ, {"FORESTCODE_SHELL": "pwsh"}, clear=True):
            self.assertEqual(resolve_shell(), ("pwsh", "pwsh", ["-NoProfile", "-NonInteractive", "-Command"]))

    def test_shell_override_bash(self):
        with patch.dict(os.environ, {"FORESTCODE_SHELL": "bash"}, clear=True):
            self.assertEqual(resolve_shell(), ("bash", "bash", ["-c"]))

    def test_shell_override_path_preserves_original_case(self):
        raw = r"C:\Program Files\Git\bin\bash.exe"
        with patch.dict(os.environ, {"FORESTCODE_SHELL": raw}, clear=True):
            self.assertEqual(resolve_shell(), ("bash.exe", raw, ["-c"]))

    def test_describe_shell_returns_label(self):
        with patch.dict(os.environ, {"FORESTCODE_SHELL": "bash"}, clear=True):
            self.assertEqual(describe_shell(), "bash")

    def test_decode_output_utf8(self):
        self.assertEqual(decode_output("hello".encode("utf-8")), "hello")

    def test_decode_output_non_utf8_does_not_raise(self):
        decoded = decode_output("中文".encode("gbk"))

        self.assertIsInstance(decoded, str)
        self.assertTrue(decoded)

    def test_decode_output_empty_bytes(self):
        self.assertEqual(decode_output(b""), "")


if __name__ == "__main__":
    unittest.main()
