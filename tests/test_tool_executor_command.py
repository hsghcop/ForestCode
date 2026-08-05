import subprocess
import tempfile
import unittest
from pathlib import Path

from forestcode.core import ToolCall, ToolExecutor
from forestcode.core.run_state import RunState
from forestcode.tools import ToolRuntimeServices, ToolRegistry
from forestcode.tools.command import create_run_command_tool


class FakeCommandService:
    def __init__(
        self,
        *,
        stdout: str = "ok",
        stderr: str = "",
        exit_code: int = 0,
        timeout: bool = False,
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.timeout = timeout
        self.executed = []
        self.rejected = []

    def execute(self, proposal, *, abort=None) -> None:
        self.executed.append(proposal)
        proposal.stdout = self.stdout
        proposal.stderr = self.stderr
        if self.timeout:
            proposal.status = "timeout"
            proposal.error = f"Command timed out after {proposal.timeout}s"
            raise subprocess.TimeoutExpired(cmd=proposal.command, timeout=proposal.timeout)
        proposal.exit_code = self.exit_code
        proposal.status = "executed"

    def reject(self, proposal) -> None:
        self.rejected.append(proposal)
        proposal.status = "rejected"


def _executor(root: Path, confirm, command_service=None, enable_command_tools: bool = True) -> ToolExecutor:
    registry = ToolRegistry([create_run_command_tool()])
    runtime = ToolRuntimeServices(confirm=confirm, command_service=command_service)
    return ToolExecutor(
        registry,
        workspace_root=root,
        runtime=runtime,
        enable_command_tools=enable_command_tools,
    )


class ToolExecutorCommandTest(unittest.TestCase):
    def test_confirm_true_zero_exit_returns_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = FakeCommandService(stdout="done", exit_code=0)
            executor = _executor(Path(tmp), lambda _request: True, service)

            result = executor.execute(
                ToolCall(id="cmd", name="run_command", arguments={"command": "git status"}),
                RunState.start("test"),
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.content, "done")
        self.assertEqual(result.data["status"], "executed")
        self.assertEqual(result.data["exit_code"], 0)

    def test_confirm_true_nonzero_exit_returns_error_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = FakeCommandService(stdout="", stderr="bad", exit_code=2)
            executor = _executor(Path(tmp), lambda _request: True, service)

            result = executor.execute(
                ToolCall(id="cmd", name="run_command", arguments={"command": "false"}),
                RunState.start("test"),
            )

        self.assertFalse(result.ok)
        self.assertIn("[stderr]\nbad", result.content)
        self.assertIn("[exit code: 2]", result.content)

    def test_confirm_true_timeout_returns_timeout_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = FakeCommandService(stdout="partial", timeout=True)
            executor = _executor(Path(tmp), lambda _request: True, service)

            result = executor.execute(
                ToolCall(id="cmd", name="run_command", arguments={"command": "sleep", "timeout": 1}),
                RunState.start("test"),
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.data["status"], "timeout")
        self.assertIn("partial", result.content)
        self.assertIn("[timed out after 1s]", result.content)

    def test_confirm_false_rejects_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = FakeCommandService()
            executor = _executor(Path(tmp), lambda _request: False, service)

            result = executor.execute(
                ToolCall(id="cmd", name="run_command", arguments={"command": "git status"}),
                RunState.start("test"),
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "User declined command.")
        self.assertEqual(result.data["status"], "rejected")
        self.assertEqual(len(service.rejected), 1)

    def test_missing_command_service_returns_permission_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = _executor(Path(tmp), lambda _request: True)

            result = executor.execute(
                ToolCall(id="cmd", name="run_command", arguments={"command": "git status"}),
                RunState.start("test"),
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.error, "Permission required")
        self.assertEqual(result.data["permission"], "ask")

    def test_command_disabled_denies_without_executing(self):
        with tempfile.TemporaryDirectory() as tmp:
            service = FakeCommandService()
            executor = _executor(
                Path(tmp), lambda _request: True, service, enable_command_tools=False
            )

            result = executor.execute(
                ToolCall(id="cmd", name="run_command", arguments={"command": "git status"}),
                RunState.start("test"),
            )

        self.assertFalse(result.ok)
        self.assertEqual(result.data["permission"], "deny")
        self.assertIn("not enabled", result.error)
        self.assertEqual(service.executed, [])
        self.assertEqual(service.rejected, [])

    def test_dangerous_command_is_marked_but_can_be_approved(self):
        captured = {}

        def confirm(request):
            captured["preview"] = request.preview
            return True

        with tempfile.TemporaryDirectory() as tmp:
            service = FakeCommandService()
            executor = _executor(Path(tmp), confirm, service)

            result = executor.execute(
                ToolCall(id="cmd", name="run_command", arguments={"command": "rm -rf tmp"}),
                RunState.start("test"),
            )

        self.assertTrue(result.ok)
        self.assertIn("⚠️", captured["preview"])
        self.assertTrue(service.executed[0].is_dangerous)


if __name__ == "__main__":
    unittest.main()
