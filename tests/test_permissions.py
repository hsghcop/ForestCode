import unittest

from forestcode.tools import PermissionManager, ToolDefinition


def _command_tool() -> ToolDefinition:
    return ToolDefinition(
        name="run_command",
        description="Run a shell command.",
        input_schema={"type": "object"},
        runner=lambda _context: "",
        risk_level="command",
        is_read_only=False,
    )


def _read_tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_file",
        description="Read a file.",
        input_schema={"type": "object"},
        runner=lambda _context: "",
    )


class PermissionManagerCommandGateTest(unittest.TestCase):
    def test_command_denied_when_disabled(self):
        decision = PermissionManager(enable_command_tools=False).decide(_command_tool(), [])
        self.assertEqual(decision.behavior, "deny")
        self.assertIn("not enabled", decision.reason)

    def test_command_not_denied_by_gate_when_enabled(self):
        # When enabled, the command gate does not deny; command tools fall through to ask.
        decision = PermissionManager(enable_command_tools=True).decide(_command_tool(), [])
        self.assertEqual(decision.behavior, "ask")

    def test_default_disables_command_tools(self):
        decision = PermissionManager().decide(_command_tool(), [])
        self.assertEqual(decision.behavior, "deny")

    def test_read_only_tool_unaffected(self):
        decision = PermissionManager(enable_command_tools=False).decide(_read_tool(), [])
        self.assertEqual(decision.behavior, "allow")


if __name__ == "__main__":
    unittest.main()
