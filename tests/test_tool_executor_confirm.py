import tempfile
import unittest
from pathlib import Path

from forestcode.core import ToolCall, ToolExecutor
from forestcode.core.run_state import RunState
from forestcode.tools import (
    PatchService,
    PathAccess,
    ReadStateStore,
    ToolDefinition,
    ToolRegistry,
    ToolRuntimeServices,
    create_builtin_tool_registry,
)


class ToolExecutorConfirmTest(unittest.TestCase):
    def _executor(self, root: Path, confirm=None, runtime_internal_dirs=frozenset()):
        store = ReadStateStore()
        patch_service = PatchService(read_state_store=store)
        runtime = ToolRuntimeServices(
            read_state_store=store,
            patch_service=patch_service,
            confirm=confirm,
        )
        return ToolExecutor(
            create_builtin_tool_registry(),
            workspace_root=root,
            runtime_internal_dirs=runtime_internal_dirs,
            runtime=runtime,
        )

    def _read(self, executor: ToolExecutor, path: str):
        return executor.execute(
            ToolCall(id="read", name="read_file", arguments={"path": path, "offset": 0, "limit": 20000}),
            RunState.start("test"),
        )

    def test_ask_confirm_true_applies_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("old\n", encoding="utf-8")
            executor = self._executor(root, confirm=lambda _request: True)
            self._read(executor, "a.txt")

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "a.txt", "old_text": "old", "new_text": "new"},
                ),
                RunState.start("test"),
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["status"], "applied")
            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")

    def test_ask_confirm_false_rejects_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("old\n", encoding="utf-8")
            executor = self._executor(root, confirm=lambda _request: False)
            self._read(executor, "a.txt")

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "a.txt", "old_text": "old", "new_text": "new"},
                ),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("User declined", result.error or "")
            self.assertEqual(result.data["status"], "rejected")
            self.assertEqual(target.read_text(encoding="utf-8"), "old\n")

    def test_ask_missing_runtime_returns_permission_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("old\n", encoding="utf-8")
            executor = ToolExecutor(create_builtin_tool_registry(), workspace_root=root)

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "a.txt", "old_text": "old", "new_text": "new"},
                ),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error, "Permission required")
            self.assertEqual(result.data["permission"], "ask")

    def test_allow_tool_does_not_confirm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("hello\n", encoding="utf-8")
            calls = []
            executor = self._executor(root, confirm=lambda request: calls.append((request.tool_name, request.preview)) or True)

            result = executor.execute(
                ToolCall(id="read", name="read_file", arguments={"path": "a.txt", "offset": 0, "limit": 20000}),
                RunState.start("test"),
            )

            self.assertTrue(result.ok)
            self.assertEqual(calls, [])

    def test_write_outside_workspace_denies_without_confirm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            outside = Path(tmp) / "outside.txt"
            calls = []
            executor = self._executor(root, confirm=lambda request: calls.append((request.tool_name, request.preview)) or True)

            result = executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": str(outside), "content": "x"}),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("outside workspace", result.error or "")
            self.assertEqual(result.data["permission"], "deny")
            self.assertEqual(calls, [])

    def test_write_runtime_internal_denies_without_confirm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_root = root / ".forestcode"
            runtime_root.mkdir()
            calls = []
            executor = self._executor(
                root,
                confirm=lambda request: calls.append((request.tool_name, request.preview)) or True,
                runtime_internal_dirs=frozenset({runtime_root}),
            )

            result = executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": ".forestcode/a.txt", "content": "x"}),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.error, "Permission denied.")
            self.assertEqual(result.data["permission"], "deny")
            self.assertEqual(calls, [])

    def test_read_only_permission_regression_cases(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="peek",
                description="Peek path.",
                input_schema={"type": "object"},
                runner=lambda _context, path: "ok",
                path_getter=lambda args: [PathAccess(args["path"], "read")],
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            runtime_root = root / ".forestcode"
            runtime_root.mkdir()
            (root / ".env").write_text("TOKEN=x", encoding="utf-8")
            outside = Path(tmp) / "outside.txt"
            outside.write_text("x", encoding="utf-8")
            executor = ToolExecutor(
                registry,
                workspace_root=root,
                runtime_internal_dirs=frozenset({runtime_root}),
            )

            outside_result = executor.execute(
                ToolCall(id="outside", name="peek", arguments={"path": str(outside)}),
                RunState.start("test"),
            )
            sensitive_result = executor.execute(
                ToolCall(id="sensitive", name="peek", arguments={"path": ".env"}),
                RunState.start("test"),
            )
            runtime_result = executor.execute(
                ToolCall(id="runtime", name="peek", arguments={"path": ".forestcode/a.txt"}),
                RunState.start("test"),
            )

            self.assertFalse(outside_result.ok)
            self.assertEqual(outside_result.data["permission"], "ask")
            self.assertFalse(sensitive_result.ok)
            self.assertEqual(sensitive_result.data["permission"], "ask")
            self.assertFalse(runtime_result.ok)
            self.assertEqual(runtime_result.data["permission"], "deny")


if __name__ == "__main__":
    unittest.main()
