import tempfile
import unittest
from pathlib import Path

from forestcode.core import ToolCall, ToolExecutor
from forestcode.core.run_state import RunState
from forestcode.tools import PatchService, ReadStateStore, ToolRuntimeServices, create_builtin_tool_registry


class EditingToolsTest(unittest.TestCase):
    def _executor(self, root: Path, confirm=None):
        store = ReadStateStore()
        patch_service = PatchService(read_state_store=store)
        runtime = ToolRuntimeServices(
            read_state_store=store,
            patch_service=patch_service,
            confirm=confirm or (lambda _request: True),
        )
        return ToolExecutor(create_builtin_tool_registry(), workspace_root=root, runtime=runtime), store

    def _read(self, executor: ToolExecutor, path: str, limit: int = 20000):
        return executor.execute(
            ToolCall(id=f"read_{path}", name="read_file", arguments={"path": path, "offset": 0, "limit": limit}),
            RunState.start("test"),
        )

    def test_edit_file_replaces_text_after_full_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "app.py"
            target.write_text("hello = 1\n", encoding="utf-8")
            executor, _store = self._executor(root)
            self._read(executor, "app.py")

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "app.py", "old_text": "hello = 1", "new_text": "hello = 2"},
                ),
                RunState.start("test"),
            )

            self.assertTrue(result.ok)
            self.assertIn("Applied patch", result.content)
            self.assertEqual(result.data["status"], "applied")
            self.assertIn("patch_id", result.data)
            self.assertEqual(target.read_text(encoding="utf-8"), "hello = 2\n")

    def test_edit_file_requires_old_text_to_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("hello\n", encoding="utf-8")
            executor, _store = self._executor(root)
            self._read(executor, "app.py")

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "app.py", "old_text": "missing", "new_text": "new"},
                ),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("old_text was not found", result.error or "")

    def test_edit_file_rejects_multiple_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("x\nx\n", encoding="utf-8")
            executor, _store = self._executor(root)
            self._read(executor, "app.py")

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "app.py", "old_text": "x", "new_text": "y"},
                ),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("exactly once", result.error or "")

    def test_edit_file_requires_prior_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("hello\n", encoding="utf-8")
            executor, _store = self._executor(root)

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "app.py", "old_text": "hello", "new_text": "hi"},
                ),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("Must read file before editing", result.error or "")

    def test_edit_file_rejects_partial_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("hello world\n", encoding="utf-8")
            executor, _store = self._executor(root)
            self._read(executor, "app.py", limit=5)

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "app.py", "old_text": "hello", "new_text": "hi"},
                ),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("full file", result.error or "")

    def test_edit_file_rejects_changed_since_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "app.py"
            target.write_text("hello\n", encoding="utf-8")
            executor, _store = self._executor(root)
            self._read(executor, "app.py")
            target.write_text("external\n", encoding="utf-8")

            result = executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "app.py", "old_text": "hello", "new_text": "hi"},
                ),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("changed since read", (result.error or "").lower())

    def test_edit_file_diff_is_unified_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("old\n", encoding="utf-8")
            captured = {}

            def confirm(request):
                captured["preview"] = request.preview
                return False

            executor, _store = self._executor(root, confirm=confirm)
            self._read(executor, "app.py")

            executor.execute(
                ToolCall(
                    id="edit",
                    name="edit_file",
                    arguments={"path": "app.py", "old_text": "old", "new_text": "new"},
                ),
                RunState.start("test"),
            )

            self.assertIn("--- a/app.py", captured["preview"])
            self.assertIn("+++ b/app.py", captured["preview"])

    def test_write_file_creates_new_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor, _store = self._executor(root)

            result = executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": "new.txt", "content": "hello\n"}),
                RunState.start("test"),
            )

            self.assertTrue(result.ok)
            self.assertEqual((root / "new.txt").read_text(encoding="utf-8"), "hello\n")

    def test_write_file_overwrites_existing_file_after_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("old\n", encoding="utf-8")
            executor, _store = self._executor(root)
            self._read(executor, "a.txt")

            result = executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": "a.txt", "content": "new\n"}),
                RunState.start("test"),
            )

            self.assertTrue(result.ok)
            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")

    def test_write_file_create_preview_is_content_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            captured = {}

            def confirm(request):
                captured["preview"] = request.preview
                return False

            executor, _store = self._executor(root, confirm=confirm)

            executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": "new.txt", "content": "hello"}),
                RunState.start("test"),
            )

            self.assertIn("new file content preview", captured["preview"])
            self.assertIn("hello", captured["preview"])

    def test_write_file_overwrite_preview_is_unified_diff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("old\n", encoding="utf-8")
            captured = {}

            def confirm(request):
                captured["preview"] = request.preview
                return False

            executor, _store = self._executor(root, confirm=confirm)
            self._read(executor, "a.txt")

            executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": "a.txt", "content": "new\n"}),
                RunState.start("test"),
            )

            self.assertIn("--- a/a.txt", captured["preview"])
            self.assertIn("+++ b/a.txt", captured["preview"])

    def test_write_file_create_fails_if_file_appears_before_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "new.txt"

            def confirm(_request):
                target.write_text("external\n", encoding="utf-8")
                return True

            executor, _store = self._executor(root, confirm=confirm)

            result = executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": "new.txt", "content": "hello"}),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertEqual(result.data["status"], "failed")
            self.assertEqual(target.read_text(encoding="utf-8"), "external\n")

    def test_write_file_rejects_protected_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executor, _store = self._executor(root)

            git_result = executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": ".git/config", "content": "x"}),
                RunState.start("test"),
            )
            env_result = executor.execute(
                ToolCall(id="write", name="write_file", arguments={"path": ".env", "content": "x"}),
                RunState.start("test"),
            )

            self.assertFalse(git_result.ok)
            self.assertIn("protected", git_result.error or "")
            self.assertFalse(env_result.ok)
            self.assertIn("sensitive", env_result.error or "")


if __name__ == "__main__":
    unittest.main()
