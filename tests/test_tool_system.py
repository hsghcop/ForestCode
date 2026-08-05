import json
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

from forestcode.cli import build_agent_loop
from forestcode.core import FakeModelClient, ModelOutput, ToolCall, ToolExecutor
from forestcode.core.run_state import RunState
from forestcode.memory import MemoryEntry, SessionStore
from forestcode.tools import (
    ToolContext,
    ToolDefinition,
    ToolRegistry,
    ToolRuntimeServices,
    create_builtin_tool_registry,
)
from forestcode.tools.builtin import (
    _run_get_file_info,
    _run_glob_files,
    _run_grep_text,
    _run_list_files,
    _run_read_file,
)


class ToolSystemTest(unittest.TestCase):
    def _runtime_workspace(self):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        private_root = root / "private_root"
        private_root.mkdir()
        (private_root / "x.txt").write_text("alpha-517\n", encoding="utf-8")
        (private_root / "foo.json").write_text('{"value": "alpha-517"}\n', encoding="utf-8")
        (root / "public.txt").write_text("public\n", encoding="utf-8")
        context = ToolContext(
            workspace_root=root,
            runtime_internal_dirs=frozenset({private_root}),
        )
        return temp, root, private_root, context

    @staticmethod
    def _normalize_error_path(message: str, path: str) -> str:
        return message.replace(path, "{path}")

    def test_registry_registers_and_exports_model_schema(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="hello",
                description="Say hello.",
                input_schema={"type": "object", "properties": {}},
                runner=lambda _context: "hello",
            )
        )

        self.assertIsNotNone(registry.get("hello"))
        schemas = registry.list_model_schemas()
        self.assertEqual(schemas[0]["function"]["name"], "hello")

    def test_executor_keeps_legacy_dict_tools_working(self):
        executor = ToolExecutor({"echo": lambda args: args["text"]})
        result = executor.execute(
            ToolCall(id="call_1", name="echo", arguments={"text": "ok"}),
            RunState.start("test"),
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.content, "ok")

    def test_builtin_read_only_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("def main():\n    return 'needle'\n", encoding="utf-8")
            (root / "README.md").write_text("hello\n", encoding="utf-8")

            executor = ToolExecutor(create_builtin_tool_registry(), workspace_root=root)
            state = RunState.start("test")

            list_result = executor.execute(
                ToolCall(id="list", name="list_files", arguments={"path": "."}),
                state,
            )
            self.assertTrue(list_result.ok)
            self.assertIn("dir  src", list_result.content)

            glob_result = executor.execute(
                ToolCall(id="glob", name="glob_files", arguments={"pattern": "*.py"}),
                state,
            )
            self.assertTrue(glob_result.ok)
            self.assertIn("src/app.py", glob_result.content)

            grep_result = executor.execute(
                ToolCall(id="grep", name="grep_text", arguments={"pattern": "needle"}),
                state,
            )
            self.assertTrue(grep_result.ok)
            self.assertIn("src/app.py:2:", grep_result.content)

            read_result = executor.execute(
                ToolCall(
                    id="read",
                    name="read_file",
                    arguments={"path": "src/app.py", "offset": 0, "limit": 8},
                ),
                state,
            )
            self.assertTrue(read_result.ok)
            self.assertIn("FILE src/app.py", read_result.content)
            self.assertIn("def main", read_result.content)

            info_result = executor.execute(
                ToolCall(id="info", name="get_file_info", arguments={"path": "README.md"}),
                state,
            )
            self.assertTrue(info_result.ok)
            self.assertIn("type: file", info_result.content)
            self.assertIn("size_bytes:", info_result.content)

    def test_read_session_history_reads_only_bound_session_with_zero_declared_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            store.append_entry("current", MemoryEntry(kind="message", role="user", content="current secret"))
            store.append_entry("other", MemoryEntry(kind="message", role="user", content="other secret"))
            registry = create_builtin_tool_registry(session_store=store, session_id="current")
            tool = registry.get("read_session_history")
            self.assertIsNotNone(tool)
            assert tool is not None

            self.assertEqual(tool.get_paths({"offset": 0, "limit": 20}), [])
            self.assertNotIn("session_id", tool.input_schema["properties"])

            result = ToolExecutor(registry, workspace_root=root).execute(
                ToolCall(
                    id="history",
                    name="read_session_history",
                    arguments={"offset": 0, "limit": 20},
                ),
                RunState.start("read history"),
            )

            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["session_id"], "current")
            self.assertEqual(payload["entries"][0]["content"], "current secret")
            self.assertNotIn("other secret", result.content)

    def test_read_session_history_rejects_session_id_argument(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            registry = create_builtin_tool_registry(session_store=store, session_id="current")

            result = ToolExecutor(registry, workspace_root=root).execute(
                ToolCall(
                    id="history",
                    name="read_session_history",
                    arguments={"session_id": "other", "offset": 0},
                ),
                RunState.start("read history"),
            )

            self.assertFalse(result.ok)
            self.assertIn("unsupported arguments: session_id", result.error or "")

    def test_read_session_history_returns_parseable_json_with_large_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            for index in range(30):
                store.append_entry(
                    "current",
                    MemoryEntry(kind="message", role="user", content=f"{index}-" + "x" * 5000),
                )
            registry = create_builtin_tool_registry(session_store=store, session_id="current")

            result = ToolExecutor(registry, workspace_root=root, max_output_chars=20_000).execute(
                ToolCall(
                    id="history",
                    name="read_session_history",
                    arguments={"offset": 0, "limit": 20},
                ),
                RunState.start("read history"),
            )

            self.assertTrue(result.ok)
            self.assertLessEqual(len(result.content), 20_000)
            payload = json.loads(result.content)
            self.assertEqual(payload["session_id"], "current")
            self.assertEqual(payload["total_entries"], 30)
            self.assertTrue(payload["has_more"])
            self.assertIn(payload["entry_content_chars"], {4000, 2000, 800, 200})

    def test_read_session_history_reports_actual_entry_content_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            for index in range(3):
                store.append_entry(
                    "current",
                    MemoryEntry(kind="message", role="user", content=f"{index}-" + "x" * 3000),
                )
            registry = create_builtin_tool_registry(session_store=store, session_id="current")

            result = ToolExecutor(registry, workspace_root=root, max_output_chars=4_000).execute(
                ToolCall(
                    id="history",
                    name="read_session_history",
                    arguments={"offset": 0, "limit": 3},
                ),
                RunState.start("read history"),
            )

            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertLess(payload["entry_content_chars"], 4000)
            self.assertTrue(any(entry.get("content_truncated") for entry in payload["entries"]))

    def test_read_session_history_keeps_small_entry_at_default_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            store.append_entry("current", MemoryEntry(kind="message", role="user", content="x" * 1000))
            registry = create_builtin_tool_registry(session_store=store, session_id="current")

            result = ToolExecutor(registry, workspace_root=root, max_output_chars=20_000).execute(
                ToolCall(
                    id="history",
                    name="read_session_history",
                    arguments={"offset": 0, "limit": 1},
                ),
                RunState.start("read history"),
            )

            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["entry_content_chars"], 4000)
            self.assertFalse(any(entry.get("content_truncated") for entry in payload["entries"]))

    def test_read_session_history_falls_back_when_entries_cannot_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            for index in range(200):
                store.append_entry(
                    "current",
                    MemoryEntry(kind="message", role="user", content=f"{index}-" + "x" * 1000),
                )
            registry = create_builtin_tool_registry(session_store=store, session_id="current")

            result = ToolExecutor(registry, workspace_root=root, max_output_chars=500).execute(
                ToolCall(
                    id="history",
                    name="read_session_history",
                    arguments={"offset": 0, "limit": 200},
                ),
                RunState.start("read history"),
            )

            self.assertTrue(result.ok)
            payload = json.loads(result.content)
            self.assertEqual(payload["entries"], [])
            self.assertIn("entries omitted", payload["warning"])

    def test_workspace_outside_path_requires_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "workspace"
            root.mkdir()
            outside = Path(tmp) / "outside.txt"
            outside.write_text("secret", encoding="utf-8")

            executor = ToolExecutor(create_builtin_tool_registry(), workspace_root=root)
            result = executor.execute(
                ToolCall(id="read", name="read_file", arguments={"path": str(outside)}),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("outside workspace", result.error or "")

    def test_sensitive_file_requires_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("TOKEN=value", encoding="utf-8")

            executor = ToolExecutor(create_builtin_tool_registry(), workspace_root=root)
            result = executor.execute(
                ToolCall(id="read", name="read_file", arguments={"path": ".env"}),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("Sensitive path", result.error or "")

    def test_binary_file_returns_tool_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data.bin").write_bytes(b"abc\x00def")

            executor = ToolExecutor(create_builtin_tool_registry(), workspace_root=root)
            result = executor.execute(
                ToolCall(id="read", name="read_file", arguments={"path": "data.bin"}),
                RunState.start("test"),
            )

            self.assertFalse(result.ok)
            self.assertIn("binary", result.error or "")

    def test_list_files_hides_runtime_internal(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            result = _run_list_files(context, ".", 100)

        self.assertIn("file public.txt", result)
        self.assertNotIn("private_root", result)

    def test_list_files_runtime_internal_direct_access(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaisesRegex(FileNotFoundError, "Path not found: private_root"):
                _run_list_files(context, "private_root", 100)

    def test_glob_files_skips_runtime_internal(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            result = _run_glob_files(context, "**/*.json", ".", 100)

        self.assertNotIn("private_root/foo.json", result)

    def test_glob_files_pattern_targeting_runtime_internal(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            result = _run_glob_files(context, "private_root/**", ".", 100)

        self.assertEqual(result, "")

    def test_grep_text_skips_runtime_internal(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            result = _run_grep_text(context, "alpha-517", ".", None, 100)

        self.assertEqual(result, "")

    def test_grep_text_uses_regex_pattern(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            (root / "src" / "app.py").write_text("error_404\nERROR_500\n", encoding="utf-8")
            context = ToolContext(workspace_root=root)

            result = _run_grep_text(context, r"error_\d+", ".", "*.py", 100)

        self.assertIn("src/app.py:1:error_404", result)
        self.assertNotIn("ERROR_500", result)

    def test_grep_text_invalid_regex_returns_value_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("hello\n", encoding="utf-8")
            context = ToolContext(workspace_root=root)

            with self.assertRaisesRegex(ValueError, "Invalid regex pattern"):
                _run_grep_text(context, "[", ".", None, 100)

    def test_read_file_runtime_internal_not_found(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaises(FileNotFoundError) as hidden:
                _run_read_file(context, "private_root/x.txt", 0, 100)
            with self.assertRaises(FileNotFoundError) as missing:
                _run_read_file(context, "missing.txt", 0, 100)

        self.assertEqual(
            self._normalize_error_path(str(hidden.exception), "private_root/x.txt"),
            self._normalize_error_path(str(missing.exception), "missing.txt"),
        )

    def test_get_file_info_runtime_internal_file_not_found(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaises(FileNotFoundError) as hidden:
                _run_get_file_info(context, "private_root/x.txt")
            with self.assertRaises(FileNotFoundError) as missing:
                _run_get_file_info(context, "missing.txt")

        self.assertEqual(
            self._normalize_error_path(str(hidden.exception), "private_root/x.txt"),
            self._normalize_error_path(str(missing.exception), "missing.txt"),
        )

    def test_get_file_info_runtime_internal_dir_not_found(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaisesRegex(FileNotFoundError, "Path not found: private_root"):
                _run_get_file_info(context, "private_root")

    def test_runtime_internal_does_not_match_prefix(self):
        temp, root, _private_root, context = self._runtime_workspace()
        with temp:
            (root / "private_root_extra").mkdir()
            result = _run_list_files(context, ".", 100)

        self.assertIn("dir  private_root_extra", result)
        self.assertNotIn("dir  private_root\n", result)

    def test_empty_runtime_internal_dirs_preserves_legacy_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".forestcode").mkdir()
            context = ToolContext(workspace_root=root)
            result = _run_list_files(context, ".", 100)

        self.assertIn("dir  .forestcode", result)

    def test_multiple_runtime_internal_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / ".forestcode"
            second = root / ".plugin_cache"
            first.mkdir()
            second.mkdir()
            (first / "a.txt").write_text("alpha-517\n", encoding="utf-8")
            (second / "b.txt").write_text("alpha-517\n", encoding="utf-8")
            (root / "public.txt").write_text("public\n", encoding="utf-8")
            context = ToolContext(
                workspace_root=root,
                runtime_internal_dirs=frozenset({first, second}),
            )

            listed = _run_list_files(context, ".", 100)
            grepped = _run_grep_text(context, "alpha-517", ".", None, 100)

        self.assertNotIn(".forestcode", listed)
        self.assertNotIn(".plugin_cache", listed)
        self.assertEqual(grepped, "")

    def test_nested_private_root_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_root = root / ".forestcode"
            (runtime_root / "sessions").mkdir(parents=True)
            (runtime_root / "sessions" / "a.json").write_text("alpha-517\n", encoding="utf-8")
            (root / ".forestcode_config").mkdir()
            context = ToolContext(
                workspace_root=root,
                runtime_internal_dirs=frozenset({runtime_root}),
            )

            listed = _run_list_files(context, ".", 100)
            with self.assertRaises(FileNotFoundError):
                _run_read_file(context, ".forestcode/sessions/a.json", 0, 100)

        self.assertIn("dir  .forestcode_config", listed)
        self.assertNotIn("dir  .forestcode", listed.splitlines())

    def test_dotdot_traversal_read_file(self):
        temp, root, _private_root, context = self._runtime_workspace()
        with temp:
            (root / "normal").mkdir()
            with self.assertRaises(FileNotFoundError):
                _run_read_file(context, "normal/../private_root/x.txt", 0, 100)

    def test_absolute_path_read_file(self):
        temp, _root, private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaises(FileNotFoundError):
                _run_read_file(context, str(private_root / "x.txt"), 0, 100)

    @unittest.skipUnless(os.name == "nt", "Windows path casing behavior")
    def test_case_insensitive_windows(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaises(FileNotFoundError):
                _run_read_file(context, "PRIVATE_ROOT/x.txt", 0, 100)

    @unittest.skipUnless(os.name == "nt", "Windows backslash path behavior")
    def test_backslash_path_read_file(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaises(FileNotFoundError):
                _run_read_file(context, "private_root\\x.txt", 0, 100)

    def test_session_store_runtime_root_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            store = SessionStore(root)

        self.assertEqual(store.runtime_root, root / ".forestcode")

    def test_session_store_runtime_root_custom_session_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            store = SessionStore(root, session_dir=root / "alt" / "sess")

        self.assertEqual(store.runtime_root, root / "alt")

    def test_agent_loop_wires_runtime_internal_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            loop = build_agent_loop(
                FakeModelClient([ModelOutput(text="done")]),
                root,
                session_id="memory-test",
            )
            no_session_loop = build_agent_loop(
                FakeModelClient([ModelOutput(text="done")]),
                root,
                session_id=None,
            )

        self.assertEqual(loop.tool_executor._runtime_internal_dirs, frozenset({root / ".forestcode"}))
        self.assertEqual(no_session_loop.tool_executor._runtime_internal_dirs, frozenset())

    def test_agent_loop_skips_injection_when_runtime_root_invalid(self):
        class InvalidSessionStore:
            def __init__(self, workspace_root):
                self.workspace_root = Path(workspace_root).resolve()
                self.runtime_root = self.workspace_root

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            with patch("forestcode.runtime.factory.SessionStore", InvalidSessionStore):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    loop = build_agent_loop(
                        FakeModelClient([ModelOutput(text="done")]),
                        root,
                        session_id="memory-test",
                    )

        self.assertEqual(loop.tool_executor._runtime_internal_dirs, frozenset())
        self.assertTrue(any("Skipping runtime isolation injection" in str(item.message) for item in caught))

    def test_read_file_error_text_consistency(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaises(FileNotFoundError) as missing:
                _run_read_file(context, "missing.txt", 0, 100)
            with self.assertRaises(FileNotFoundError) as hidden:
                _run_read_file(context, "private_root/x.txt", 0, 100)

        self.assertEqual(
            self._normalize_error_path(str(missing.exception), "missing.txt"),
            self._normalize_error_path(str(hidden.exception), "private_root/x.txt"),
        )

    def test_get_file_info_error_text_consistency(self):
        temp, _root, _private_root, context = self._runtime_workspace()
        with temp:
            with self.assertRaises(FileNotFoundError) as missing:
                _run_get_file_info(context, "missing.txt")
            with self.assertRaises(FileNotFoundError) as hidden:
                _run_get_file_info(context, "private_root/x.txt")

        self.assertEqual(
            self._normalize_error_path(str(missing.exception), "missing.txt"),
            self._normalize_error_path(str(hidden.exception), "private_root/x.txt"),
        )


class _SpyCommandService:
    def __init__(self) -> None:
        self.executed = []
        self.rejected = []

    def execute(self, proposal, *, abort=None) -> None:
        self.executed.append(proposal)

    def reject(self, proposal) -> None:
        self.rejected.append(proposal)


class CommandGateIntegrationTest(unittest.TestCase):
    def test_disabled_command_tool_denied_without_proposer_or_service(self):
        """build_agent_loop(enable_command_tools=False): a model that calls run_command
        (e.g. replayed from history) is denied before any proposer/confirm/service runs."""
        from forestcode.core import ModelOutput

        confirm_calls = []
        service = _SpyCommandService()
        runtime = ToolRuntimeServices(
            command_service=service,
            confirm=lambda request: confirm_calls.append(request.tool_name) or True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            model = FakeModelClient(
                [
                    ModelOutput(tool_calls=[ToolCall(id="c1", name="run_command", arguments={"command": "git status"})]),
                    ModelOutput(text="done"),
                ]
            )
            loop = build_agent_loop(
                model,
                root,
                session_id=None,
                runtime=runtime,
            )
            state = loop.run("inspect the repo")

        command_results = [r for r in state.tool_results if r.tool_name == "run_command"]
        self.assertEqual(len(command_results), 1)
        self.assertFalse(command_results[0].ok)
        self.assertEqual(command_results[0].data["permission"], "deny")
        self.assertEqual(service.executed, [])
        self.assertEqual(confirm_calls, [])


if __name__ == "__main__":
    unittest.main()
