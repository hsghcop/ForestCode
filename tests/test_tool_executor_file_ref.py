import tempfile
import unittest
from pathlib import Path

from forestcode.core import ToolCall, ToolExecutor
from forestcode.core.run_state import RunState
from forestcode.tools import (
    ReadStateStore,
    ToolDefinition,
    ToolRegistry,
    ToolRuntimeServices,
    create_builtin_tool_registry,
)

_THRESHOLD = 8_000


def _large_tool_registry(output: str) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                name="large_tool",
                description="returns large output for testing",
                input_schema={"type": "object"},
                runner=lambda _ctx, **_kw: output,
            )
        ]
    )


def _executor(
    root: Path,
    registry: ToolRegistry,
    *,
    tool_results_dir: Path | None = None,
    session_id: str | None = None,
    file_ref_threshold: int = _THRESHOLD,
    runtime_internal_dirs: frozenset = frozenset(),
    runtime_exception_dirs: frozenset = frozenset(),
) -> ToolExecutor:
    store = ReadStateStore()
    runtime = ToolRuntimeServices(read_state_store=store)
    return ToolExecutor(
        registry,
        workspace_root=root,
        runtime=runtime,
        tool_results_dir=tool_results_dir,
        session_id=session_id,
        file_ref_threshold=file_ref_threshold,
        runtime_internal_dirs=runtime_internal_dirs,
        runtime_exception_dirs=runtime_exception_dirs,
    )


def _call(executor: ToolExecutor, tool_name: str = "large_tool", call_id: str = "c1"):
    return executor.execute(
        ToolCall(id=call_id, name=tool_name, arguments={}),
        RunState.start("test"),
    )


class FileRefWriteTest(unittest.TestCase):
    def test_file_ref_written_when_above_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool_results = root / ".forestcode" / "tool-results"
            big_output = "x" * (_THRESHOLD + 1_000)
            ex = _executor(
                root,
                _large_tool_registry(big_output),
                tool_results_dir=tool_results,
                session_id="s1",
                runtime_internal_dirs=frozenset({root / ".forestcode"}),
                runtime_exception_dirs=frozenset({tool_results}),
            )
            result = _call(ex)
            self.assertTrue(result.ok)
            self.assertIn("[output written to", result.content)
            written = list(tool_results.glob("s1/c1-large_tool.txt"))
            self.assertEqual(len(written), 1)
            self.assertEqual(written[0].read_text(encoding="utf-8"), big_output)

    def test_no_file_ref_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool_results = root / ".forestcode" / "tool-results"
            small_output = "x" * 100
            ex = _executor(
                root,
                _large_tool_registry(small_output),
                tool_results_dir=tool_results,
                session_id="s1",
            )
            result = _call(ex)
            self.assertTrue(result.ok)
            self.assertNotIn("[output written to", result.content)
            self.assertEqual(result.content, small_output)
            self.assertFalse(tool_results.exists())

    def test_file_ref_graceful_without_session_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tool_results = root / ".forestcode" / "tool-results"
            big_output = "y" * (_THRESHOLD + 1_000)
            ex = _executor(
                root,
                _large_tool_registry(big_output),
                tool_results_dir=tool_results,
                session_id=None,
            )
            result = _call(ex)
            self.assertTrue(result.ok)
            self.assertNotIn("[output written to", result.content)

    def test_file_ref_graceful_when_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp_ws:
            with tempfile.TemporaryDirectory() as tmp_out:
                root = Path(tmp_ws)
                outside_results = Path(tmp_out) / "tool-results"
                big_output = "z" * (_THRESHOLD + 1_000)
                ex = _executor(
                    root,
                    _large_tool_registry(big_output),
                    tool_results_dir=outside_results,
                    session_id="s1",
                )
                result = _call(ex)
                self.assertTrue(result.ok)
                self.assertNotIn("[output written to", result.content)


class FileRefEndToEndTest(unittest.TestCase):
    def test_read_file_can_read_tool_results(self):
        """Write large output to file-ref, then read it back via read_file."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            forestcode_dir = root / ".forestcode"
            tool_results = forestcode_dir / "tool-results"
            runtime_internal = frozenset({forestcode_dir})
            runtime_exception = frozenset({tool_results})
            big_output = "A" * (_THRESHOLD + 1_000)

            store = ReadStateStore()
            runtime = ToolRuntimeServices(read_state_store=store)
            builtin_registry = create_builtin_tool_registry()
            large_tool = ToolDefinition(
                name="large_tool",
                description="returns large output for testing",
                input_schema={"type": "object"},
                runner=lambda _ctx, **_kw: big_output,
            )
            combined = ToolRegistry([large_tool] + builtin_registry.list_tools())
            ex = ToolExecutor(
                combined,
                workspace_root=root,
                runtime=runtime,
                tool_results_dir=tool_results,
                session_id="s1",
                file_ref_threshold=_THRESHOLD,
                runtime_internal_dirs=runtime_internal,
                runtime_exception_dirs=runtime_exception,
            )

            r1 = ex.execute(
                ToolCall(id="c1", name="large_tool", arguments={}),
                RunState.start("test"),
            )
            self.assertTrue(r1.ok)
            self.assertIn("[output written to", r1.content)

            # Extract the ref path from "[output written to <path>] ..."
            ref_path = r1.content.split("[output written to ")[1].split("]")[0]

            r2 = ex.execute(
                ToolCall(
                    id="c2",
                    name="read_file",
                    arguments={"path": ref_path, "offset": 0, "limit": 20000},
                ),
                RunState.start("test"),
            )
            # read_file must succeed (no "File not found" from sandbox block).
            # The result itself may be file-referenced again if large, but ok=True
            # proves the sandbox exception opened the path for reading.
            self.assertTrue(r2.ok, msg=f"read_file failed: {r2.error}")
            self.assertIsNone(r2.error)


if __name__ == "__main__":
    unittest.main()
