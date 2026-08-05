import tempfile
import unittest
from pathlib import Path

from forestcode.context import ContextBudget
from forestcode.context.providers import SessionContextProvider
from forestcode.core.run_state import RunState
from forestcode.core.types import ToolResult
from forestcode.memory import SessionRecorder, SessionStore


class SessionRecorderToolResultTest(unittest.TestCase):
    def test_records_tool_result_up_to_store_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            recorder = SessionRecorder(store, max_tool_result_store_chars=20_000)
            state = RunState.start("read large output")
            state.tool_results.append(
                ToolResult(
                    tool_call_id="call_1",
                    tool_name="read_file",
                    ok=True,
                    content="x" * 8_000,
                )
            )

            recorder.record_run(state)
            memory = store.load("default")
            stored = [entry for entry in memory.entries if entry.kind == "tool_result"][0]

            # Well under the store budget: kept intact, far beyond the 2k display budget.
            self.assertGreater(len(stored.content), 2_000)
            self.assertEqual(stored.content, "ok:read_file:call_1:" + "x" * 8_000)

    def test_caps_oversized_tool_result_at_store_budget(self):
        # Simulates run_command, whose content (stdout 50KB + stderr 10KB) bypasses
        # ToolExecutor._truncate; the recorder must enforce its own store cap.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            recorder = SessionRecorder(store, max_tool_result_store_chars=20_000)
            state = RunState.start("run a noisy command")
            state.tool_results.append(
                ToolResult(
                    tool_call_id="call_1",
                    tool_name="run_command",
                    ok=True,
                    content="y" * 60_000,
                )
            )

            recorder.record_run(state)
            memory = store.load("default")
            stored = [entry for entry in memory.entries if entry.kind == "tool_result"][0]

            self.assertLessEqual(len(stored.content), 20_000 + len("ok:run_command:call_1:"))
            self.assertIn("<truncated", stored.content)

    def test_historical_tool_result_context_still_uses_display_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            recorder = SessionRecorder(store, max_tool_result_store_chars=20_000)
            state = RunState.start("read large output")
            state.tool_results.append(
                ToolResult(
                    tool_call_id="call_1",
                    tool_name="read_file",
                    ok=True,
                    content="x" * 20_000,
                )
            )
            recorder.record_run(state)

            messages, _metadata, _truncated = SessionContextProvider(store).build_session_context(
                "default",
                ContextBudget(max_tool_result_chars=2_000),
            )
            historical = [message for message in messages if "Historical tool result summary" in message.content][0]

            self.assertLess(len(historical.content), 3_000)
            self.assertIn("<truncated", historical.content)


if __name__ == "__main__":
    unittest.main()
