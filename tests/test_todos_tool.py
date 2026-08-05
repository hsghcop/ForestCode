import io
import tempfile
import unittest
from pathlib import Path

from forestcode.cli import print_agent_event
from forestcode.core import ToolExecutor
from forestcode.core.run_state import RunState
from forestcode.core.types import RunEvent, ToolCall, ToolResult
from forestcode.memory import SessionRecorder, SessionStore
from forestcode.plan import PlanStore, TodoItem, todos_to_dicts
from forestcode.tools import PathAccess, ToolDefinition, ToolRegistry, ToolRuntimeServices
from forestcode.tools.todos import _run_write_todos, _validate_write_todos, create_write_todos_tool
from forestcode.tools.types import ToolContext


class ValidatorTest(unittest.TestCase):
    def test_returns_todo_items(self):
        out = _validate_write_todos({"todos": [{"content": "Run tests", "status": "pending"}]})
        self.assertIsInstance(out["todos"][0], TodoItem)

    def test_missing_active_form_ok(self):
        out = _validate_write_todos({"todos": [{"content": "Run tests"}]})
        self.assertEqual(out["todos"][0].active_form, "Run tests")

    def test_missing_todos_raises(self):
        with self.assertRaises(ValueError):
            _validate_write_todos({})


class RunnerTest(unittest.TestCase):
    def _ctx(self, plan_store):
        return ToolContext(workspace_root=Path("."), plan_store=plan_store)

    def test_sets_plan(self):
        store = PlanStore()
        items = [TodoItem("a", "doing a", "in_progress"), TodoItem("b", "doing b")]
        msg = _run_write_todos(self._ctx(store), items)
        self.assertEqual(store.get(), items)
        self.assertIn("1 in progress", msg)

    def test_all_done_clears(self):
        store = PlanStore()
        items = [TodoItem("a", "doing a", "completed"), TodoItem("b", "doing b", "completed")]
        msg = _run_write_todos(self._ctx(store), items)
        self.assertEqual(store.get(), [])
        self.assertIn("Plan cleared", msg)


class ExecutorTest(unittest.TestCase):
    def _executor(self, plan_store):
        registry = ToolRegistry([create_write_todos_tool()])
        return ToolExecutor(registry, workspace_root=".", runtime=ToolRuntimeServices(plan_store=plan_store))

    def test_state_tool_allowed_and_tagged(self):
        store = PlanStore()
        executor = self._executor(store)
        call = ToolCall(id="c1", name="write_todos", arguments={"todos": [{"content": "a"}]})
        result = executor.execute(call, RunState.start("task"))
        self.assertTrue(result.ok)
        self.assertTrue(result.data and result.data.get("state_only"))
        self.assertEqual(store.get()[0].content, "a")

    def test_missing_plan_store_returns_error(self):
        executor = self._executor(None)
        call = ToolCall(id="c1", name="write_todos", arguments={"todos": [{"content": "a"}]})
        result = executor.execute(call, RunState.start("task"))
        self.assertFalse(result.ok)

    def test_state_tool_with_path_access_denied(self):
        tool = ToolDefinition(
            name="bad_state",
            description="state tool that wrongly declares path access",
            input_schema={"type": "object"},
            runner=lambda context: "x",
            risk_level="state",
            is_read_only=False,
            path_getter=lambda args: [PathAccess("somefile.txt", "read")],
        )
        executor = ToolExecutor(
            ToolRegistry([tool]),
            workspace_root=".",
            runtime=ToolRuntimeServices(),
        )
        call = ToolCall(id="c1", name="bad_state", arguments={})
        result = executor.execute(call, RunState.start("task"))
        self.assertFalse(result.ok)
        self.assertEqual(result.data.get("permission"), "deny")


class PersistenceTest(unittest.TestCase):
    def test_write_through_persists_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            plan_store = PlanStore(on_change=lambda items: store.save_plan("s1", todos_to_dicts(items)))
            plan_store.set([TodoItem("a", "doing a", "in_progress")])
            loaded = store.load("s1")
            self.assertEqual(
                loaded.plan,
                [{"content": "a", "active_form": "doing a", "status": "in_progress"}],
            )

    def test_recorder_skips_state_only_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            state = RunState.start("task")
            state.tool_results.append(
                ToolResult(
                    tool_call_id="c1",
                    tool_name="write_todos",
                    ok=True,
                    content="Plan updated",
                    summary="Plan updated",
                    data={"state_only": True},
                )
            )
            state.tool_results.append(
                ToolResult(tool_call_id="c2", tool_name="read_file", ok=True, content="data", summary="data")
            )
            state.finish("done")
            SessionRecorder(store, "s1").record_run(state)
            loaded = store.load("s1")
            tool_entries = [entry for entry in loaded.entries if entry.kind == "tool_result"]
            self.assertEqual(len(tool_entries), 1)
            self.assertEqual(tool_entries[0].metadata["tool_name"], "read_file")


class PlanRenderTest(unittest.TestCase):
    def test_plan_line_printed_for_write_todos(self):
        buf = io.StringIO()
        event = RunEvent(
            "tool_call_finished",
            {"tool_name": "write_todos", "ok": True, "summary": "Plan updated (0 done / 1 in progress / 0 pending)."},
        )
        print_agent_event(event, buf)
        self.assertIn("Plan> Plan updated", buf.getvalue())

    def test_no_plan_line_for_other_tools(self):
        buf = io.StringIO()
        event = RunEvent("tool_call_finished", {"tool_name": "read_file", "ok": True, "summary": "data"})
        print_agent_event(event, buf)
        self.assertNotIn("Plan>", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
