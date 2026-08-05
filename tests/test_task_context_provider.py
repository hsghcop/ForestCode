import unittest

from forestcode.context.builder import ContextBuilder
from forestcode.context.providers import TaskContextProvider
from forestcode.context.types import ContextBudget
from forestcode.core.run_state import RunState
from forestcode.core.types import Message
from forestcode.plan import PlanStore, TodoItem


def _store(items):
    store = PlanStore()
    store.seed(items)
    return store


class TaskContextProviderTest(unittest.TestCase):
    def test_none_store(self):
        provider = TaskContextProvider(None)
        self.assertEqual(provider.build_plan_message([], ContextBudget()), (None, False))

    def test_empty_plan(self):
        provider = TaskContextProvider(PlanStore())
        self.assertEqual(provider.build_plan_message([], ContextBudget()), (None, False))

    def test_injects_when_no_recent_write(self):
        provider = TaskContextProvider(_store([TodoItem("a", "doing a", "in_progress")]))
        message, truncated = provider.build_plan_message([], ContextBudget())
        self.assertIsNotNone(message)
        self.assertIn("[Current plan]", message.content)
        self.assertIn("[~] doing a", message.content)
        self.assertFalse(truncated)

    def test_suppressed_when_recent_has_successful_write(self):
        provider = TaskContextProvider(_store([TodoItem("a", "doing a")]))
        recent = [Message(role="tool_result", content="ok:write_todos:c1:Plan updated", tool_call_id="c1")]
        self.assertEqual(provider.build_plan_message(recent, ContextBudget()), (None, False))

    def test_not_suppressed_by_failed_write(self):
        provider = TaskContextProvider(_store([TodoItem("a", "doing a")]))
        recent = [Message(role="tool_result", content="error:write_todos:c1:bad", tool_call_id="c1")]
        message, _ = provider.build_plan_message(recent, ContextBudget())
        self.assertIsNotNone(message)

    def test_status_marks(self):
        provider = TaskContextProvider(
            _store([TodoItem("a", "doing a", "completed"), TodoItem("b", "doing b", "pending")])
        )
        message, _ = provider.build_plan_message([], ContextBudget())
        self.assertIn("[x] a", message.content)
        self.assertIn("[ ] b", message.content)

    def test_truncation(self):
        long = "x" * 5000
        provider = TaskContextProvider(_store([TodoItem(long, long)]))
        message, truncated = provider.build_plan_message([], ContextBudget(max_plan_chars=50))
        self.assertTrue(truncated)
        self.assertLessEqual(len(message.content), 50)


class ContextBuilderPlanTest(unittest.TestCase):
    def test_plan_injected_before_recent(self):
        state = RunState.start("do work")
        model_input = ContextBuilder(plan_store=_store([TodoItem("a", "doing a", "in_progress")])).build(state)

        plan_messages = [m for m in model_input.messages if "[Current plan]" in (m.content or "")]
        self.assertEqual(len(plan_messages), 1)
        self.assertEqual(model_input.metadata["plan_item_count"], 1)
        self.assertTrue(model_input.metadata["plan_reminder_injected"])
        self.assertIn("plan", model_input.metadata["context_sources"])

        plan_index = model_input.messages.index(plan_messages[0])
        user_index = next(
            i for i, m in enumerate(model_input.messages) if m.role == "user" and m.content == "do work"
        )
        self.assertLess(plan_index, user_index)

    def test_truncation_sets_metadata(self):
        long = "x" * 5000
        state = RunState.start("do work")
        model_input = ContextBuilder(
            budget=ContextBudget(max_plan_chars=50),
            plan_store=_store([TodoItem(long, long)]),
        ).build(state)
        self.assertTrue(model_input.metadata["truncated"])

    def test_no_plan_store_no_error(self):
        state = RunState.start("do work")
        model_input = ContextBuilder().build(state)
        self.assertNotIn("plan", model_input.metadata.get("context_sources", []))


if __name__ == "__main__":
    unittest.main()
