import unittest
from dataclasses import FrozenInstanceError

from forestcode.plan import PlanStore, TodoItem


class PlanStoreTest(unittest.TestCase):
    def test_get_returns_copy(self):
        store = PlanStore()
        store.set([TodoItem("a", "doing a", "pending")])
        got = store.get()
        got.append(TodoItem("b", "doing b"))
        self.assertEqual(len(store.get()), 1)

    def test_todoitem_is_frozen(self):
        item = TodoItem("a", "doing a")
        with self.assertRaises(FrozenInstanceError):
            item.status = "completed"  # type: ignore[misc]

    def test_set_triggers_on_change(self):
        seen: list[list[TodoItem]] = []
        store = PlanStore(on_change=lambda items: seen.append(list(items)))
        store.set([TodoItem("a", "doing a")])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][0].content, "a")

    def test_seed_does_not_trigger_on_change(self):
        seen: list[list[TodoItem]] = []
        store = PlanStore(on_change=lambda items: seen.append(list(items)))
        store.seed([TodoItem("a", "doing a")])
        self.assertEqual(seen, [])
        self.assertEqual(len(store.get()), 1)

    def test_set_without_callback_does_not_raise(self):
        store = PlanStore()
        store.set([TodoItem("a", "doing a")])
        self.assertEqual(len(store.get()), 1)


if __name__ == "__main__":
    unittest.main()
