import unittest

from forestcode.plan import TodoItem, normalize_todos, todos_from_dicts, todos_to_dicts
from forestcode.plan.types import MAX_TODOS


class NormalizeStrictTest(unittest.TestCase):
    def test_missing_active_form_falls_back_to_content(self):
        items = normalize_todos([{"content": "Run tests"}], strict=True)
        self.assertEqual(items[0].active_form, "Run tests")

    def test_empty_content_raises(self):
        with self.assertRaises(ValueError):
            normalize_todos([{"content": "  "}], strict=True)

    def test_unknown_status_raises(self):
        with self.assertRaises(ValueError):
            normalize_todos([{"content": "x", "status": "bogus"}], strict=True)

    def test_too_many_raises(self):
        raw = [{"content": f"t{i}"} for i in range(MAX_TODOS + 1)]
        with self.assertRaises(ValueError):
            normalize_todos(raw, strict=True)

    def test_multiple_in_progress_raises(self):
        raw = [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ]
        with self.assertRaises(ValueError):
            normalize_todos(raw, strict=True)


class NormalizeLenientTest(unittest.TestCase):
    def test_missing_active_form_falls_back(self):
        items = todos_from_dicts([{"content": "Run tests"}])
        self.assertEqual(items[0].active_form, "Run tests")

    def test_unknown_or_missing_status_defaults_pending(self):
        items = todos_from_dicts([{"content": "a", "status": "bogus"}, {"content": "b"}])
        self.assertEqual([i.status for i in items], ["pending", "pending"])

    def test_empty_or_missing_content_skipped(self):
        items = todos_from_dicts([{"content": ""}, {"active_form": "x"}, {"content": "ok"}])
        self.assertEqual([i.content for i in items], ["ok"])

    def test_too_many_truncated(self):
        raw = [{"content": f"t{i}"} for i in range(MAX_TODOS + 5)]
        items = todos_from_dicts(raw)
        self.assertEqual(len(items), MAX_TODOS)

    def test_multiple_in_progress_keeps_first(self):
        raw = [
            {"content": "a", "status": "in_progress"},
            {"content": "b", "status": "in_progress"},
        ]
        items = todos_from_dicts(raw)
        self.assertEqual([i.status for i in items], ["in_progress", "pending"])

    def test_bad_entries_skipped(self):
        items = todos_from_dicts(["not a dict", 42, {"content": "ok"}])
        self.assertEqual([i.content for i in items], ["ok"])

    def test_non_list_returns_empty(self):
        self.assertEqual(todos_from_dicts("nope"), [])


class RoundTripTest(unittest.TestCase):
    def test_round_trip(self):
        items = [
            TodoItem("a", "doing a", "in_progress"),
            TodoItem("b", "doing b", "completed"),
        ]
        self.assertEqual(todos_from_dicts(todos_to_dicts(items)), items)


if __name__ == "__main__":
    unittest.main()
