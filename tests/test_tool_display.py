import unittest

from forestcode.terminal.tool_display import (
    command_output_lines,
    key_args,
    metric,
)


class KeyArgsTest(unittest.TestCase):
    def test_path_tools(self):
        for tool in ("read_file", "write_file", "edit_file", "list_files"):
            self.assertEqual(key_args(tool, {"path": "src/cli.py"}), "src/cli.py")

    def test_run_command_uses_command(self):
        self.assertEqual(key_args("run_command", {"command": "npm test"}), "npm test")

    def test_run_command_truncated(self):
        long = "echo " + "x" * 100
        out = key_args("run_command", {"command": long})
        self.assertLessEqual(len(out), 60)
        self.assertTrue(out.endswith("…"))

    def test_grep_quotes_pattern_and_scope(self):
        self.assertEqual(key_args("grep", {"pattern": "foo", "glob": "*.py"}), '"foo" *.py')
        self.assertEqual(key_args("grep", {"pattern": "foo"}), '"foo"')

    def test_glob(self):
        self.assertEqual(key_args("glob", {"pattern": "**/*.ts"}), "**/*.ts")

    def test_write_todos_count(self):
        self.assertEqual(key_args("write_todos", {"todos": [1, 2, 3]}), "(3 items)")
        self.assertEqual(key_args("write_todos", {}), "")

    def test_unknown_tool_first_present_field(self):
        self.assertEqual(key_args("mystery", {"query": "abc"}), "abc")
        self.assertEqual(key_args("mystery", {}), "")

    def test_none_arguments(self):
        self.assertEqual(key_args("read_file", None), "")


class MetricTest(unittest.TestCase):
    def test_no_data(self):
        self.assertEqual(metric("read_file", None), "")
        self.assertEqual(metric("read_file", {}), "")

    def test_generic_read_lines(self):
        self.assertEqual(metric("read_file", {"lines": 42}), "(42 lines)")

    def test_grep_matches(self):
        self.assertEqual(metric("grep", {"lines": 3}), "(3 matches)")

    def test_list_entries(self):
        self.assertEqual(metric("list_files", {"lines": 7}), "(7 entries)")
        self.assertEqual(metric("glob", {"lines": 2}), "(2 entries)")

    def test_command_exit_code(self):
        self.assertEqual(metric("run_command", {"exit_code": 0}), "(exit 0)")
        self.assertEqual(metric("run_command", {"exit_code": 1}), "(exit 1)")

    def test_edit_diff_counts(self):
        diff = "--- a\n+++ b\n@@ -1,2 +1,2 @@\n-old\n+new\n+extra\n"
        self.assertEqual(metric("edit_file", {"diff": diff}), "(2+/1-)")

    def test_state_only_no_metric(self):
        # state tools carry state_only but no lines -> no metric.
        self.assertEqual(metric("write_todos", {"state_only": True}), "")


class CommandOutputTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(command_output_lines(None), [])
        self.assertEqual(command_output_lines({}), [])

    def test_stdout_then_stderr(self):
        rows = command_output_lines({"stdout": "a\nb", "stderr": "err"})
        self.assertEqual(rows, [("a", False), ("b", False), ("err", True)])

    def test_truncation_marker(self):
        stdout = "\n".join(str(i) for i in range(15))
        rows = command_output_lines({"stdout": stdout}, head=10)
        self.assertEqual(len(rows), 11)
        self.assertEqual(rows[-1], ("… 5 more lines", False))

    def test_no_truncation_when_within_head(self):
        rows = command_output_lines({"stdout": "a\nb\nc"}, head=10)
        self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
