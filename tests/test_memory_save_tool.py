import tempfile
import unittest
from pathlib import Path

from forestcode.core import ToolCall, ToolExecutor
from forestcode.core.run_state import RunState
from forestcode.tools import PatchService, ToolContext, ToolRuntimeServices, create_builtin_tool_registry
from forestcode.tools.builtin import _propose_save_memory, _validate_save_memory


class SaveMemoryToolTest(unittest.TestCase):
    def test_validator_accepts_valid_arguments(self):
        args = _validate_save_memory(
            {
                "name": "terse-responses",
                "memory_type": "feedback",
                "content": "Keep responses concise.",
            }
        )

        self.assertEqual(args["name"], "terse-responses")
        self.assertEqual(args["memory_type"], "feedback")
        self.assertEqual(args["content"], "Keep responses concise.")

    def test_validator_rejects_invalid_arguments(self):
        cases = [
            {"name": "BadName", "memory_type": "feedback", "content": "ok"},
            {"name": "x" * 65, "memory_type": "feedback", "content": "ok"},
            {"name": "valid-name", "memory_type": "other", "content": "ok"},
            {"name": "valid-name", "memory_type": "feedback", "content": ""},
            {"name": "valid-name", "memory_type": "feedback", "content": "x" * 4001},
            {"name": "valid-name", "memory_type": "feedback", "content": "ok\n## bad\n"},
        ]

        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    _validate_save_memory(arguments)

    def test_proposer_creates_memory_file_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolContext(workspace_root=root, patch_service=PatchService())

            proposal = _propose_save_memory(
                context,
                {"name": "user-role", "memory_type": "user", "content": "User is a backend engineer."},
                "call_1",
            )

            self.assertEqual(proposal.path, "MEMORY.md")
            self.assertEqual(proposal.operation, "create")
            self.assertIsNone(proposal.base_hash)
            self.assertIn("--- a/MEMORY.md", proposal.diff)
            self.assertIn("## [user] user-role", proposal.updated_content)

    def test_proposer_edits_existing_memory_file_patch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "MEMORY.md"
            target.write_text("# Long-term Memory\n\n## [user] user-role\nOld.\n", encoding="utf-8")
            context = ToolContext(workspace_root=root, patch_service=PatchService())

            proposal = _propose_save_memory(
                context,
                {"name": "user-role", "memory_type": "feedback", "content": "New."},
                "call_1",
            )

            self.assertEqual(proposal.operation, "edit")
            self.assertIsNotNone(proposal.base_hash)
            self.assertIn("-Old.", proposal.diff)
            self.assertIn("+New.", proposal.diff)

    def test_proposer_treats_empty_existing_file_as_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MEMORY.md").write_text("", encoding="utf-8")
            context = ToolContext(workspace_root=root, patch_service=PatchService())

            proposal = _propose_save_memory(
                context,
                {"name": "project-decision", "memory_type": "project", "content": "Use patch-first edits."},
                "call_1",
            )

            self.assertEqual(proposal.operation, "edit")
            self.assertIsNotNone(proposal.base_hash)
            self.assertIn("## [project] project-decision", proposal.updated_content)

    def test_registry_exposes_save_memory_path_access_when_enabled(self):
        registry = create_builtin_tool_registry()
        tool = registry.get("save_memory")

        self.assertIsNotNone(tool)
        assert tool is not None
        accesses = tool.get_paths({"name": "user-role", "memory_type": "user", "content": "x"})
        self.assertEqual(len(accesses), 1)
        self.assertEqual(accesses[0].path, "MEMORY.md")
        self.assertEqual(accesses[0].intent, "write")

    def test_registry_can_disable_save_memory(self):
        registry = create_builtin_tool_registry(enable_memory_write=False)

        self.assertIsNone(registry.get("save_memory"))

    def test_executor_applies_save_memory_through_patch_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            previews = []
            runtime = ToolRuntimeServices(
                patch_service=PatchService(),
                confirm=lambda request: previews.append(request.preview) or True,
            )
            executor = ToolExecutor(create_builtin_tool_registry(), workspace_root=root, runtime=runtime)

            result = executor.execute(
                ToolCall(
                    id="save",
                    name="save_memory",
                    arguments={
                        "name": "user-role",
                        "memory_type": "user",
                        "content": "User is a backend engineer.",
                    },
                ),
                RunState.start("remember this"),
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["status"], "applied")
            self.assertIn("## [user] user-role", (root / "MEMORY.md").read_text(encoding="utf-8"))
            self.assertTrue(previews)
            self.assertIn("MEMORY.md", previews[0])


if __name__ == "__main__":
    unittest.main()
