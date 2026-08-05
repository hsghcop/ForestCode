import tempfile
import unittest
from pathlib import Path

from forestcode.tools import PatchService, ReadStateStore, compute_content_hash, compute_diff


class PatchServiceTest(unittest.TestCase):
    def test_compute_diff_returns_unified_diff(self):
        diff = compute_diff("a\nb\n", "a\nc\n", "file.txt")

        self.assertIn("--- a/file.txt", diff)
        self.assertIn("+++ b/file.txt", diff)
        self.assertIn("-b", diff)
        self.assertIn("+c", diff)

    def test_compute_diff_no_changes(self):
        self.assertEqual(compute_diff("same", "same", "file.txt"), "(no changes)")

    def test_compute_content_hash_is_stable(self):
        self.assertEqual(compute_content_hash("abc"), compute_content_hash("abc"))

    def test_apply_writes_edit_patch_and_updates_read_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "a.txt"
            target.write_text("old\n", encoding="utf-8")
            store = ReadStateStore()
            store.record(target, "old\n", target.stat().st_mtime, is_partial=False)
            service = PatchService(read_state_store=store)
            proposal = service.propose(
                path="a.txt",
                resolved_path=target,
                operation="edit",
                base_hash=compute_content_hash("old\n"),
                diff=compute_diff("old\n", "new\n", "a.txt"),
                updated_content="new\n",
                tool_call_id="call_1",
            )

            service.apply(proposal)

            self.assertEqual(target.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(proposal.status, "applied")
            self.assertEqual(store.get(target).content_hash, compute_content_hash("new\n"))

    def test_apply_rejects_base_hash_mismatch_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.txt"
            target.write_text("current\n", encoding="utf-8")
            service = PatchService()
            proposal = service.propose(
                path="a.txt",
                resolved_path=target,
                operation="write",
                base_hash=compute_content_hash("old\n"),
                diff=compute_diff("old\n", "new\n", "a.txt"),
                updated_content="new\n",
                tool_call_id="call_1",
            )

            with self.assertRaisesRegex(ValueError, "File changed"):
                service.apply(proposal)

            self.assertEqual(target.read_text(encoding="utf-8"), "current\n")
            self.assertEqual(proposal.status, "failed")
            self.assertIn("File changed", proposal.error or "")

    def test_apply_create_rejects_existing_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.txt"
            target.write_text("current\n", encoding="utf-8")
            service = PatchService()
            proposal = service.propose(
                path="a.txt",
                resolved_path=target,
                operation="create",
                base_hash=None,
                diff="preview",
                updated_content="new\n",
                tool_call_id="call_1",
            )

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                service.apply(proposal)

            self.assertEqual(target.read_text(encoding="utf-8"), "current\n")
            self.assertEqual(proposal.status, "failed")

    def test_reject_marks_proposal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.txt"
            service = PatchService()
            proposal = service.propose(
                path="a.txt",
                resolved_path=target,
                operation="create",
                base_hash=None,
                diff="preview",
                updated_content="new\n",
                tool_call_id="call_1",
            )

            service.reject(proposal)

            self.assertEqual(proposal.status, "rejected")

    def test_read_state_store_record_get_clear(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "a.txt"
            target.write_text("hello", encoding="utf-8")
            store = ReadStateStore()

            self.assertIsNone(store.get(target))
            store.record(target, "hello", target.stat().st_mtime, is_partial=False, offset=0, limit=5)

            state = store.get(target)
            self.assertIsNotNone(state)
            self.assertEqual(state.content_hash, compute_content_hash("hello"))
            self.assertFalse(state.is_partial)

            store.clear(target)
            self.assertIsNone(store.get(target))


if __name__ == "__main__":
    unittest.main()
