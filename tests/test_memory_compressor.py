import tempfile
import unittest
from pathlib import Path

from forestcode.memory import (
    MemoryEntry,
    SessionCompressor,
    SessionStore,
    build_compaction_entry,
    effective_compactions,
    serialize_conversation,
)


class RecordingSummarizer:
    def __init__(self, text: str = "<analysis>scratch</analysis><summary>compact summary</summary>") -> None:
        self.text = text
        self.calls = []

    def summarize(self, entries, prior_summary):
        self.calls.append((list(entries), prior_summary))
        return self.text


class FailingSummarizer:
    def __init__(self) -> None:
        self.calls = 0

    def summarize(self, entries, prior_summary):
        self.calls += 1
        raise RuntimeError("model unavailable")


class SessionCompressorTest(unittest.TestCase):
    def test_effective_compactions_uses_latest_major_as_cutoff(self):
        entries = [
            build_compaction_entry("normal before", compaction_kind="normal"),
            build_compaction_entry("major", compaction_kind="major"),
            build_compaction_entry("normal after", compaction_kind="normal"),
        ]

        self.assertEqual([entry.content for _index, entry in effective_compactions(entries)], ["major", "normal after"])

    def test_serialize_conversation_uses_flat_recorded_entries(self):
        text = serialize_conversation(
            [
                MemoryEntry(kind="message", role="user", content="task"),
                MemoryEntry(kind="message", role="assistant", content="answer"),
                MemoryEntry(kind="tool_result", content="ok:read_file:call_1:" + "x" * 50),
            ],
            max_tool_result_chars=45,
        )

        self.assertIn("[User]: task", text)
        self.assertIn("[Assistant]: answer", text)
        self.assertIn("[Tool result]: ok:read_file:call_1:", text)
        self.assertIn("<truncated", text)

    def test_normal_compaction_appends_delta_summary_at_legal_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            for entry in [
                MemoryEntry(kind="message", role="user", content="task 1"),
                MemoryEntry(kind="message", role="assistant", content="answer 1"),
                MemoryEntry(kind="message", role="user", content="task 2"),
                MemoryEntry(kind="tool_result", content="ok:read_file:call_1:content"),
                MemoryEntry(kind="message", role="assistant", content="answer 2"),
                MemoryEntry(kind="message", role="user", content="task 3"),
                MemoryEntry(kind="message", role="assistant", content="answer 3"),
            ]:
                store.append_entry("default", entry)
            summarizer = RecordingSummarizer()
            compressor = SessionCompressor(
                store,
                "default",
                summarizer,
                keep_recent_entries=2,
                compact_trigger_entries=3,
            )

            self.assertTrue(compressor.maybe_normal_compact())
            memory = store.load("default")
            compaction = memory.entries[-1]

            self.assertEqual(compaction.kind, "compaction")
            self.assertEqual(compaction.content, "compact summary")
            self.assertEqual(compaction.metadata["compaction_kind"], "normal")
            self.assertEqual(compaction.metadata["first_kept_entry_id"], "entry_000006")
            self.assertEqual(compaction.metadata["source_start"], "entry_000001")
            self.assertEqual(compaction.metadata["source_end"], "entry_000005")
            summarized_ids = [entry.id for entry in summarizer.calls[0][0]]
            self.assertEqual(summarized_ids, ["entry_000001", "entry_000002", "entry_000003", "entry_000004", "entry_000005"])

    def test_normal_compaction_is_incremental_after_latest_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            for label in ["u1", "a1", "u2", "a2", "u3", "a3"]:
                role = "user" if label.startswith("u") else "assistant"
                store.append_entry("default", MemoryEntry(kind="message", role=role, content=label))
            summarizer = RecordingSummarizer("first")
            compressor = SessionCompressor(
                store,
                "default",
                summarizer,
                keep_recent_entries=2,
                compact_trigger_entries=3,
            )
            self.assertTrue(compressor.maybe_normal_compact())
            for label in ["u4", "a4", "u5", "a5"]:
                role = "user" if label.startswith("u") else "assistant"
                store.append_entry("default", MemoryEntry(kind="message", role=role, content=label))

            self.assertTrue(compressor.maybe_normal_compact())

            self.assertEqual(len(summarizer.calls), 2)
            self.assertEqual([entry.id for entry in summarizer.calls[1][0]], ["entry_000005", "entry_000006", "entry_000008", "entry_000009"])
            self.assertEqual(summarizer.calls[1][1], "first")

    def test_split_turn_compaction_summarizes_history_and_turn_prefix_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            for entry in [
                MemoryEntry(kind="message", role="user", content="old task"),
                MemoryEntry(kind="message", role="assistant", content="old answer"),
                MemoryEntry(kind="message", role="user", content="large task"),
                MemoryEntry(kind="tool_result", content="ok:read_file:call_1:large output"),
                MemoryEntry(kind="tool_result", content="ok:grep_text:call_2:more output"),
                MemoryEntry(kind="message", role="assistant", content="large answer"),
            ]:
                store.append_entry("default", entry)
            summarizer = RecordingSummarizer("segment summary")
            compressor = SessionCompressor(
                store,
                "default",
                summarizer,
                keep_recent_entries=1,
                compact_trigger_entries=2,
            )

            self.assertTrue(compressor.maybe_normal_compact())
            memory = store.load("default")
            compaction = memory.entries[-1]

            self.assertEqual(len(summarizer.calls), 2)
            self.assertEqual([entry.content for entry in summarizer.calls[0][0]], ["old task", "old answer"])
            self.assertEqual(
                [entry.content for entry in summarizer.calls[1][0]],
                ["large task", "ok:read_file:call_1:large output", "ok:grep_text:call_2:more output"],
            )
            self.assertIn("History:\nsegment summary", compaction.content)
            self.assertIn("Turn prefix:\nsegment summary", compaction.content)
            self.assertEqual(compaction.metadata["first_kept_entry_id"], "entry_000006")
            self.assertEqual(compaction.metadata["source_end"], "entry_000005")

    def test_summarizer_failures_do_not_modify_session_and_disable_after_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            for label in ["u1", "a1", "u2", "a2", "u3", "a3"]:
                role = "user" if label.startswith("u") else "assistant"
                store.append_entry("default", MemoryEntry(kind="message", role=role, content=label))
            summarizer = FailingSummarizer()
            compressor = SessionCompressor(
                store,
                "default",
                summarizer,
                keep_recent_entries=1,
                compact_trigger_entries=2,
                max_consecutive_failures=2,
            )

            self.assertFalse(compressor.maybe_normal_compact())
            self.assertFalse(compressor.maybe_normal_compact())
            self.assertTrue(compressor.disabled)
            self.assertFalse(compressor.maybe_normal_compact())
            self.assertEqual(summarizer.calls, 2)
            self.assertFalse(any(entry.kind == "compaction" for entry in store.load("default").entries))

    def test_major_compaction_merges_effective_summaries_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            store.append_entry("default", MemoryEntry(kind="message", role="user", content="old task"))
            store.append_entry("default", MemoryEntry(kind="message", role="assistant", content="old answer"))
            store.append_entry(
                "default",
                build_compaction_entry(
                    "normal one",
                    compaction_kind="normal",
                    first_kept_entry_id="entry_000002",
                    source_start="entry_000001",
                    source_end="entry_000001",
                ),
            )
            store.append_entry("default", MemoryEntry(kind="message", role="user", content="new task"))
            store.append_entry("default", MemoryEntry(kind="message", role="assistant", content="new answer"))
            store.append_entry(
                "default",
                build_compaction_entry(
                    "normal two",
                    compaction_kind="normal",
                    first_kept_entry_id="entry_000004",
                    source_start="entry_000002",
                    source_end="entry_000003",
                ),
            )
            summarizer = RecordingSummarizer("major summary")
            compressor = SessionCompressor(
                store,
                "default",
                summarizer,
                keep_recent_entries=1,
                compact_trigger_entries=2,
            )

            self.assertTrue(compressor.maybe_major_compact())
            self.assertFalse(compressor.maybe_major_compact())
            memory = store.load("default")
            major_entries = [
                entry
                for entry in memory.entries
                if entry.kind == "compaction" and entry.metadata.get("compaction_kind") == "major"
            ]

            self.assertEqual(len(major_entries), 1)
            self.assertEqual(major_entries[0].metadata["first_kept_entry_id"], "entry_000005")
            self.assertEqual([entry.content for entry in summarizer.calls[0][0]], ["normal one", "normal two", "new task"])


if __name__ == "__main__":
    unittest.main()
