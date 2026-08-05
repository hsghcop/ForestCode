import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from forestcode.core.run_state import RunState
from forestcode.core.types import ToolCall, ToolResult
from forestcode.memory import MemoryEntry, SessionMemory, SessionRecorder, SessionStore, build_compaction_entry


def _jsonl_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SessionStoreTest(unittest.TestCase):
    def test_append_entry_creates_header_then_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            path = store._jsonl_path("default")

            store.append_entry("default", MemoryEntry(kind="message", role="user", content="hello"))
            records = _jsonl_records(path)
            loaded = store.load("default")

            self.assertEqual([record["_t"] for record in records], ["header", "entry"])
            self.assertEqual(records[0]["session_id"], "default")
            self.assertEqual(records[0]["version"], 2)
            self.assertEqual(records[1]["id"], "entry_000001")
            self.assertEqual(loaded.session_id, "default")
            self.assertEqual(loaded.version, 2)
            self.assertEqual(loaded.entries[0].content, "hello")

    def test_append_entry_appends_one_line_per_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)

            store.append_entry("default", MemoryEntry(kind="message", role="user", content="first"))
            store.append_entry("default", build_compaction_entry("summary"))
            store.append_entry("default", MemoryEntry(kind="message", role="assistant", content="second"))
            records = _jsonl_records(store._jsonl_path("default"))

            self.assertEqual([record["_t"] for record in records], ["header", "entry", "entry", "entry"])
            self.assertEqual(
                [record["id"] for record in records if record["_t"] == "entry"],
                ["entry_000001", "entry_000002", "entry_000003"],
            )

    def test_entry_round_trip_preserves_tool_result_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            content = "ok:read_file:call_1:value:with:colon\nand newline"
            memory = SessionMemory(
                session_id="default",
                title="round trip",
                entries=[MemoryEntry(kind="tool_result", content=content, metadata={"ok": True})],
            )

            store.save(memory)
            loaded = store.load("default")

            self.assertEqual(loaded.title, "round trip")
            self.assertEqual(loaded.entries[0].content, content)
            self.assertEqual(loaded.entries[0].metadata, {"ok": True})

    def test_save_plan_last_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)

            store.save_plan("default", [{"content": "old", "status": "pending"}])
            store.save_plan("default", [{"content": "new", "status": "in_progress"}])
            records = _jsonl_records(store._jsonl_path("default"))
            loaded = store.load("default")

            self.assertEqual([record["_t"] for record in records], ["header", "plan", "plan"])
            self.assertEqual(loaded.plan, [{"content": "new", "status": "in_progress"}])

    def test_append_run_adds_timestamp_only_when_missing_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            existing = datetime(2026, 1, 2, tzinfo=UTC).isoformat()

            store.append_run("default", {"user_task": "first", "timestamp": existing})
            store.append_run("default", {"user_task": "second"})
            records = _jsonl_records(store._jsonl_path("default"))
            loaded = store.load("default")

            self.assertEqual([record["_t"] for record in records], ["header", "run", "meta", "run", "meta"])
            self.assertEqual(loaded.runs[0]["timestamp"], existing)
            datetime.fromisoformat(loaded.runs[1]["timestamp"])
            datetime.fromisoformat(loaded.updated_at)

    def test_update_meta_appends_title_and_read_meta_is_lightweight(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)

            store.append_entry("default", MemoryEntry(kind="message", role="user", content="hello"))
            store.append_run("default", {"user_task": "hello"})
            store.update_meta("default", title="Session title")
            meta = store.read_meta("default")
            loaded = store.load("default")

            self.assertIsNotNone(meta)
            self.assertEqual(meta.session_id, "default")
            self.assertEqual(meta.title, "Session title")
            self.assertEqual(meta.entry_count, 1)
            datetime.fromisoformat(meta.created_at)
            datetime.fromisoformat(meta.updated_at)
            self.assertEqual(loaded.title, "Session title")

    def test_read_meta_returns_none_for_missing_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)

            self.assertIsNone(store.read_meta("missing"))

    def test_build_compaction_entry(self):
        entry = build_compaction_entry("summary")

        self.assertEqual(entry.kind, "compaction")
        self.assertEqual(entry.content, "summary")

    def test_compaction_entry_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)

            store.append_entry("default", build_compaction_entry("summary", first_kept_entry_id="entry_000003"))
            loaded = store.load("default")

            self.assertEqual(loaded.entries[0].kind, "compaction")
            self.assertEqual(loaded.entries[0].metadata["first_kept_entry_id"], "entry_000003")

    def test_session_recorder_records_run_without_compaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            recorder = SessionRecorder(store)
            state = RunState.start("inspect files")
            state.tool_calls.append(ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"}))
            state.tool_results.append(
                ToolResult(
                    tool_call_id="call_1",
                    tool_name="read_file",
                    ok=True,
                    content="long file content",
                    summary="README summary",
                )
            )
            state.turns = 1
            state.finish("done")

            recorder.record_run(state)
            memory = store.load("default")

            self.assertEqual([entry.kind for entry in memory.entries], ["message", "tool_result", "message"])
            self.assertEqual(memory.entries[1].content, "ok:read_file:call_1:long file content")
            self.assertEqual(memory.entries[1].metadata["summary"], "README summary")
            self.assertNotIn("compaction", [entry.kind for entry in memory.entries])
            self.assertEqual(memory.runs[0]["user_task"], "inspect files")
            self.assertEqual(memory.runs[0]["tool_result_count"], 1)
            datetime.fromisoformat(memory.runs[0]["timestamp"])

    def test_migrates_legacy_session_to_jsonl_and_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            legacy = store._legacy_json_path("legacy")
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(
                json.dumps(
                    {
                        "session_id": "legacy",
                        "title": None,
                        "entries": [
                            {"kind": "message", "role": "user", "content": "old", "metadata": {}},
                            {"kind": "tool_result", "role": None, "content": "tool", "metadata": {}},
                        ],
                        "runs": [{"user_task": "old task"}],
                        "plan": [],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            loaded = store.load("legacy")
            records = _jsonl_records(store._jsonl_path("legacy"))

            self.assertFalse(legacy.exists())
            self.assertTrue(Path(str(legacy) + ".bak").exists())
            self.assertEqual(records[0]["_t"], "header")
            self.assertEqual(records[0]["version"], 2)
            self.assertEqual([entry.id for entry in loaded.entries], ["entry_000001", "entry_000002"])
            self.assertEqual(loaded.runs[0]["user_task"], "old task")

    def test_migration_preserves_timestamps_and_uses_unique_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            legacy = store._legacy_json_path("legacy")
            legacy.parent.mkdir(parents=True, exist_ok=True)
            Path(str(legacy) + ".bak").write_text("existing backup", encoding="utf-8")
            legacy.write_text(
                json.dumps(
                    {
                        "session_id": "legacy",
                        "title": "Legacy title",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-02T00:00:00+00:00",
                        "workspace_root": str(Path(tmp).resolve()),
                        "entries": [],
                        "runs": [],
                        "plan": [],
                    }
                ),
                encoding="utf-8",
            )

            loaded = store.load("legacy")
            records = _jsonl_records(store._jsonl_path("legacy"))

            self.assertEqual(Path(str(legacy) + ".bak").read_text(encoding="utf-8"), "existing backup")
            self.assertTrue(Path(str(legacy) + ".bak.1").exists())
            self.assertEqual(records[0]["created_at"], "2026-01-01T00:00:00+00:00")
            self.assertEqual(records[-1]["updated_at"], "2026-01-02T00:00:00+00:00")
            self.assertEqual(loaded.title, "Legacy title")

    def test_empty_jsonl_retriggers_legacy_migration(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            legacy = store._legacy_json_path("legacy")
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(
                json.dumps(
                    {
                        "session_id": "legacy",
                        "entries": [{"kind": "message", "role": "user", "content": "old"}],
                    }
                ),
                encoding="utf-8",
            )
            store._jsonl_path("legacy").write_text("", encoding="utf-8")

            loaded = store.load("legacy")

            self.assertEqual(loaded.entries[0].content, "old")
            self.assertFalse(legacy.exists())

    def test_entry_counter_is_instance_scoped(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_store = SessionStore(first)
            second_store = SessionStore(second)

            first_store.append_entry("default", MemoryEntry(kind="message", role="user", content="first"))
            second_store.append_entry("default", MemoryEntry(kind="message", role="user", content="second"))

            self.assertEqual(first_store.load("default").entries[0].id, "entry_000001")
            self.assertEqual(second_store.load("default").entries[0].id, "entry_000001")

    def test_save_resets_counter_before_next_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            for index in range(10):
                store.append_entry("default", MemoryEntry(kind="message", role="user", content=str(index)))
            memory = store.load("default")

            store.save(memory)
            store.append_entry("default", MemoryEntry(kind="message", role="assistant", content="after save"))
            loaded = store.load("default")

            self.assertEqual(loaded.entries[-1].id, "entry_000011")

    def test_load_skips_malformed_tail_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            store.append_entry("default", MemoryEntry(kind="message", role="user", content="ok"))
            path = store._jsonl_path("default")
            with path.open("a", encoding="utf-8") as file:
                file.write('{"_t":"entry","id":"')

            loaded = store.load("default")

            self.assertEqual(len(loaded.entries), 1)
            self.assertEqual(loaded.entries[0].content, "ok")

    def test_entry_ids_continue_after_new_store_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            store.append_entry("default", MemoryEntry(kind="message", role="user", content="one"))
            store.append_entry("default", MemoryEntry(kind="message", role="assistant", content="two"))

            restarted = SessionStore(tmp)
            restarted.append_entry("default", MemoryEntry(kind="message", role="user", content="three"))
            loaded = restarted.load("default")

            self.assertEqual([entry.id for entry in loaded.entries], ["entry_000001", "entry_000002", "entry_000003"])


if __name__ == "__main__":
    unittest.main()
