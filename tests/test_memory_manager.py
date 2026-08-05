import tempfile
import unittest

from forestcode.core.run_state import RunState
from forestcode.core.types import AssistantTurn, ReasoningArtifact
from forestcode.memory import MemoryManager, SessionRecorder, SessionStore


class MemoryManagerTest(unittest.TestCase):
    def test_records_run_through_session_recorder(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            manager = MemoryManager(SessionRecorder(store, session_id="default"))
            state = RunState.start("hello")
            state.turns = 1
            state.finish("hi")

            manager.record_run(state)
            memory = store.load("default")

            self.assertEqual([entry.role for entry in memory.entries], ["user", "assistant"])
            self.assertEqual(memory.runs[0]["final_text"], "hi")
            self.assertEqual(memory.runs[0]["reasoning_artifact_count"], 0)

    def test_records_reasoning_artifacts_in_metadata_not_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(tmp)
            manager = MemoryManager(SessionRecorder(store, session_id="default"))
            artifact = ReasoningArtifact(
                provider="deepseek",
                kind="reasoning_content",
                payload={"reasoning_content": "thinking"},
                required_for_followup=True,
                visible=True,
                display_text="thinking",
            )
            state = RunState.start("hello")
            state.add_assistant_turn(AssistantTurn(text="hi", reasoning_artifacts=[artifact]))
            state.turns = 1
            state.finish("hi")

            manager.record_run(state)
            memory = store.load("default")

            assistant_entry = memory.entries[-1]
            self.assertEqual(assistant_entry.content, "hi")
            self.assertNotIn("thinking", assistant_entry.content)
            self.assertEqual(assistant_entry.metadata["reasoning_artifacts"][0]["provider"], "deepseek")
            self.assertEqual(memory.runs[0]["reasoning_artifact_count"], 1)

    def test_no_session_recorder_is_noop(self):
        manager = MemoryManager()
        state = RunState.start("hello")

        manager.record_run(state)

        self.assertEqual(state.user_task, "hello")


if __name__ == "__main__":
    unittest.main()
