import tempfile
import unittest
from pathlib import Path

from forestcode.core import AssistantTurn, ModelOutput, ToolCall
from forestcode.core.run_state import RunState
from forestcode.core.turn_processor import TurnProcessor
from forestcode.memory import SessionRecorder, SessionStore
from forestcode.models import DeepSeekAdapter, OpenAICompatibleAdapter


class FinishReasonAdapterTest(unittest.TestCase):
    def test_openai_maps_finish_reason(self):
        adapter = OpenAICompatibleAdapter()

        self.assertEqual(adapter._map_finish_reason("stop"), "stop")
        self.assertEqual(adapter._map_finish_reason("tool_calls"), "tool_use")
        self.assertEqual(adapter._map_finish_reason("content_filter"), "error")
        self.assertEqual(adapter._map_finish_reason("function_call"), "tool_use")
        self.assertIsNone(adapter._map_finish_reason(None))
        with self.assertLogs("forestcode.models.openai_compatible", level="WARNING") as logs:
            self.assertIsNone(adapter._map_finish_reason("new_reason"))
        self.assertIn("unknown openai finish_reason", logs.output[0])

    def test_deepseek_maps_finish_reason(self):
        adapter = DeepSeekAdapter()

        self.assertEqual(adapter._map_finish_reason("stop"), "stop")
        self.assertEqual(adapter._map_finish_reason("tool_calls"), "tool_use")
        self.assertEqual(adapter._map_finish_reason("content_filter"), "error")
        self.assertEqual(adapter._map_finish_reason("insufficient_system_resource"), "error")
        self.assertIsNone(adapter._map_finish_reason(None))
        with self.assertLogs("forestcode.models.deepseek_adapter", level="WARNING") as logs:
            self.assertIsNone(adapter._map_finish_reason("new_reason"))
        self.assertIn("unknown deepseek finish_reason", logs.output[0])

    def test_parse_response_preserves_finish_reason_and_missing_field(self):
        adapter = OpenAICompatibleAdapter()

        output = adapter._parse_response(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": ""},
                    }
                ]
            }
        )
        missing = adapter._parse_response({"choices": [{"message": {"role": "assistant", "content": ""}}]})

        self.assertEqual(output.finish_reason, "stop")
        self.assertIsNone(missing.finish_reason)


class FinishReasonTurnProcessorTest(unittest.TestCase):
    def test_tool_calls_do_not_depend_on_finish_reason(self):
        call = ToolCall(id="call_1", name="read_file", arguments={})
        output = ModelOutput(AssistantTurn(tool_calls=[call], finish_reason="stop"))

        result = TurnProcessor().process(output, RunState.start("inspect"))

        self.assertIsNone(result.final_text)
        self.assertEqual(result.tool_calls, [call])

    def test_text_does_not_depend_on_finish_reason(self):
        output = ModelOutput(AssistantTurn(text="done", finish_reason="length"))

        result = TurnProcessor().process(output, RunState.start("inspect"))

        self.assertEqual(result.final_text, "done")
        self.assertEqual(result.tool_calls, [])

    def test_empty_stop_returns_successful_empty_final_text(self):
        output = ModelOutput(AssistantTurn(text="", finish_reason="stop"))

        result = TurnProcessor().process(output, RunState.start("inspect"))

        self.assertEqual(result.final_text, "")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.events[0].type, "empty_stop")

    def test_empty_non_stop_stays_empty_model_output(self):
        for finish_reason in ("length", "error", None):
            with self.subTest(finish_reason=finish_reason):
                output = ModelOutput(AssistantTurn(text="", finish_reason=finish_reason))

                result = TurnProcessor().process(output, RunState.start("inspect"))

                self.assertIsNone(result.final_text)
                self.assertEqual(result.events[0].type, "empty_model_output")
                self.assertEqual(result.events[0].payload["finish_reason"], finish_reason)

    def test_recorder_does_not_write_empty_assistant_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SessionStore(Path(tmp))
            recorder = SessionRecorder(store)
            state = RunState.start("empty stop")
            state.finish("")

            recorder.record_run(state)
            memory = store.load("default")

            self.assertEqual([entry.role for entry in memory.entries], ["user"])
            self.assertEqual(memory.runs[0]["final_text"], "")


if __name__ == "__main__":
    unittest.main()
