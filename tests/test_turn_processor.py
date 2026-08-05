import unittest

from forestcode.core.run_state import RunState
from forestcode.core.turn_processor import TurnProcessor
from forestcode.core.types import AssistantTurn, ModelOutput, ReasoningArtifact, ToolCall


class TurnProcessorTest(unittest.TestCase):
    def test_detects_tool_calls_from_assistant_turn(self):
        call = ToolCall(id="call_1", name="list_files", arguments={})
        output = ModelOutput(AssistantTurn(tool_calls=[call]))

        result = TurnProcessor().process(output, RunState.start("inspect"))

        self.assertIsNone(result.final_text)
        self.assertEqual(result.tool_calls, [call])
        self.assertEqual(result.events[0].type, "tool_calls_detected")

    def test_detects_final_text_from_assistant_turn(self):
        output = ModelOutput(AssistantTurn(text="done"))

        result = TurnProcessor().process(output, RunState.start("inspect"))

        self.assertEqual(result.final_text, "done")
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.events[0].type, "final_text_detected")

    def test_empty_assistant_turn_returns_empty_output_event(self):
        output = ModelOutput(AssistantTurn())

        result = TurnProcessor().process(output, RunState.start("inspect"))

        self.assertIsNone(result.final_text)
        self.assertEqual(result.tool_calls, [])
        self.assertEqual(result.events[0].type, "empty_model_output")

    def test_does_not_copy_reasoning_into_turn_result(self):
        artifact = ReasoningArtifact(provider="deepseek", kind="reasoning_content", display_text="thinking")
        output = ModelOutput(AssistantTurn(text="done", reasoning_artifacts=[artifact]))

        result = TurnProcessor().process(output, RunState.start("inspect"))

        self.assertFalse(hasattr(result, "reasoning_artifacts"))


if __name__ == "__main__":
    unittest.main()
