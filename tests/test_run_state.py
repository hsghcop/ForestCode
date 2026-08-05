import unittest

from forestcode.core.run_state import RunState
from forestcode.core.types import (
    MAX_TOOL_CALL_ID_CHARS,
    AssistantTurn,
    ReasoningArtifact,
    ToolCall,
    ToolResult,
)


class RunStateTest(unittest.TestCase):
    def test_tool_call_id_has_shared_runtime_bound(self):
        valid = ToolCall(
            id="c" * MAX_TOOL_CALL_ID_CHARS,
            name="list_files",
            arguments={},
        )
        self.assertEqual(len(valid.id), MAX_TOOL_CALL_ID_CHARS)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            ToolCall(
                id="c" * (MAX_TOOL_CALL_ID_CHARS + 1),
                name="list_files",
                arguments={},
            )

    def test_add_tool_calls_records_one_assistant_message_for_batch(self):
        state = RunState.start("inspect")
        calls = [
            ToolCall(id="call_1", name="list_files", arguments={"path": "."}),
            ToolCall(id="call_2", name="read_file", arguments={"path": "README.md"}),
        ]

        state.add_tool_calls(calls)

        self.assertEqual(state.tool_calls, calls)
        self.assertEqual(len(state.messages), 2)
        assistant_message = state.messages[-1]
        self.assertEqual(assistant_message.role, "assistant")
        self.assertIsNone(assistant_message.content)
        self.assertEqual(assistant_message.tool_calls, calls)

    def test_add_assistant_turn_records_reasoning_with_tool_calls(self):
        state = RunState.start("inspect")
        artifact = ReasoningArtifact(
            provider="deepseek",
            kind="reasoning_content",
            payload={"reasoning_content": "thinking"},
            required_for_followup=True,
            visible=True,
            display_text="thinking",
        )
        call = ToolCall(id="call_1", name="list_files", arguments={})

        state.add_assistant_turn(AssistantTurn(tool_calls=[call], reasoning_artifacts=[artifact]))

        assistant_message = state.messages[-1]
        self.assertEqual(assistant_message.tool_calls, [call])
        self.assertEqual(assistant_message.reasoning_artifacts, [artifact])

    def test_add_assistant_turn_records_reasoning_with_text(self):
        state = RunState.start("hello")
        artifact = ReasoningArtifact(provider="deepseek", kind="reasoning_content", display_text="thinking")

        state.add_assistant_turn(AssistantTurn(text="hi", reasoning_artifacts=[artifact]))

        assistant_message = state.messages[-1]
        self.assertEqual(assistant_message.content, "hi")
        self.assertEqual(assistant_message.reasoning_artifacts, [artifact])

    def test_add_tool_result_records_tool_call_id(self):
        state = RunState.start("inspect")

        state.add_tool_result(
            ToolResult(
                tool_call_id="call_1",
                tool_name="list_files",
                ok=True,
                content="file README.md",
            )
        )

        message = state.messages[-1]
        self.assertEqual(message.role, "tool_result")
        self.assertEqual(message.tool_call_id, "call_1")
        self.assertIn("file README.md", message.content or "")


if __name__ == "__main__":
    unittest.main()
