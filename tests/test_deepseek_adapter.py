import unittest

from forestcode.core.types import (
    MAX_TOOL_CALL_ID_CHARS,
    Message,
    ModelInput,
    ReasoningArtifact,
    ToolCall,
)
from forestcode.models import DeepSeekAdapter, ModelAdapterError, ModelConfig


def build_config(**overrides) -> ModelConfig:
    values = {
        "api_type": "deepseek",
        "model": "deepseek-v4-pro",
        "base_url": "https://api.deepseek.com",
        "api_key": "secret",
    }
    values.update(overrides)
    return ModelConfig(**values)


class DeepSeekAdapterTest(unittest.TestCase):
    def test_rejects_tool_call_id_over_runtime_bound(self):
        adapter = DeepSeekAdapter()

        with self.assertRaisesRegex(ModelAdapterError, "tool call id exceeds"):
            adapter._parse_tool_call(
                {
                    "id": "c" * (MAX_TOOL_CALL_ID_CHARS + 1),
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            )

    def test_parses_reasoning_content(self):
        adapter = DeepSeekAdapter()

        output = adapter._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "I should inspect files.",
                            "content": "ok",
                        }
                    }
                ]
            }
        )

        self.assertEqual(output.assistant_turn.text, "ok")
        self.assertEqual(len(output.assistant_turn.reasoning_artifacts), 1)
        artifact = output.assistant_turn.reasoning_artifacts[0]
        self.assertEqual(artifact.provider, "deepseek")
        self.assertEqual(artifact.kind, "reasoning_content")
        self.assertTrue(artifact.required_for_followup)
        self.assertTrue(artifact.visible)
        self.assertEqual(artifact.display_text, "I should inspect files.")

    def test_replays_reasoning_content_on_assistant_text(self):
        adapter = DeepSeekAdapter()
        artifact = ReasoningArtifact(
            provider="deepseek",
            kind="reasoning_content",
            payload={"reasoning_content": "thinking"},
            required_for_followup=True,
        )
        model_input = ModelInput(messages=[Message(role="assistant", content="answer", reasoning_artifacts=[artifact])])

        payload = adapter._build_payload(build_config(), model_input)

        self.assertEqual(
            payload["messages"],
            [{"role": "assistant", "content": "answer", "reasoning_content": "thinking"}],
        )

    def test_replays_reasoning_content_on_assistant_tool_calls(self):
        adapter = DeepSeekAdapter()
        artifact = ReasoningArtifact(
            provider="deepseek",
            kind="reasoning_content",
            payload={"reasoning_content": "need files"},
            required_for_followup=True,
        )
        model_input = ModelInput(
            messages=[
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="call_1", name="list_files", arguments={"path": "."})],
                    reasoning_artifacts=[artifact],
                ),
                Message(role="tool_result", content="ok:list_files:call_1:file README.md", tool_call_id="call_1"),
            ]
        )

        payload = adapter._build_payload(build_config(), model_input)

        assistant = payload["messages"][0]
        self.assertEqual(assistant["reasoning_content"], "need files")
        self.assertEqual(assistant["tool_calls"][0]["function"]["name"], "list_files")
        self.assertEqual(payload["messages"][1]["role"], "tool")

    def test_does_not_emit_reasoning_content_without_deepseek_artifact(self):
        adapter = DeepSeekAdapter()
        model_input = ModelInput(
            messages=[
                Message(
                    role="assistant",
                    content="answer",
                    reasoning_artifacts=[
                        ReasoningArtifact(
                            provider="openai-responses",
                            kind="encrypted_reasoning_item",
                            payload={"encrypted_content": "opaque"},
                            required_for_followup=True,
                        )
                    ],
                )
            ]
        )

        payload = adapter._build_payload(build_config(), model_input)

        self.assertNotIn("reasoning_content", payload["messages"][0])

    def test_raises_when_required_deepseek_artifact_cannot_be_serialized(self):
        adapter = DeepSeekAdapter()
        model_input = ModelInput(
            messages=[
                Message(
                    role="assistant",
                    content="answer",
                    reasoning_artifacts=[
                        ReasoningArtifact(
                            provider="deepseek",
                            kind="reasoning_content",
                            payload={},
                            required_for_followup=True,
                        )
                    ],
                )
            ]
        )

        with self.assertRaisesRegex(ModelAdapterError, "reasoning_content"):
            adapter._build_payload(build_config(), model_input)

    def test_preserves_openai_tool_call_payload_shape(self):
        adapter = DeepSeekAdapter()
        model_input = ModelInput(
            messages=[
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})],
                )
            ]
        )

        payload = adapter._build_payload(build_config(reasoning_effort="high"), model_input)

        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["messages"][0]["tool_calls"][0]["type"], "function")
        self.assertEqual(payload["messages"][0]["tool_calls"][0]["function"]["arguments"], '{"path": "README.md"}')


if __name__ == "__main__":
    unittest.main()
