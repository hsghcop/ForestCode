import unittest

from forestcode.core.types import (
    MAX_TOOL_CALL_ID_CHARS,
    Message,
    ModelInput,
    ReasoningArtifact,
    ToolCall,
)
from forestcode.models import ModelAdapterError, ModelConfig, OpenAICompatibleAdapter


def build_config() -> ModelConfig:
    return ModelConfig(
        api_type="openai-compatible",
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
        api_key="secret",
    )


class OpenAICompatibleAdapterTest(unittest.TestCase):
    def test_builds_payload_from_messages(self):
        adapter = OpenAICompatibleAdapter()
        model_input = ModelInput(
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
                Message(
                    role="assistant",
                    tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})],
                ),
                Message(role="tool_result", content="FILE README.md\ncontent", tool_call_id="call_1"),
            ]
        )

        payload = adapter._build_payload(build_config(), model_input)

        self.assertEqual(payload["model"], "deepseek-chat")
        self.assertFalse(payload["stream"])
        self.assertEqual(
            payload["messages"],
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "README.md"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "FILE README.md\ncontent"},
            ],
        )

    def test_rejects_tool_result_without_tool_call_id(self):
        adapter = OpenAICompatibleAdapter()
        model_input = ModelInput(messages=[Message(role="tool_result", content="orphan")])

        with self.assertRaisesRegex(ModelAdapterError, "tool_call_id"):
            adapter._build_payload(build_config(), model_input)

    def test_builds_payload_with_system_prompt_and_tools(self):
        adapter = OpenAICompatibleAdapter()
        model_input = ModelInput(
            system_prompt="system rules",
            messages=[Message(role="user", content="hello")],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "description": "Read a file.",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )

        payload = adapter._build_payload(build_config(), model_input)

        self.assertEqual(payload["messages"][0], {"role": "system", "content": "system rules"})
        self.assertEqual(payload["tools"][0]["function"]["name"], "read_file")

    def test_does_not_emit_reasoning_content_for_openai_payload(self):
        adapter = OpenAICompatibleAdapter()
        model_input = ModelInput(
            messages=[
                Message(
                    role="assistant",
                    content="answer",
                    reasoning_artifacts=[
                        ReasoningArtifact(
                            provider="deepseek",
                            kind="reasoning_content",
                            payload={"reasoning_content": "thinking"},
                            required_for_followup=True,
                        )
                    ],
                )
            ]
        )

        payload = adapter._build_payload(build_config(), model_input)

        self.assertNotIn("reasoning_content", payload["messages"][0])

    def test_parses_assistant_text(self):
        adapter = OpenAICompatibleAdapter()

        output = adapter._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "完成",
                        }
                    }
                ]
            }
        )

        self.assertEqual(output.text, "完成")
        self.assertEqual(output.tool_calls, [])

    def test_parses_tool_calls(self):
        adapter = OpenAICompatibleAdapter()

        output = adapter._parse_response(
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "{\"path\":\"README.md\"}",
                                    },
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {
                                        "name": "grep",
                                        "arguments": "{\"pattern\":\"TODO\"}",
                                    },
                                },
                            ],
                        }
                    }
                ]
            }
        )

        self.assertIsNone(output.text)
        self.assertEqual(len(output.tool_calls), 2)
        self.assertEqual(output.tool_calls[0].id, "call_1")
        self.assertEqual(output.tool_calls[0].name, "read_file")
        self.assertEqual(output.tool_calls[0].arguments, {"path": "README.md"})
        self.assertEqual(output.tool_calls[1].name, "grep")

    def test_rejects_invalid_tool_arguments_json(self):
        adapter = OpenAICompatibleAdapter()

        with self.assertRaisesRegex(ModelAdapterError, "valid JSON"):
            adapter._parse_tool_call(
                {
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": "{bad",
                    },
                }
            )

    def test_rejects_tool_call_id_over_runtime_bound(self):
        adapter = OpenAICompatibleAdapter()

        with self.assertRaisesRegex(ModelAdapterError, "tool call id exceeds"):
            adapter._parse_tool_call(
                {
                    "id": "c" * (MAX_TOOL_CALL_ID_CHARS + 1),
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            )

    def test_rejects_tool_arguments_that_are_not_object(self):
        adapter = OpenAICompatibleAdapter()

        with self.assertRaisesRegex(ModelAdapterError, "object"):
            adapter._parse_tool_call(
                {
                    "id": "call_1",
                    "function": {
                        "name": "read_file",
                        "arguments": "[1, 2]",
                    },
                }
            )

    def test_rejects_empty_choices(self):
        adapter = OpenAICompatibleAdapter()

        with self.assertRaisesRegex(ModelAdapterError, "choices"):
            adapter._parse_response({"choices": []})

    def test_rejects_missing_message(self):
        adapter = OpenAICompatibleAdapter()

        with self.assertRaisesRegex(ModelAdapterError, "message"):
            adapter._parse_response({"choices": [{}]})

    def test_complete_uses_injected_transport(self):
        calls = []

        def fake_transport(config, payload):
            calls.append((config, payload))
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "ok",
                            "tool_calls": [],
                        }
                    }
                ]
            }

        config = build_config()
        adapter = OpenAICompatibleAdapter(transport=fake_transport)
        model_input = ModelInput(messages=[Message(role="user", content="hello")])

        output = adapter.complete(config, model_input)

        self.assertEqual(output.text, "ok")
        self.assertEqual(calls[0][0], config)
        self.assertEqual(calls[0][1]["messages"], [{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
