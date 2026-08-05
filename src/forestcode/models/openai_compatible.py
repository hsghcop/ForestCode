"""OpenAI-compatible non-streaming chat completions adapter."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable

from forestcode.context import ModelInput
from forestcode.core.types import (
    MAX_TOOL_CALL_ID_CHARS,
    AssistantTurn,
    FinishReason,
    Message,
    ModelOutput,
    ToolCall,
)

from .types import ModelAdapterError, ModelConfig


Transport = Callable[[ModelConfig, dict[str, Any]], dict[str, Any]]

logger = logging.getLogger(__name__)

_OPENAI_FINISH_MAP: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "error",
}


class OpenAICompatibleAdapter:
    def __init__(self, transport: Transport | None = None) -> None:
        self.transport = transport or self._default_transport

    def complete(self, config: ModelConfig, model_input: ModelInput) -> ModelOutput:
        payload = self._build_payload(config, model_input)
        raw = self.transport(config, payload)
        return self._parse_response(raw)

    def _build_payload(self, config: ModelConfig, model_input: ModelInput) -> dict[str, Any]:
        messages = [self._convert_message(message) for message in model_input.messages]
        if model_input.system_prompt:
            messages.insert(0, {"role": "system", "content": model_input.system_prompt})

        payload: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            "stream": False,
        }
        if model_input.tools:
            payload["tools"] = model_input.tools
        return payload

    def _convert_message(self, message: Message) -> dict[str, Any]:
        if message.role == "user":
            return {"role": "user", "content": message.content}
        if message.role == "assistant":
            if message.tool_calls:
                return {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [self._convert_tool_call(call) for call in message.tool_calls],
                }
            return {"role": "assistant", "content": message.content}
        if message.role == "tool_result":
            if not message.tool_call_id:
                raise ModelAdapterError("tool_result message missing tool_call_id")
            return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}

        raise ModelAdapterError(f"unsupported message role: {message.role}")

    def _convert_tool_call(self, call: ToolCall) -> dict[str, Any]:
        return {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": json.dumps(call.arguments, ensure_ascii=False),
            },
        }

    def _parse_response(self, raw: dict[str, Any]) -> ModelOutput:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelAdapterError("model response missing non-empty choices")

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise ModelAdapterError("model response choice must be an object")

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise ModelAdapterError("model response choice missing message object")

        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ModelAdapterError("model response message.content must be a string or null")

        raw_tool_calls = message.get("tool_calls") or []
        if not isinstance(raw_tool_calls, list):
            raise ModelAdapterError("model response message.tool_calls must be a list")

        tool_calls = [self._parse_tool_call(raw_call) for raw_call in raw_tool_calls]
        finish_reason = self._map_finish_reason(first_choice.get("finish_reason"))
        return ModelOutput(AssistantTurn(text=content, tool_calls=tool_calls, raw=raw, finish_reason=finish_reason))

    def _map_finish_reason(self, raw: Any) -> FinishReason | None:
        if not isinstance(raw, str) or not raw:
            return None
        mapped = _OPENAI_FINISH_MAP.get(raw)
        if mapped is None:
            logger.warning("unknown openai finish_reason: %s", raw)
            return None
        return mapped

    def _parse_tool_call(self, raw_call: dict[str, Any]) -> ToolCall:
        if not isinstance(raw_call, dict):
            raise ModelAdapterError("tool call must be an object")

        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            raise ModelAdapterError("tool call missing id")
        if len(call_id) > MAX_TOOL_CALL_ID_CHARS:
            raise ModelAdapterError(
                f"tool call id exceeds {MAX_TOOL_CALL_ID_CHARS} characters"
            )

        function = raw_call.get("function")
        if not isinstance(function, dict):
            raise ModelAdapterError("tool call missing function object")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ModelAdapterError("tool call function missing name")

        raw_arguments = function.get("arguments", "{}")
        if raw_arguments is None or raw_arguments == "":
            arguments: Any = {}
        elif isinstance(raw_arguments, str):
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ModelAdapterError("tool call function.arguments must be valid JSON") from exc
        else:
            raise ModelAdapterError("tool call function.arguments must be a JSON string")

        if not isinstance(arguments, dict):
            raise ModelAdapterError("tool call function.arguments must decode to an object")

        return ToolCall(id=call_id, name=name, arguments=arguments)

    def _default_transport(self, config: ModelConfig, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{config.base_url.rstrip('/')}/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=body,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=config.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ModelAdapterError(f"model HTTP request failed with {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ModelAdapterError(f"model HTTP request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise ModelAdapterError("model HTTP request timed out") from exc

        try:
            raw = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise ModelAdapterError("model HTTP response was not valid JSON") from exc

        if not isinstance(raw, dict):
            raise ModelAdapterError("model HTTP response JSON must be an object")
        return raw
