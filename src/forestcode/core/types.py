"""Shared data structures for the core agent runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from forestcode.context.types import ModelInput

from .abort import AbortSignal


MessageRole = Literal["user", "assistant", "tool_result"]
FinishReason = Literal["stop", "length", "tool_use", "error", "aborted"]
MAX_TOOL_CALL_ID_CHARS = 64


@dataclass(slots=True)
class Message:
    role: MessageRole
    content: str | None = None
    tool_calls: list["ToolCall"] = field(default_factory=list)
    tool_call_id: str | None = None
    reasoning_artifacts: list["ReasoningArtifact"] = field(default_factory=list)


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("tool call id must be a non-empty string")
        if len(self.id) > MAX_TOOL_CALL_ID_CHARS:
            raise ValueError(
                f"tool call id exceeds {MAX_TOOL_CALL_ID_CHARS} characters"
            )


@dataclass(slots=True)
class ReasoningArtifact:
    provider: str
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    required_for_followup: bool = False
    visible: bool = False
    display_text: str | None = None


@dataclass(slots=True)
class AssistantTurn:
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    reasoning_artifacts: list[ReasoningArtifact] = field(default_factory=list)
    raw: Any | None = None
    finish_reason: FinishReason | None = None


@dataclass(slots=True, init=False)
class ModelOutput:
    assistant_turn: AssistantTurn

    def __init__(
        self,
        assistant_turn: AssistantTurn | None = None,
        *,
        text: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        reasoning_artifacts: list[ReasoningArtifact] | None = None,
        raw: Any | None = None,
        finish_reason: FinishReason | None = None,
    ) -> None:
        self.assistant_turn = assistant_turn or AssistantTurn(
            text=text,
            tool_calls=list(tool_calls or []),
            reasoning_artifacts=list(reasoning_artifacts or []),
            raw=raw,
            finish_reason=finish_reason,
        )

    @property
    def text(self) -> str | None:
        return self.assistant_turn.text

    @property
    def tool_calls(self) -> list[ToolCall]:
        return self.assistant_turn.tool_calls

    @property
    def reasoning_artifacts(self) -> list[ReasoningArtifact]:
        return self.assistant_turn.reasoning_artifacts

    @property
    def raw(self) -> Any | None:
        return self.assistant_turn.raw

    @property
    def finish_reason(self) -> FinishReason | None:
        return self.assistant_turn.finish_reason


class ModelClient(Protocol):
    def complete(self, model_input: ModelInput, *, abort: AbortSignal | None = None) -> ModelOutput:
        """Return the next model output for the given input.

        ``abort`` is an optional per-turn cancellation token (plan §11-B3).
        Implementations may ignore it (behavior unchanged) or register
        ``abort.on_abort`` to close the connection for mid-request cancellation.
        """


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    ok: bool
    content: str
    error: str | None = None
    summary: str | None = None
    data: dict[str, Any] | None = None


@dataclass(slots=True)
class RunEvent:
    type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnResult:
    final_text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    events: list[RunEvent] = field(default_factory=list)
