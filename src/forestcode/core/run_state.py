"""Per-run state for a single user task."""

from __future__ import annotations

from dataclasses import dataclass, field

from .types import AssistantTurn, Message, ReasoningArtifact, ToolCall, ToolResult


def is_successful_tool_result(message: Message, tool_name: str) -> bool:
    """Whether a message is a successful tool_result for tool_name.

    Parses the ``ok:{tool}:{id}:{content}`` format produced by
    ``RunState.add_tool_result`` so the string protocol stays in one place.
    """
    if message.role != "tool_result":
        return False
    return (message.content or "").startswith(f"ok:{tool_name}:")


@dataclass(slots=True)
class RunState:
    user_task: str
    messages: list[Message]
    turns: int = 0
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    final_text: str | None = None
    error: str | None = None

    @classmethod
    def start(cls, user_task: str) -> "RunState":
        return cls(
            user_task=user_task,
            messages=[Message(role="user", content=user_task)],
        )

    def add_assistant_text(
        self,
        text: str,
        reasoning_artifacts: list[ReasoningArtifact] | None = None,
    ) -> None:
        self.messages.append(
            Message(
                role="assistant",
                content=text,
                reasoning_artifacts=list(reasoning_artifacts or []),
            )
        )

    def add_tool_call(self, call: ToolCall) -> None:
        self.add_tool_calls([call])

    def add_tool_calls(
        self,
        calls: list[ToolCall],
        content: str | None = None,
        reasoning_artifacts: list[ReasoningArtifact] | None = None,
    ) -> None:
        if not calls:
            return
        self.tool_calls.extend(calls)
        self.messages.append(
            Message(
                role="assistant",
                content=content,
                tool_calls=list(calls),
                reasoning_artifacts=list(reasoning_artifacts or []),
            )
        )

    def add_assistant_turn(self, turn: AssistantTurn) -> None:
        if turn.tool_calls:
            self.add_tool_calls(
                turn.tool_calls,
                content=turn.text,
                reasoning_artifacts=turn.reasoning_artifacts,
            )
            return
        if turn.text is not None:
            self.add_assistant_text(turn.text, reasoning_artifacts=turn.reasoning_artifacts)

    def add_tool_result(self, result: ToolResult) -> None:
        self.tool_results.append(result)
        status = "ok" if result.ok else "error"
        content = result.content if result.ok else (result.error or result.content)
        self.messages.append(
            Message(
                role="tool_result",
                content=f"{status}:{result.tool_name}:{result.tool_call_id}:{content}",
                tool_call_id=result.tool_call_id,
            )
        )

    def finish(self, text: str) -> None:
        self.final_text = text
        self.error = None

    def fail(self, error: str) -> None:
        self.error = error
