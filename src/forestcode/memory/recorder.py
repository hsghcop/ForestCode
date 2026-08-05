"""Record run facts into append-only session memory."""

from __future__ import annotations

from dataclasses import asdict

from forestcode.core.run_state import RunState
from forestcode.core.types import Message, ReasoningArtifact, ToolResult

from .session_store import SessionStore
from .types import MemoryEntry


class SessionRecorder:
    def __init__(self, store: SessionStore, session_id: str = "default", max_tool_result_store_chars: int = 20_000) -> None:
        self.store = store
        self.session_id = session_id
        self.max_tool_result_store_chars = max_tool_result_store_chars

    def record_run(self, state: RunState) -> None:
        self.store.append_entry(
            self.session_id,
            MemoryEntry(kind="message", role="user", content=state.user_task),
        )

        for result in state.tool_results:
            if result.data and result.data.get("state_only"):
                continue
            self.store.append_entry(
                self.session_id,
                MemoryEntry(
                    kind="tool_result",
                    content=self._tool_result_content(result),
                    metadata={
                        "tool_call_id": result.tool_call_id,
                        "tool_name": result.tool_name,
                        "ok": result.ok,
                        "summary": result.summary,
                    },
                ),
            )

        if state.final_text:
            self.store.append_entry(
                self.session_id,
                MemoryEntry(
                    kind="message",
                    role="assistant",
                    content=state.final_text,
                    metadata=self._assistant_message_metadata(state),
                ),
            )
        elif state.error:
            self.store.append_entry(
                self.session_id,
                MemoryEntry(kind="message", role="assistant", content=f"Run failed: {state.error}"),
            )

        self.store.append_run(
            self.session_id,
            {
                "user_task": state.user_task,
                "turns": state.turns,
                "tool_call_count": len(state.tool_calls),
                "tool_result_count": len(state.tool_results),
                "reasoning_artifact_count": self._reasoning_artifact_count(state.messages),
                "final_text": state.final_text,
                "error": state.error,
            },
        )

    def _tool_result_content(self, result: ToolResult) -> str:
        status = "ok" if result.ok else "error"
        content = result.content if result.ok else (result.error or result.content)
        content = self._truncate(content, self.max_tool_result_store_chars)
        return f"{status}:{result.tool_name}:{result.tool_call_id}:{content}"

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        marker = f"\n...<truncated {len(text) - max_chars} chars>"
        keep = max(max_chars - len(marker), 0)
        return text[:keep] + marker

    @staticmethod
    def _assistant_message_metadata(state: RunState) -> dict[str, object]:
        message = SessionRecorder._last_final_assistant_message(state)
        if message is None or not message.reasoning_artifacts:
            return {}
        return {"reasoning_artifacts": SessionRecorder._serialize_reasoning_artifacts(message.reasoning_artifacts)}

    @staticmethod
    def _last_final_assistant_message(state: RunState) -> Message | None:
        for message in reversed(state.messages):
            if message.role == "assistant" and message.content == state.final_text:
                return message
        return None

    @staticmethod
    def _reasoning_artifact_count(messages: list[Message]) -> int:
        return sum(len(message.reasoning_artifacts) for message in messages)

    @staticmethod
    def _serialize_reasoning_artifacts(artifacts: list[ReasoningArtifact]) -> list[dict[str, object]]:
        return [asdict(artifact) for artifact in artifacts]
