"""Core runtime primitives for ForestCode."""

from .types import (
    AssistantTurn,
    FinishReason,
    Message,
    ModelClient,
    ModelInput,
    ModelOutput,
    ReasoningArtifact,
    ToolCall,
    ToolResult,
    TurnResult,
)

__all__ = [
    "AgentLoop",
    "AssistantTurn",
    "CallbackEventSink",
    "ContextBuilder",
    "EventSink",
    "FakeModelClient",
    "FinishReason",
    "InMemoryEventSink",
    "MaxTurnsStopPolicy",
    "Message",
    "ModelClient",
    "ModelInput",
    "ModelOutput",
    "ReasoningArtifact",
    "RunRecorder",
    "RunEvent",
    "RunState",
    "StopPolicy",
    "ToolCall",
    "ToolExecutor",
    "ToolResult",
    "TurnProcessor",
    "TurnResult",
]


def __getattr__(name: str):
    if name == "AgentLoop":
        from .agent_loop import AgentLoop

        return AgentLoop
    if name == "RunRecorder":
        from .agent_loop import RunRecorder

        return RunRecorder
    if name == "ContextBuilder":
        from .context_builder import ContextBuilder

        return ContextBuilder
    if name == "FakeModelClient":
        from .fake_model import FakeModelClient

        return FakeModelClient
    if name == "CallbackEventSink":
        from .events import CallbackEventSink

        return CallbackEventSink
    if name == "EventSink":
        from .events import EventSink

        return EventSink
    if name == "InMemoryEventSink":
        from .events import InMemoryEventSink

        return InMemoryEventSink
    if name == "RunEvent":
        from .events import RunEvent

        return RunEvent
    if name == "RunState":
        from .run_state import RunState

        return RunState
    if name == "MaxTurnsStopPolicy":
        from .stop_policy import MaxTurnsStopPolicy

        return MaxTurnsStopPolicy
    if name == "StopPolicy":
        from .stop_policy import StopPolicy

        return StopPolicy
    if name == "ToolExecutor":
        from .tool_executor import ToolExecutor

        return ToolExecutor
    if name == "TurnProcessor":
        from .turn_processor import TurnProcessor

        return TurnProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
