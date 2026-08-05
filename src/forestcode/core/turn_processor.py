"""Normalize a model output into an agent turn result."""

from __future__ import annotations

from .run_state import RunState
from .types import ModelOutput, RunEvent, TurnResult


class TurnProcessor:
    def process(self, output: ModelOutput, state: RunState) -> TurnResult:
        turn = output.assistant_turn
        if turn.tool_calls:
            return TurnResult(
                final_text=None,
                tool_calls=list(turn.tool_calls),
                events=[RunEvent("tool_calls_detected", {"count": len(turn.tool_calls)})],
            )

        if turn.text is not None and turn.text.strip():
            return TurnResult(
                final_text=turn.text,
                events=[RunEvent("final_text_detected", {"turn": state.turns})],
            )

        if turn.finish_reason == "stop":
            return TurnResult(
                final_text="",
                tool_calls=[],
                events=[RunEvent("empty_stop", {"turn": state.turns})],
            )

        return TurnResult(
            final_text=None,
            tool_calls=[],
            events=[RunEvent("empty_model_output", {"turn": state.turns, "finish_reason": turn.finish_reason})],
        )
