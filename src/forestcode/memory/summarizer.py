"""Model-backed session summarizer used by compaction."""

from __future__ import annotations

from forestcode.context import ModelInput
from forestcode.core.types import Message, ModelClient

from .compressor import serialize_conversation


class ModelSummarizer:
    def __init__(self, model: ModelClient, max_tool_result_chars: int = 2_000) -> None:
        self.model = model
        self.max_tool_result_chars = max_tool_result_chars

    def summarize(self, entries, prior_summary: str | None) -> str:
        history = serialize_conversation(entries, self.max_tool_result_chars)
        prior = f"Previous summary for background only. Do not repeat it verbatim:\n{prior_summary}\n\n" if prior_summary else ""
        model_input = ModelInput(
            system_prompt=(
                "You summarize ForestCode session history for context compaction. "
                "Do not call tools. Return only text. If <analysis> is used, put the final "
                "answer inside <summary>."
            ),
            messages=[
                Message(
                    role="user",
                    content=(
                        prior
                        + "Summarize only the recent session entries below. Do not repeat prior summary "
                        "content verbatim.\n\n"
                        "Use this structure:\n"
                        "## Goal\n"
                        "## Constraints & Preferences\n"
                        "## Progress\n"
                        "## Key Decisions\n"
                        "## Next Steps\n"
                        "## Critical Context\n\n"
                        "Preserve file paths, commands, decisions, blockers, and exact next actions.\n\n"
                        + history
                    ),
                )
            ],
            tools=[],
        )
        output = self.model.complete(model_input)
        text = output.text
        if text is None or not text.strip():
            raise RuntimeError("compaction summarizer returned empty output")
        return text
