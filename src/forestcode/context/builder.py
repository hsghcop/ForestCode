"""Build model input from memory, state, and visible tool context."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from forestcode.core.types import Message

from .budget import truncate_text
from .providers import (
    SessionContextProvider,
    SystemContextProvider,
    TaskContextProvider,
    ToolContextProvider,
    UserContextProvider,
)
from .tool_catalog import ToolCatalog
from .types import ContextBudget, ContextRequest, ModelInput

if TYPE_CHECKING:
    from forestcode.core.run_state import RunState
    from forestcode.plan.types import PlanReader


class ContextBuilder:
    def __init__(
        self,
        workspace_root: str | Path = ".",
        tool_catalog: ToolCatalog | None = None,
        budget: ContextBudget | None = None,
        request: ContextRequest | None = None,
        session_provider: SessionContextProvider | None = None,
        plan_store: PlanReader | None = None,
        system_prefix: str | None = None,
    ) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.budget = budget or ContextBudget()
        self.request = request or ContextRequest(
            workspace_root=str(self.workspace_root)
        )
        self.system_provider = SystemContextProvider()
        self.user_provider = UserContextProvider()
        self.session_provider = session_provider or SessionContextProvider()
        self.tool_provider = ToolContextProvider(tool_catalog)
        self.task_provider = TaskContextProvider(plan_store)
        # Child system instruction (design §Child Construction and Context): the
        # agent instructions become the first system part, before the generic
        # system prompt, so the child's role framing is never buried.
        self.system_prefix = system_prefix

    def build(self, state: RunState) -> ModelInput:
        metadata: dict[str, Any] = {"truncated": False, "context_sources": []}

        selected_tools = self.tool_provider.selected_tool_names()
        system_parts: list[str] = []
        if self.system_prefix is not None and self.system_prefix.strip():
            system_parts.append(self.system_prefix)
        system_parts.append(
            self.system_provider.build(
                self.workspace_root,
                include_memory_instructions="save_memory" in selected_tools,
            )
        )
        if self.request.include_project_rules:
            project_rules = self.user_provider.load_project_rules(self.workspace_root)
            if project_rules:
                system_parts.append("Project rules:\n" + project_rules)
                metadata["context_sources"].append("AGENTS.md")
        # Transient fragments (design §Skills runtime / Context fragments): the
        # bounded skills catalog joins the system prompt; a manually activated
        # skill body becomes its own user message. Both are transient — they live
        # only in this model input, never in RunState or the session store, and
        # metadata records only the generic source type (no names, paths, hashes).
        fragments = self.request.transient_fragments
        for fragment in fragments:
            if fragment.kind == "skills_catalog":
                system_parts.append(fragment.content)
        if fragments:
            metadata["context_sources"].append("skills")
        system_prompt = "\n\n".join(system_parts)

        messages: list[Message] = []
        for fragment in fragments:
            if fragment.kind == "skills_catalog":
                continue
            content = fragment.content
            if fragment.label:
                content = f"{fragment.label}\n{content}"
            messages.append(Message(role="user", content=content))
        if self.request.include_long_term_memory:
            memory_text = self.user_provider.load_memory(self.workspace_root)
            if memory_text:
                compacted = truncate_text(memory_text, self.budget.max_memory_chars)
                messages.append(
                    Message(role="user", content="Long-term memory:\n" + compacted.text)
                )
                metadata["context_sources"].append("MEMORY.md")
                metadata["truncated"] = (
                    bool(metadata["truncated"]) or compacted.truncated
                )

        session_messages, session_metadata, session_truncated = (
            self.session_provider.build_session_context(
                self.request.session_id,
                self.budget,
            )
        )
        if session_messages:
            messages.extend(session_messages)
            session_id = session_metadata.get("session_id")
            if session_id:
                metadata["context_sources"].append(
                    f".forestcode/sessions/{session_id}.jsonl"
                )
            metadata.update(session_metadata)
            metadata["truncated"] = bool(metadata["truncated"]) or session_truncated

        recent = self.session_provider.select_recent_messages(
            state.messages, self.budget
        )
        plan_message, plan_truncated = self.task_provider.build_plan_message(
            recent, self.budget
        )
        if plan_message is not None:
            messages.append(plan_message)
            metadata["context_sources"].append("plan")
            plan_store = self.task_provider.plan_store
            if plan_store is not None:
                metadata["plan_item_count"] = len(plan_store.get())
            metadata["plan_reminder_injected"] = True
            metadata["truncated"] = bool(metadata["truncated"]) or plan_truncated
        messages.extend(recent)

        tools = self.tool_provider.list_tool_schemas()
        metadata["selected_tools"] = selected_tools
        metadata["message_count"] = len(messages)
        metadata["tool_count"] = len(tools)
        metadata["char_count"] = len(system_prompt) + sum(
            self._message_char_count(message) for message in messages
        )
        reserved_for_summary = min(
            self.budget.max_session_summary_chars, self.budget.max_context_chars // 2
        )
        effective_budget = max(self.budget.max_context_chars - reserved_for_summary, 1)
        metadata["compact_effective_budget"] = effective_budget
        metadata["needs_major_compact"] = bool(
            metadata["char_count"] > effective_budget
        )
        if metadata["needs_major_compact"]:
            metadata["compact_reason"] = "context_budget_headroom"
        metadata["truncated"] = bool(metadata["truncated"]) or bool(
            metadata["needs_major_compact"]
        )

        return ModelInput(
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            metadata=metadata,
        )

    def _message_char_count(self, message: Message) -> int:
        content_chars = len(message.content or "")
        tool_call_chars = sum(
            len(str(call.arguments)) + len(call.name) + len(call.id)
            for call in message.tool_calls
        )
        reasoning_chars = sum(
            len(json.dumps(artifact.payload, ensure_ascii=False))
            for artifact in message.reasoning_artifacts
        )
        return content_chars + tool_call_chars + reasoning_chars
