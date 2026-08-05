"""Context providers used by the first-stage context builder."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from forestcode.core.run_state import is_successful_tool_result
from forestcode.core.types import Message
from forestcode.memory.compaction import effective_compactions
from forestcode.memory.retriever import DirectMarkdownMemoryRetriever
from forestcode.memory.session_store import SessionStore
from forestcode.memory.types import MemoryEntry, SessionMemory
from forestcode.plan.types import PlanReader
from forestcode.tools.shell import describe_shell

from .budget import truncate_text
from .tool_catalog import ToolCatalog
from .types import ContextBudget


class SystemContextProvider:
    def build(self, workspace_root: Path, include_memory_instructions: bool = False) -> str:
        lines = [
            "You are ForestCode, a terminal Coding Agent runtime.",
            f"Workspace root: {workspace_root}",
            f"Operating system: {platform.system()} ({sys.platform})",
            f"Shell for run_command: {describe_shell()}",
            "When using run_command, write commands compatible with the shell above.",
            "Use tools when repository facts are needed. Keep changes scoped and explainable.",
        ]
        if include_memory_instructions:
            lines.extend(
                [
                    "You have a save_memory tool to persist information across sessions.",
                    "When to save: user preferences, repeated feedback, role/context facts, project decisions.",
                    "When NOT to save: code patterns, git history, debugging fixes, current task state, "
                    "and never secrets/tokens/passwords/credentials or sensitive personal information.",
                    "Check existing MEMORY.md entries before saving; update rather than duplicate.",
                ]
            )
        return "\n".join(lines)


class UserContextProvider:
    def __init__(self, memory_retriever: DirectMarkdownMemoryRetriever | None = None) -> None:
        self.memory_retriever = memory_retriever

    def load_project_rules(self, workspace_root: Path) -> str:
        path = workspace_root / "AGENTS.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8", errors="replace")

    def load_memory(self, workspace_root: Path) -> str:
        retriever = self.memory_retriever or DirectMarkdownMemoryRetriever(workspace_root)
        return retriever.retrieve()


class SessionContextProvider:
    def __init__(self, session_store: SessionStore | None = None) -> None:
        self.session_store = session_store

    def build_session_context(
        self,
        session_id: str | None,
        budget: ContextBudget,
    ) -> tuple[list[Message], dict[str, object], bool]:
        if self.session_store is None or session_id is None:
            return [], {}, False

        session = self.session_store.load(session_id)
        session_messages, compaction_truncated = self.select_session_messages(session, budget)
        metadata: dict[str, object] = {
            "session_entry_count": len(session.entries),
            "session_id": session.session_id,
        }
        return session_messages, metadata, compaction_truncated

    def select_recent_messages(self, messages: list[Message], budget: ContextBudget) -> list[Message]:
        if budget.max_recent_messages <= 0:
            return []

        groups = self._group_current_run_messages(messages, budget)
        selected: list[Message] = []
        selected_count = 0
        for group in reversed(groups):
            group_count = len(group)
            if selected and selected_count + group_count > budget.max_recent_messages:
                break
            selected = list(group) + selected
            selected_count += group_count

        if messages and messages[0].role == "user" and all(message is not messages[0] for message in selected):
            selected.insert(0, messages[0])

        return selected

    def _group_current_run_messages(self, messages: list[Message], budget: ContextBudget) -> list[list[Message]]:
        groups: list[list[Message]] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if message.role == "assistant" and message.tool_calls:
                tool_call_ids = {call.id for call in message.tool_calls}
                group = [message]
                index += 1
                while index < len(messages):
                    candidate = messages[index]
                    if candidate.role != "tool_result" or candidate.tool_call_id not in tool_call_ids:
                        break
                    group.append(self._compact_current_tool_result(candidate, budget))
                    index += 1
                groups.append(group)
                continue

            if message.role == "tool_result":
                groups.append([self._orphan_tool_result_summary(message, budget)])
                index += 1
                continue

            groups.append([message])
            index += 1
        return groups

    def _compact_current_tool_result(self, message: Message, budget: ContextBudget) -> Message:
        # A loaded skill body must stay fully visible in the current run (AC2):
        # the loader caps the body at 16,000 chars, so the dedicated budget
        # (default 20,000) covers it instead of the generic 2,000 tool-result
        # cap which would truncate it.
        if is_successful_tool_result(message, "load_skill"):
            compacted = truncate_text(message.content or "", budget.max_skill_result_chars)
        else:
            compacted = truncate_text(message.content or "", budget.max_tool_result_chars)
        return Message(role="tool_result", content=compacted.text, tool_call_id=message.tool_call_id)

    def _orphan_tool_result_summary(self, message: Message, budget: ContextBudget) -> Message:
        compacted = truncate_text(message.content or "", budget.max_tool_result_chars)
        return Message(role="user", content="Orphan tool result summary:\n" + compacted.text)

    def select_compaction_messages(self, session: SessionMemory, budget: ContextBudget) -> tuple[list[Message], bool]:
        compactions = [entry for _index, entry in self._effective_compactions(session.entries)]
        return self._compaction_messages(compactions, budget)

    def select_session_messages(self, session: SessionMemory, budget: ContextBudget) -> tuple[list[Message], bool]:
        compactions = [entry for _index, entry in self._effective_compactions(session.entries)]
        compaction_messages, truncated = self._compaction_messages(compactions, budget)
        boundary_index = self._latest_boundary_index(session, compactions)
        if boundary_index is None:
            return compaction_messages + self.select_session_entry_messages(session, budget), truncated

        selected = [
            entry
            for index, entry in enumerate(session.entries)
            if index >= boundary_index and entry.kind in {"message", "tool_result"}
        ]
        return compaction_messages + self._entry_messages(selected, budget), truncated

    def _compaction_messages(self, compactions: list, budget: ContextBudget) -> tuple[list[Message], bool]:
        messages: list[Message] = []
        truncated = False
        for entry in compactions:
            compacted = truncate_text(entry.content, budget.max_session_summary_chars)
            truncated = truncated or compacted.truncated
            messages.append(Message(role="user", content=self._compaction_message_content(entry, compacted.text)))
        return messages, truncated

    def select_session_entry_messages(self, session: SessionMemory, budget: ContextBudget) -> list[Message]:
        if budget.max_recent_messages <= 0:
            return []

        entries = [entry for entry in session.entries if entry.kind in {"message", "tool_result"}]
        selected = entries[-budget.max_recent_messages :]
        return self._entry_messages(selected, budget)

    def _entry_messages(self, selected, budget: ContextBudget) -> list[Message]:
        messages: list[Message] = []
        for entry in selected:
            if entry.kind == "tool_result":
                compacted = truncate_text(entry.content, budget.max_tool_result_chars)
                messages.append(Message(role="user", content="Historical tool result summary:\n" + compacted.text))
                continue
            if entry.role == "assistant":
                messages.append(Message(role="user", content="Historical assistant message:\n" + entry.content))
                continue
            role = entry.role if entry.role in {"user", "assistant"} else "user"
            messages.append(Message(role=role, content=entry.content))
        return messages

    def _effective_compactions(self, entries: list[MemoryEntry]) -> list[tuple[int, MemoryEntry]]:
        return effective_compactions(entries)

    def _latest_boundary_index(self, session: SessionMemory, compactions) -> int | None:
        for entry in reversed(compactions):
            boundary = entry.metadata.get("first_kept_entry_id")
            if not isinstance(boundary, str) or not boundary:
                continue
            for index, candidate in enumerate(session.entries):
                if candidate.id == boundary:
                    return index
        return None

    def _compaction_message_content(self, entry: MemoryEntry, summary: str) -> str:
        if entry.metadata.get("compaction_kind") == "major":
            return (
                "This session is being continued from a previous conversation that ran out of context.\n"
                "The summary below covers the earlier portion of the conversation.\n"
                "\n"
                "Session compaction summary:\n"
                f"{summary}\n\n"
                "Continue from where it left off without asking the user any further questions.\n"
                "Resume directly — do not acknowledge the summary, do not recap."
            )
        return "Session compaction summary:\n" + summary


class TaskContextProvider:
    def __init__(self, plan_store: PlanReader | None = None) -> None:
        self.plan_store = plan_store

    def build_plan_message(
        self, recent_messages: list[Message], budget: ContextBudget
    ) -> tuple[Message | None, bool]:
        if self.plan_store is None:
            return None, False
        items = self.plan_store.get()
        if not items:
            return None, False
        if any(is_successful_tool_result(message, "write_todos") for message in recent_messages):
            return None, False

        lines = ["[Current plan] (reminder)"]
        for item in items:
            mark = {"completed": "x", "in_progress": "~", "pending": " "}[item.status]
            label = item.active_form if item.status == "in_progress" else item.content
            lines.append(f"- [{mark}] {label}")
        compacted = truncate_text("\n".join(lines), budget.max_plan_chars)
        return Message(role="user", content=compacted.text), compacted.truncated


class ToolContextProvider:
    def __init__(self, tool_catalog: ToolCatalog | None = None) -> None:
        self.tool_catalog = tool_catalog or ToolCatalog()

    def list_tool_schemas(self) -> list[dict[str, object]]:
        return self.tool_catalog.list_model_schemas()

    def selected_tool_names(self) -> list[str]:
        return self.tool_catalog.selected_tool_names()
