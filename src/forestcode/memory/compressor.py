"""Session compaction helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .compaction import effective_compactions
from .session_store import SessionStore
from .types import MemoryEntry, SessionMemory


SUMMARY_SECTION_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL | re.IGNORECASE)
ANALYSIS_SECTION_RE = re.compile(r"<analysis>.*?</analysis>", re.DOTALL | re.IGNORECASE)


class Summarizer(Protocol):
    def summarize(self, entries: list[MemoryEntry], prior_summary: str | None) -> str:
        """Summarize the provided session entries."""


@dataclass(slots=True)
class _CompactionPlan:
    entries: list[MemoryEntry]
    first_kept_entry_id: str
    source_start: str
    source_end: str
    prior_summary: str | None = None
    segments: list[list[MemoryEntry]] | None = None


def build_compaction_entry(
    summary: str,
    *,
    compaction_kind: str = "normal",
    first_kept_entry_id: str | None = None,
    source_start: str | None = None,
    source_end: str | None = None,
    preserved: dict[str, object] | None = None,
) -> MemoryEntry:
    metadata: dict[str, object] = {"compaction_kind": compaction_kind}
    if first_kept_entry_id:
        metadata["first_kept_entry_id"] = first_kept_entry_id
    if source_start:
        metadata["source_start"] = source_start
    if source_end:
        metadata["source_end"] = source_end
    if preserved:
        metadata["preserved"] = preserved
    return MemoryEntry(kind="compaction", content=format_compact_summary(summary), metadata=metadata)


def serialize_conversation(entries: list[MemoryEntry], max_tool_result_chars: int = 2_000) -> str:
    lines: list[str] = []
    for entry in entries:
        content = entry.content or ""
        if entry.kind == "message":
            if entry.role == "assistant":
                label = "Assistant"
            else:
                label = "User"
            lines.append(f"[{label}]: {_truncate(content, max_tool_result_chars)}")
        elif entry.kind == "tool_result":
            lines.append(f"[Tool result]: {_truncate(content, max_tool_result_chars)}")
        elif entry.kind == "compaction":
            kind = str(entry.metadata.get("compaction_kind", "normal"))
            lines.append(f"[Compaction summary: {kind}]: {_truncate(content, max_tool_result_chars)}")
    return "\n\n".join(lines)


def format_compact_summary(text: str) -> str:
    match = SUMMARY_SECTION_RE.search(text)
    if match:
        return match.group(1).strip()
    cleaned = ANALYSIS_SECTION_RE.sub("", text)
    return cleaned.strip()


class SessionCompressor:
    def __init__(
        self,
        store: SessionStore,
        session_id: str,
        summarizer: Summarizer | None,
        *,
        keep_recent_entries: int = 20,
        compact_trigger_entries: int = 40,
        max_summary_chars: int = 8_000,
        max_tool_result_chars: int = 2_000,
        max_consecutive_failures: int = 3,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.summarizer = summarizer
        self.keep_recent_entries = keep_recent_entries
        self.compact_trigger_entries = compact_trigger_entries
        self.max_summary_chars = max_summary_chars
        self.max_tool_result_chars = max_tool_result_chars
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0
        self.disabled = False

    def maybe_normal_compact(self) -> bool:
        if not self._can_summarize():
            return False
        memory = self.store.load(self.session_id)
        plan = self._build_normal_plan(memory)
        if plan is None:
            return False
        return self._run_plan(plan, "normal")

    def maybe_major_compact(self) -> bool:
        if not self._can_summarize():
            return False
        memory = self.store.load(self.session_id)
        plan = self._build_major_plan(memory)
        if plan is None:
            return False
        return self._run_plan(plan, "major")

    def _can_summarize(self) -> bool:
        return self.summarizer is not None and not self.disabled

    def _run_plan(self, plan: _CompactionPlan, compaction_kind: str) -> bool:
        assert self.summarizer is not None
        try:
            summary = self._summarize_plan(plan)
        except Exception:
            self.consecutive_failures += 1
            if self.consecutive_failures >= self.max_consecutive_failures:
                self.disabled = True
            return False

        self.consecutive_failures = 0
        summary = _truncate(format_compact_summary(summary), self.max_summary_chars)
        self.store.append_entry(
            self.session_id,
            build_compaction_entry(
                summary,
                compaction_kind=compaction_kind,
                first_kept_entry_id=plan.first_kept_entry_id,
                source_start=plan.source_start,
                source_end=plan.source_end,
            ),
        )
        return True

    def _summarize_plan(self, plan: _CompactionPlan) -> str:
        assert self.summarizer is not None
        if not plan.segments or len(plan.segments) <= 1:
            return self.summarizer.summarize(plan.entries, plan.prior_summary)

        parts: list[str] = []
        prior = plan.prior_summary
        labels = ["History", "Turn prefix"]
        for index, segment in enumerate(plan.segments):
            label = labels[index] if index < len(labels) else f"Segment {index + 1}"
            segment_summary = format_compact_summary(self.summarizer.summarize(segment, prior))
            parts.append(f"{label}:\n{segment_summary}")
            prior = "\n\n".join(filter(None, [prior, segment_summary]))
        return "\n\n".join(parts)

    def _build_normal_plan(self, memory: SessionMemory) -> _CompactionPlan | None:
        entries = memory.entries
        start_index = self._active_boundary_index(entries)
        visible_indices = self._visible_indices(entries, start_index)
        if len(visible_indices) <= self.compact_trigger_entries:
            return None
        cut_index = self._cut_index(entries, visible_indices)
        if cut_index is None or cut_index <= start_index:
            return None
        summarized = [entry for index, entry in enumerate(entries) if start_index <= index < cut_index and _is_visible(entry)]
        if not summarized:
            return None
        segments = self._split_turn_segments(entries, start_index, cut_index)
        return _CompactionPlan(
            entries=summarized,
            first_kept_entry_id=entries[cut_index].id,
            source_start=summarized[0].id,
            source_end=summarized[-1].id,
            prior_summary=self._latest_effective_summary(entries),
            segments=segments,
        )

    def _build_major_plan(self, memory: SessionMemory) -> _CompactionPlan | None:
        entries = memory.entries
        effective_compactions = self._effective_compactions(entries)
        start_index = self._active_boundary_index(entries)
        visible_indices = self._visible_indices(entries, start_index)
        cut_index = self._cut_index(entries, visible_indices)
        if cut_index is not None and cut_index > start_index:
            raw_entries = [
                entry
                for index, entry in enumerate(entries)
                if start_index <= index < cut_index and _is_visible(entry)
            ]
            summarized = [entry for _index, entry in effective_compactions] + raw_entries
            if raw_entries and summarized:
                return _CompactionPlan(
                    entries=summarized,
                    first_kept_entry_id=entries[cut_index].id,
                    source_start=summarized[0].id,
                    source_end=summarized[-1].id,
                )

        if len(effective_compactions) > 1:
            boundary_id = self._active_boundary_id(entries)
            if boundary_id:
                summarized = [entry for _index, entry in effective_compactions]
                return _CompactionPlan(
                    entries=summarized,
                    first_kept_entry_id=boundary_id,
                    source_start=summarized[0].id,
                    source_end=summarized[-1].id,
                )
        return None

    def _active_boundary_id(self, entries: list[MemoryEntry]) -> str | None:
        for _index, entry in reversed(self._effective_compactions(entries)):
            boundary = entry.metadata.get("first_kept_entry_id")
            if isinstance(boundary, str) and boundary:
                return boundary
        return None

    def _active_boundary_index(self, entries: list[MemoryEntry]) -> int:
        boundary_id = self._active_boundary_id(entries)
        if not boundary_id:
            return 0
        for index, entry in enumerate(entries):
            if entry.id == boundary_id:
                return index
        return 0

    def _latest_effective_summary(self, entries: list[MemoryEntry]) -> str | None:
        for _index, entry in reversed(self._effective_compactions(entries)):
            if entry.content:
                return entry.content
        return None

    def _effective_compactions(self, entries: list[MemoryEntry]) -> list[tuple[int, MemoryEntry]]:
        return effective_compactions(entries)

    def _visible_indices(self, entries: list[MemoryEntry], start_index: int) -> list[int]:
        return [index for index, entry in enumerate(entries) if index >= start_index and _is_visible(entry)]

    def _cut_index(self, entries: list[MemoryEntry], visible_indices: list[int]) -> int | None:
        if self.keep_recent_entries <= 0:
            candidate_pos = len(visible_indices)
        else:
            candidate_pos = len(visible_indices) - self.keep_recent_entries
        if candidate_pos <= 0:
            return None
        candidate_pos = min(candidate_pos, len(visible_indices) - 1)
        while candidate_pos > 0:
            candidate_index = visible_indices[candidate_pos]
            if _is_legal_cut_entry(entries[candidate_index]):
                return candidate_index
            candidate_pos -= 1
        return None

    def _split_turn_segments(
        self,
        entries: list[MemoryEntry],
        start_index: int,
        cut_index: int,
    ) -> list[list[MemoryEntry]] | None:
        if entries[cut_index].role != "assistant":
            return None

        run_start = None
        for index in range(cut_index - 1, start_index - 1, -1):
            entry = entries[index]
            if entry.kind == "message" and entry.role == "user":
                run_start = index
                break
        if run_start is None or run_start <= start_index:
            return None

        history = [entry for index, entry in enumerate(entries) if start_index <= index < run_start and _is_visible(entry)]
        turn_prefix = [entry for index, entry in enumerate(entries) if run_start <= index < cut_index and _is_visible(entry)]
        if not history or not turn_prefix:
            return None
        return [history, turn_prefix]


def _is_visible(entry: MemoryEntry) -> bool:
    return entry.kind in {"message", "tool_result"}


def _is_legal_cut_entry(entry: MemoryEntry) -> bool:
    return entry.kind == "message" and entry.role in {"user", "assistant"}


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    marker = f"\n...<truncated {len(text) - max_chars} chars>"
    keep = max(max_chars - len(marker), 0)
    return text[:keep] + marker
