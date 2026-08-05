"""Process-local one-shot manual subagent selection."""

from __future__ import annotations

from dataclasses import dataclass

MARKER_TEMPLATE = "[Subagent: {name}]"


@dataclass(slots=True)
class PendingSubagentSelection:
    name: str | None = None

    def replace(self, name: str) -> None:
        self.name = name

    def clear(self) -> None:
        self.name = None

    def marker_text(self) -> str | None:
        if not self.name:
            return None
        return MARKER_TEMPLATE.format(name=self.name)
