"""Process-local one-shot skill selection (PRD R5/R6, design §Pending selection).

Holds only a skill name; never enters the session store. The input controller
reads it (read-only) to render the non-editable ``[Skill: <name>]``
marker; the bridge owns it and clears it once a run consumes it (or on session
switch). Not a module-level global: instances are owned by the bridge.
"""

from __future__ import annotations

from dataclasses import dataclass

MARKER_TEMPLATE = "[Skill: {name}]"


@dataclass(slots=True)
class PendingSkillSelection:
    name: str | None = None

    def replace(self, name: str) -> None:
        self.name = name

    def clear(self) -> None:
        self.name = None

    def marker_text(self) -> str | None:
        if not self.name:
            return None
        return MARKER_TEMPLATE.format(name=self.name)
