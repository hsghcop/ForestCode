"""Pure ``$skill-name`` token parsing (PRD R4, design §``$name`` parsing).

Only a leading token matching ``^\\$([a-z0-9][a-z0-9_-]{0,63})(?:\\s+|$)`` is
consumed; everything else is left untouched as the task text. A token with no
task after it is a user error (no model call).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SKILL_TOKEN_RE = re.compile(r"^\$([a-z0-9][a-z0-9_-]{0,63})(?:\s+|$)")


@dataclass(frozen=True, slots=True)
class SkillTokenResult:
    name: str | None  # explicit skill name when a valid token was parsed
    task: str | None  # remaining task text (None when a user error occurred)
    error: str | None  # user-facing error, or None


def parse_skill_token(text: str) -> SkillTokenResult:
    match = SKILL_TOKEN_RE.match(text)
    if match is None:
        return SkillTokenResult(name=None, task=text, error=None)
    name = match.group(1)
    rest = text[match.end() :].strip()
    if not rest:
        return SkillTokenResult(
            name=None,
            task=None,
            error="Skills> $<skill-name> requires a task after the skill name",
        )
    return SkillTokenResult(name=name, task=rest, error=None)
