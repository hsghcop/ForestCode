"""Types produced and consumed by the context layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ModelInput:
    system_prompt: str | None = None
    messages: list[Any] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextBudget:
    max_context_chars: int = 40_000
    max_memory_chars: int = 8_000
    max_session_summary_chars: int = 8_000
    max_recent_messages: int = 20
    max_tool_result_chars: int = 2_000
    max_plan_chars: int = 2_000
    # Skills (F1 budget contract): a loaded skill body is capped at 16,000
    # chars by the loader, and discovery rejects any skill whose formatted
    # text would exceed ``skills.loader.MAX_LOADED_SKILL_CHARS`` (19,000). The
    # current-run tool-result budget must cover that formatted text plus the
    # runtime wrapper ``ok:load_skill:<call_id>:`` (see
    # ``skills.loader.SKILL_RESULT_WRAP_OVERHEAD``) so the model sees the full
    # validated body, description and resource list (AC2) — never a truncated
    # tail. Invariant asserted by tests:
    #   MAX_LOADED_SKILL_CHARS + SKILL_RESULT_WRAP_OVERHEAD
    #       <= max_skill_result_chars
    # Do not change one side without updating the other and the tests.
    max_skill_result_chars: int = 20_000


@dataclass(frozen=True, slots=True)
class ContextFragment:
    """A bounded, transient context fragment (design §Skills runtime).

    Injected into model input for the current run only; never persisted.
    ``kind`` is a generic source type (e.g. ``skills_catalog``, ``skill``);
    ``label``/``content`` must not carry absolute paths or skill paths.
    """

    kind: str
    label: str
    content: str


@dataclass(slots=True)
class ContextRequest:
    workspace_root: str = "."
    session_id: str | None = None
    include_project_rules: bool = True
    include_long_term_memory: bool = True
    # Transient fragments (skills catalog / manually activated skill body) that
    # live only in the model input of this run, never in the session store.
    transient_fragments: tuple[ContextFragment, ...] = ()
