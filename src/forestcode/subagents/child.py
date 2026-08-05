"""Pure child-context helpers shared by the delegate tool and the child runner.

design §Child Construction and Context / R7:

- ``resolve_child_skill_fragments`` merges the parent run's pre-injected skill
  fragments (catalog + manually activated bodies) with the agent's
  ``default_skills``, deduplicated by *typed* names — never by parsing labels
  or bodies (the design forbids reverse-parsing display text).
- ``combined_context_chars`` computes the hard combined budget: agent
  instructions + parent prompt + all pre-injected Skill bodies. Both the
  delegate tool (pre-enqueue failure) and the child runner (final 兜底 check)
  use the same pure functions so the two can never drift apart.

No threads, no I/O, no mutable state: the factory and the tool layer call these
from different threads with the same fixed snapshot.
"""

from __future__ import annotations

from typing import Protocol

from forestcode.context.types import ContextFragment
from forestcode.core.abort import AbortSignal
from forestcode.skills.catalog import skill_body_fragment
from forestcode.skills.types import SkillSnapshot
from forestcode.tools.types import ApprovalRequest

from .types import AgentConfig


class ConfirmBridgeProtocol(Protocol):
    """The thread-bridged approval surface a child run needs (design §Mutation Gate).

    ``ConfirmProxy`` (terminal/turn_runner.py) satisfies this protocol: the
    callable bridges one approval request to the main thread (optionally tagged
    with a child task id), and ``cancel_task`` resolves only that child's
    current/queued tickets to ABORTED (R6). Defined here so the factory and the
    coordinator can depend on the narrow contract instead of the terminal layer.
    """

    def __call__(
        self,
        request: ApprovalRequest,
        *,
        task_id: str | None = None,
        abort: AbortSignal | None = None,
    ) -> bool: ...

    def cancel_task(self, task_id: str) -> None: ...


def combined_context_chars(
    instructions: str,
    prompt: str,
    fragments: tuple[ContextFragment, ...],
) -> int:
    """Total chars of instructions + prompt + every fragment's content.

    The catalog fragment is included (it is part of the pre-injected context),
    matching design §Child Construction and Context: the combined budget covers
    agent instructions, the parent task prompt and all pre-injected Skill
    bodies. Exceeding ``MAX_COMBINED_CONTEXT_CHARS`` fails the delegation —
    never truncates.
    """
    return (
        len(instructions)
        + len(prompt)
        + sum(len(fragment.content) for fragment in fragments)
    )


def resolve_child_skill_fragments(
    agent_config: AgentConfig,
    skills_snapshot: SkillSnapshot | None,
    activated_skill_names: tuple[str, ...],
    inherited_fragments: tuple[ContextFragment, ...],
) -> tuple[ContextFragment, ...]:
    """Merge inherited (parent) fragments with the agent's default skills.

    ``inherited_fragments`` are the parent run's pre-injected fragments (the
    skills catalog plus manually activated bodies); they are kept as-is so the
    child can call ``load_skill`` against the same fixed snapshot. Every
    ``default_skills`` name not already activated is loaded from the fixed
    snapshot and appended as a body fragment (design §Child Construction and
    Context, dedup by name). A missing/invalid default skill makes this agent
    invalid for the run: raise a diagnostic ``ValueError`` instead of silently
    dropping the skill.
    """
    fragments = list(inherited_fragments)
    activated = set(activated_skill_names)
    for name in agent_config.default_skills:
        if name in activated:
            continue
        if skills_snapshot is None:
            raise ValueError(
                f"agent {agent_config.name!r}: default skill {name!r} is not "
                "available (no skill snapshot for this run)"
            )
        loaded = skills_snapshot.load(name)
        if loaded is None:
            raise ValueError(
                f"agent {agent_config.name!r}: default skill {name!r} is missing "
                "or invalid in the fixed skill snapshot"
            )
        fragments.append(skill_body_fragment(loaded))
        activated.add(name)
    return tuple(fragments)
