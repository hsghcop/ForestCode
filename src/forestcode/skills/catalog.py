"""Context fragments produced from a SkillSnapshot (design §Context fragments).

The catalog uses the exact fixed format from AC2:

    Available skills (load with load_skill):
    - <name>: <description>

Manual activation produces a separate ``skill`` fragment carrying only the
validated body — no paths, no hashes (PRD R7).
"""

from __future__ import annotations

from forestcode.context.types import ContextFragment

from .types import LoadedSkill, SkillSnapshot

CATALOG_HEADER = "Available skills (load with load_skill):"
FRAGMENT_KIND_CATALOG = "skills_catalog"
FRAGMENT_KIND_SKILL = "skill"


class SkillActivationError(ValueError):
    """A selected Skill existed in the snapshot but could not be loaded."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Skill could not be loaded: {name}")
        self.name = name


def format_catalog(snapshot: SkillSnapshot) -> str:
    lines = [CATALOG_HEADER]
    for descriptor in snapshot.descriptors:  # snapshot keeps name-sorted order
        lines.append(f"- {descriptor.name}: {descriptor.description}")
    return "\n".join(lines)


class SkillCatalogContextProvider:
    """Builds the bounded catalog fragment from a fixed snapshot."""

    def build(self, snapshot: SkillSnapshot) -> ContextFragment:
        return ContextFragment(
            kind=FRAGMENT_KIND_CATALOG,
            label=CATALOG_HEADER,
            content=format_catalog(snapshot),
        )


def skill_body_fragment(loaded: LoadedSkill) -> ContextFragment:
    """Manual-activation fragment: the validated body of one skill."""
    return ContextFragment(
        kind=FRAGMENT_KIND_SKILL,
        label=f"Skill: {loaded.descriptor.name}",
        content=loaded.instructions,
    )


def build_skill_fragments(
    snapshot: SkillSnapshot,
    activation_name: str | None,
) -> tuple[ContextFragment, ...]:
    """Build the catalog fragment plus, for manual activation, the body fragment.

    Shared by the interactive bridge and the legacy chat loop (design §Context
    fragments). The snapshot is fixed for the run; no paths or hashes are
    recorded (PRD R7).
    """
    if snapshot is None or not snapshot.descriptors:
        return ()
    fragments = [SkillCatalogContextProvider().build(snapshot)]
    if activation_name is not None:
        loaded = snapshot.load(activation_name)
        if loaded is None:
            raise SkillActivationError(activation_name)
        fragments.append(skill_body_fragment(loaded))
    return tuple(fragments)
