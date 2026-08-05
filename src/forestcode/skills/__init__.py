"""Skills runtime: discovery, catalog, on-demand loading, and one-shot selection."""

from .activation import SkillTokenResult, parse_skill_token
from .catalog import (
    CATALOG_HEADER,
    SkillActivationError,
    SkillCatalogContextProvider,
    build_skill_fragments,
    format_catalog,
    skill_body_fragment,
)
from .loader import (
    MAX_BODY_CHARS,
    MAX_CATALOG_SKILLS,
    MAX_DESCRIPTION_CHARS,
    MAX_LOADED_SKILL_CHARS,
    SKILL_RESULT_WRAP_OVERHEAD,
    format_loaded_skill,
)
from .pending import PendingSkillSelection
from .registry import SkillRegistry
from .types import (
    LoadedSkill,
    SkillDescriptor,
    SkillIssue,
    SkillSnapshot,
    SkillSource,
)

__all__ = [
    "CATALOG_HEADER",
    "MAX_BODY_CHARS",
    "MAX_CATALOG_SKILLS",
    "MAX_DESCRIPTION_CHARS",
    "MAX_LOADED_SKILL_CHARS",
    "SKILL_RESULT_WRAP_OVERHEAD",
    "LoadedSkill",
    "PendingSkillSelection",
    "SkillActivationError",
    "SkillCatalogContextProvider",
    "SkillDescriptor",
    "SkillIssue",
    "SkillRegistry",
    "SkillSnapshot",
    "SkillSource",
    "SkillTokenResult",
    "build_skill_fragments",
    "format_catalog",
    "format_loaded_skill",
    "parse_skill_token",
    "skill_body_fragment",
]
