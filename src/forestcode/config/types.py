"""Configuration aggregate types for ForestCode.

The layering rule:
- ``ForestCodeConfig`` holds the API key and is only constructed/held in
  ``cli.main()``.
- ``AgentRuntimeConfig`` is the key-less slice passed down to run_chat /
  build_agent_loop and the layers below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from forestcode.context.types import ContextBudget
from forestcode.models.types import ModelConfig


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or validated."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    max_turns: int = 10
    compact_trigger_entries: int = 40
    auto_compact: bool = True


@dataclass(frozen=True, slots=True)
class FeatureFlags:
    enable_command_tools: bool = False  # B layer: CLI/process-env, not .env
    include_project_rules: bool = True  # C layer: CLI/default only, no env key
    include_long_term_memory: bool = True  # C layer: CLI/default only, no env key


@dataclass(frozen=True, slots=True)
class AgentRuntimeConfig:
    """Key-less slice passed below the CLI assembly root."""

    budget: ContextBudget = field(default_factory=ContextBudget)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)
    tool_output_max_chars: int = 20_000
    reasoning_display_max_chars: int = 2_000


@dataclass(frozen=True, slots=True)
class ForestCodeConfig:
    """Top-level aggregate. Holds the API key; only ``main()`` should keep it."""

    model: ModelConfig
    agent: AgentRuntimeConfig = field(default_factory=AgentRuntimeConfig)
