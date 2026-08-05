"""Load and validate the layered ForestCode configuration.

Source tiers:
- A (tuning/model): CLI > process env > .env file > settings.json > default
- S (capability/safety): CLI > process env > settings.json > default  (NOT .env)
- B (legacy capability/safety): CLI > process env > default  (NOT .env)
- C (governance): CLI > default  (no env key)
- D (display): not handled here; the CLI layer passes those directly.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from forestcode.context.types import ContextBudget
from forestcode.envfile import lookup_env, read_env_file
from forestcode.models.config import model_config_from_env
from forestcode.models.types import ModelAdapterError

from .types import (
    AgentRuntimeConfig,
    ConfigError,
    FeatureFlags,
    ForestCodeConfig,
    RuntimeConfig,
)

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}
_DEFAULT = object()


def load_config(
    cli_overrides: Mapping[str, object] | None = None,
    env_file: str | Path | None = ".env",
    settings_file: Path | None | object = _DEFAULT,
) -> ForestCodeConfig:
    cli = cli_overrides or {}

    from forestcode.config.settings_file import default_settings_path, load_settings, settings_to_env_keys

    if settings_file is _DEFAULT:
        settings_path = default_settings_path()
    elif settings_file is None:
        settings_path = None
    else:
        settings_path = Path(settings_file)

    json_values = settings_to_env_keys(load_settings(settings_path))
    env_file_values = read_env_file(env_file)
    file_values = {**json_values, **env_file_values}

    try:
        model = model_config_from_env(lambda name: lookup_env(name, file_values))
    except ModelAdapterError as exc:
        raise ConfigError(str(exc)) from exc

    budget = ContextBudget(
        max_context_chars=_resolve_int(cli, file_values, "max_context_chars", "FORESTCODE_MAX_CONTEXT_CHARS", 40_000, 1),
        max_memory_chars=_resolve_int(cli, file_values, "max_memory_chars", "FORESTCODE_MAX_MEMORY_CHARS", 8_000, 0),
        max_session_summary_chars=_resolve_int(
            cli, file_values, "max_session_summary_chars", "FORESTCODE_MAX_SESSION_SUMMARY_CHARS", 8_000, 0
        ),
        max_recent_messages=_resolve_int(
            cli, file_values, "max_recent_messages", "FORESTCODE_MAX_RECENT_MESSAGES", 20, 0
        ),
        max_tool_result_chars=_resolve_int(
            cli, file_values, "max_tool_result_chars", "FORESTCODE_MAX_TOOL_RESULT_CHARS", 2_000, 0
        ),
        max_plan_chars=_resolve_int(cli, file_values, "max_plan_chars", "FORESTCODE_MAX_PLAN_CHARS", 2_000, 0),
    )

    agent = AgentRuntimeConfig(
        budget=budget,
        runtime=RuntimeConfig(
            max_turns=_resolve_int(cli, file_values, "max_turns", "FORESTCODE_MAX_TURNS", 10, 1),
            compact_trigger_entries=_resolve_int(
                cli, file_values, "compact_trigger_entries", "FORESTCODE_COMPACT_TRIGGER_ENTRIES", 40, 1
            ),
            auto_compact=_resolve_bool_a(cli, file_values, "auto_compact", "FORESTCODE_AUTO_COMPACT", True),
        ),
        features=FeatureFlags(
            enable_command_tools=_resolve_bool_s(
                cli, json_values, "enable_command_tools", "FORESTCODE_ENABLE_COMMAND_TOOLS", False
            ),
            include_project_rules=_resolve_bool_cli(cli, "include_project_rules", True),
            include_long_term_memory=_resolve_bool_cli(cli, "include_long_term_memory", True),
        ),
        tool_output_max_chars=_resolve_int(
            cli, file_values, "tool_output_max_chars", "FORESTCODE_TOOL_OUTPUT_MAX_CHARS", 20_000, 1
        ),
        reasoning_display_max_chars=_resolve_int(
            cli, file_values, "reasoning_display_max_chars", "FORESTCODE_REASONING_DISPLAY_MAX_CHARS", 2_000, 1
        ),
    )

    return ForestCodeConfig(model=model, agent=agent)


def _resolve_int(
    cli: Mapping[str, object],
    file_values: dict[str, str],
    cli_key: str,
    env_key: str,
    default: int,
    minimum: int,
) -> int:
    """A-layer int: CLI > process env > .env file > default, with min check."""
    raw = cli.get(cli_key)
    if raw is None:
        env_raw = lookup_env(env_key, file_values)
        if env_raw is None or not str(env_raw).strip():
            return default
        raw = env_raw
    return _coerce_int(env_key, raw, minimum)


def _coerce_int(name: str, raw: object, minimum: int) -> int:
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be an integer (got {raw!r})") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum} (got {value})")
    return value


def _resolve_bool_b(cli: Mapping[str, object], cli_key: str, env_key: str, default: bool) -> bool:
    """B-layer bool: CLI > process env (NOT .env) > default, strict parse."""
    val = cli.get(cli_key)
    if val is not None:
        return bool(val)
    env_raw = os.getenv(env_key)
    if env_raw is None or not env_raw.strip():
        return default
    return _parse_strict_bool(env_key, env_raw)


def _resolve_bool_s(
    cli: Mapping[str, object],
    json_values: dict[str, str],
    cli_key: str,
    env_key: str,
    default: bool,
) -> bool:
    """S-layer bool: CLI > process env > settings.json > default, strict parse."""
    val = cli.get(cli_key)
    if val is not None:
        return bool(val)
    env_raw = os.getenv(env_key)
    if env_raw is not None and env_raw.strip():
        return _parse_strict_bool(env_key, env_raw)
    json_raw = json_values.get(env_key)
    if json_raw is None or not str(json_raw).strip():
        return default
    return _parse_strict_bool(env_key, str(json_raw))


def _resolve_bool_a(
    cli: Mapping[str, object],
    file_values: dict[str, str],
    cli_key: str,
    env_key: str,
    default: bool,
) -> bool:
    """A-layer bool: CLI > process env > .env file > default, strict parse."""
    val = cli.get(cli_key)
    if val is not None:
        return bool(val)
    env_raw = lookup_env(env_key, file_values)
    if env_raw is None or not str(env_raw).strip():
        return default
    return _parse_strict_bool(env_key, str(env_raw))


def _parse_strict_bool(name: str, raw: str) -> bool:
    token = raw.strip().lower()
    if token in _TRUTHY:
        return True
    if token in _FALSY:
        return False
    raise ConfigError(f"{name} must be one of 1/true/yes/on or 0/false/no/off (got {raw!r})")


def _resolve_bool_cli(cli: Mapping[str, object], cli_key: str, default: bool) -> bool:
    """C-layer bool: CLI > default (no env key)."""
    val = cli.get(cli_key)
    return default if val is None else bool(val)
