"""Read and write the global ForestCode settings file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .types import ConfigError

_KEY_TO_ENV = {
    "api_key": "FORESTCODE_API_KEY",
    "model": "FORESTCODE_MODEL",
    "base_url": "FORESTCODE_BASE_URL",
    "api_type": "FORESTCODE_API_TYPE",
    "timeout": "FORESTCODE_TIMEOUT",
    "reasoning_mode": "FORESTCODE_REASONING_MODE",
    "reasoning_effort": "FORESTCODE_REASONING_EFFORT",
    "max_turns": "FORESTCODE_MAX_TURNS",
    "auto_compact": "FORESTCODE_AUTO_COMPACT",
    "compact_trigger_entries": "FORESTCODE_COMPACT_TRIGGER_ENTRIES",
    "enable_command_tools": "FORESTCODE_ENABLE_COMMAND_TOOLS",
    "max_context_chars": "FORESTCODE_MAX_CONTEXT_CHARS",
    "max_recent_messages": "FORESTCODE_MAX_RECENT_MESSAGES",
    "max_memory_chars": "FORESTCODE_MAX_MEMORY_CHARS",
    "max_session_summary_chars": "FORESTCODE_MAX_SESSION_SUMMARY_CHARS",
    "max_tool_result_chars": "FORESTCODE_MAX_TOOL_RESULT_CHARS",
    "max_plan_chars": "FORESTCODE_MAX_PLAN_CHARS",
    "tool_output_max_chars": "FORESTCODE_TOOL_OUTPUT_MAX_CHARS",
    "reasoning_display_max_chars": "FORESTCODE_REASONING_DISPLAY_MAX_CHARS",
}

_TEMPLATE = """// NOTE: Only whole-line // comments are supported. Do NOT add inline // comments.
{
  // Required
  "api_key": "",
  "model": "deepseek-v4-pro",
  "base_url": "https://api.deepseek.com",
  "api_type": "deepseek",
  "timeout": 60,

  // Advanced parameters. Uncomment when needed.
  // "auto_compact": true,
  // "compact_trigger_entries": 40,
  // "enable_command_tools": false,

  // Context budget. Usually leave these unchanged.
  // "max_context_chars": 40000,
  // "max_recent_messages": 20,
  // "max_memory_chars": 8000,
  // "max_session_summary_chars": 8000,

  // Agent behavior
  "max_turns": 10
}
"""


def default_settings_path() -> Path:
    return Path.home() / ".forestcode" / "settings.json"


def load_settings(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        return {}

    raw = path.read_text(encoding="utf-8-sig")
    text = _strip_comment_lines(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must be a JSON object, got {type(data).__name__}")
    return data


def settings_to_env_keys(settings: dict[str, object]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key, value in settings.items():
        env_key = _KEY_TO_ENV.get(key)
        if env_key is None:
            continue
        converted = _to_env_value(key, value)
        if converted is None:
            continue
        values[env_key] = converted
    return values


def create_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE, encoding="utf-8")


def _strip_comment_lines(text: str) -> str:
    lines = [line for line in text.splitlines() if not line.strip().startswith("//")]
    return "\n".join(lines)


def _to_env_value(key: str, value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, int | float):
        return str(value)
    raise ConfigError(f"settings key {key!r} must be a scalar value, got {type(value).__name__}")
