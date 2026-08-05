"""Load model configuration from the process environment."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from forestcode.envfile import lookup_env, read_env_file

from .types import ModelAdapterError, ModelConfig

EnvLookup = Callable[[str], "str | None"]


def load_model_config_from_env(env_file: str | Path | None = ".env") -> ModelConfig:
    """Backward-compatible entry: read ``.env`` + process env, build ModelConfig.

    Kept for compatibility and unit tests. ``config.loader`` constructs the
    same ``ModelConfig`` via :func:`model_config_from_env` against its own
    merged env source.
    """
    file_values = read_env_file(env_file)
    return model_config_from_env(lambda name: lookup_env(name, file_values))


def model_config_from_env(lookup: EnvLookup) -> ModelConfig:
    """Build a ModelConfig from an env lookup (process>file already merged)."""
    return ModelConfig(
        api_type=_optional_env("FORESTCODE_API_TYPE", "openai-compatible", lookup),
        model=_required_env("FORESTCODE_MODEL", lookup),
        base_url=_required_env("FORESTCODE_BASE_URL", lookup),
        api_key=_required_env("FORESTCODE_API_KEY", lookup),
        timeout=_timeout_from_env(lookup),
        reasoning_mode=_optional_nullable_env("FORESTCODE_REASONING_MODE", lookup),
        reasoning_effort=_optional_nullable_env("FORESTCODE_REASONING_EFFORT", lookup),
    )


def _required_env(name: str, lookup: EnvLookup) -> str:
    value = lookup(name)
    if value is None or not value.strip():
        raise ModelAdapterError(f"missing required environment variable: {name}")
    return value.strip()


def _optional_env(name: str, default: str, lookup: EnvLookup) -> str:
    value = lookup(name)
    if value is None or not value.strip():
        return default
    return value.strip()


def _optional_nullable_env(name: str, lookup: EnvLookup) -> str | None:
    value = lookup(name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _timeout_from_env(lookup: EnvLookup) -> float:
    raw = lookup("FORESTCODE_TIMEOUT") or "60"
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ModelAdapterError("FORESTCODE_TIMEOUT must be a number") from exc

    if timeout <= 0:
        raise ModelAdapterError("FORESTCODE_TIMEOUT must be greater than 0")
    return timeout
