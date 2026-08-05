"""Shared types for model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from forestcode.context import ModelInput
    from forestcode.core.types import ModelOutput


@dataclass(frozen=True, slots=True)
class ModelConfig:
    api_type: str
    model: str
    base_url: str
    api_key: str = field(repr=False)
    timeout: float = 60.0
    reasoning_mode: str | None = None
    reasoning_effort: str | None = None


class ModelAdapterError(Exception):
    """Raised when a model provider cannot be called or parsed."""


class ProviderAdapter(Protocol):
    def complete(self, config: ModelConfig, model_input: ModelInput) -> ModelOutput:
        """Call a provider using ForestCode's internal model protocol."""
