"""Registry for model provider adapters."""

from __future__ import annotations

from .types import ModelAdapterError, ProviderAdapter


class ProviderRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ProviderAdapter] = {}

    def register(self, api_type: str, adapter: ProviderAdapter) -> None:
        key = api_type.strip()
        if not key:
            raise ModelAdapterError("api_type cannot be empty")
        self._adapters[key] = adapter

    def get(self, api_type: str) -> ProviderAdapter:
        key = api_type.strip()
        try:
            return self._adapters[key]
        except KeyError as exc:
            raise ModelAdapterError(f"no provider adapter registered for api_type: {key}") from exc
