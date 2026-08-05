"""Model client that routes calls to provider adapters."""

from __future__ import annotations

from forestcode.context import ModelInput
from forestcode.core.abort import AbortSignal
from forestcode.core.types import ModelOutput

from .registry import ProviderRegistry
from .types import ModelConfig


class ModelRouter:
    def __init__(self, config: ModelConfig, registry: ProviderRegistry) -> None:
        self.config = config
        self.registry = registry

    def complete(self, model_input: ModelInput, *, abort: AbortSignal | None = None) -> ModelOutput:
        # abort is accepted for the ModelClient contract (plan §11-B3). The
        # provider-adapter call is unchanged; turn-level cancellation happens at
        # the AgentLoop checkpoints around this call. Mid-request connection
        # close (§7.3) is an optional future refinement at the adapter layer.
        adapter = self.registry.get(self.config.api_type)
        return adapter.complete(self.config, model_input)
