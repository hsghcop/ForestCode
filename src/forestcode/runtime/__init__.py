"""Backend assembly for ForestCode runtimes.

This package is the composition target for any frontend (CLI today, possibly
print/JSON or RPC later). It depends only on backend modules
(core/models/context/tools/memory/plan/config) and must never import
``forestcode.terminal`` or ``forestcode.cli``.
"""

from .factory import (
    ObservedModelClient,
    build_agent_loop,
    build_model_client,
    build_model_client_from_env,
)

__all__ = [
    "ObservedModelClient",
    "build_agent_loop",
    "build_model_client",
    "build_model_client_from_env",
]
