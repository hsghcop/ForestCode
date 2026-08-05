"""User input controller for the interactive CLI (§8.1).

The main-prompt `EOFError` / `KeyboardInterrupt` are handled by
`ForestCodeCliApp` (§16), not here. Confirmation-prompt exceptions have separate
semantics defined in `ConfirmationController` (§8.2).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol


class InputController(Protocol):
    def read_user_input(self, prompt: str) -> str: ...
    def read_confirmation(self, prompt: str) -> str: ...


class StdinInputController:
    """Default controller backed by an input function (``input`` by default).

    ``marker_provider`` (optional) supplies the one-shot skill marker line; it
    is rendered as part of the prompt so it never becomes user text (PRD R5).
    """

    def __init__(
        self,
        input_func: Callable[[str], str] = input,
        *,
        marker_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._input = input_func
        self._marker_provider = marker_provider

    def read_user_input(self, prompt: str) -> str:
        marker = self._marker_provider() if self._marker_provider is not None else None
        full_prompt = f"{marker} {prompt}" if marker else prompt
        return self._input(full_prompt)

    def read_confirmation(self, prompt: str) -> str:
        return self._input(prompt)
