"""Deterministic fake model used to test the agent loop."""

from __future__ import annotations

from forestcode.context import ModelInput

from .abort import AbortSignal
from .types import ModelOutput


class FakeModelClient:
    def __init__(self, outputs: list[ModelOutput]) -> None:
        self._outputs = list(outputs)
        self._index = 0
        self.inputs: list[ModelInput] = []

    def complete(self, model_input: ModelInput, *, abort: AbortSignal | None = None) -> ModelOutput:
        self.inputs.append(model_input)
        if self._index >= len(self._outputs):
            return ModelOutput(text="No fake model output configured.")
        output = self._outputs[self._index]
        self._index += 1
        return output
