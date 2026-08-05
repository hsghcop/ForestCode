"""Memory-layer orchestration for run recording."""

from __future__ import annotations

from forestcode.core.run_state import RunState

from .compressor import SessionCompressor
from .recorder import SessionRecorder


class MemoryManager:
    def __init__(self, session_recorder: SessionRecorder | None = None) -> None:
        self.session_recorder = session_recorder

    def record_run(self, state: RunState) -> None:
        if self.session_recorder is not None:
            self.session_recorder.record_run(state)


class SessionCompactionController:
    def __init__(self, compressor: SessionCompressor | None = None, auto_compact: bool = True) -> None:
        self.compressor = compressor
        self.auto_compact = auto_compact

    def maybe_major_compact(self, model_input, state: RunState) -> bool:
        if not self.auto_compact or self.compressor is None:
            return False
        if not bool(model_input.metadata.get("needs_major_compact")):
            return False
        return self.compressor.maybe_major_compact()

    def maybe_normal_compact(self, state: RunState) -> bool:
        if not self.auto_compact or self.compressor is None:
            return False
        return self.compressor.maybe_normal_compact()
