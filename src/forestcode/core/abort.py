"""Cooperative cancellation primitive for a single agent turn (§11-B3).

Zero-dependency, thread-safe. The backend imports this; it never imports the
frontend. A single ``AbortSignal`` is created per turn by the frontend
``TurnRunner`` and injected into the per-turn orchestrators (``AgentLoop``,
``ToolExecutor``) at construction, and passed as an optional keyword to the
process-level singletons (``ModelClient.complete``, ``CommandService.execute``).

``Aborted`` deliberately subclasses ``BaseException`` (like
``KeyboardInterrupt``) so the codebase's broad ``except Exception`` handlers do
NOT swallow a cancellation — it propagates straight out of ``AgentLoop.run`` to
the worker thread.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class Aborted(BaseException):
    """Control-flow exception raised at a cancellation checkpoint.

    Subclasses ``BaseException`` on purpose so ``except Exception`` does not
    catch it.
    """


class AbortSignal:
    """A thread-safe, idempotent cancellation token with abort callbacks.

    The signal is set from the main (UI) thread on Ctrl+C; checkpoints in the
    backend worker thread poll it via ``raise_if_aborted``. Callbacks registered
    with ``on_abort`` (e.g. close an HTTP response, kill a subprocess) fire once,
    when the signal transitions to set.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[], None]] = []

    def set(self) -> None:
        """Mark the turn as cancelled and fire abort callbacks. Idempotent."""
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks)  # snapshot under lock
        # Run callbacks outside the lock so a callback that touches the signal
        # cannot deadlock; isolate failures so one bad callback does not block
        # the rest.
        for cb in callbacks:
            self._invoke(cb)

    def is_set(self) -> bool:
        return self._event.is_set()

    def raise_if_aborted(self) -> None:
        """Raise ``Aborted`` if the signal has been set. Call at checkpoints."""
        if self._event.is_set():
            raise Aborted()

    def on_abort(self, callback: Callable[[], None]) -> None:
        """Register a callback to run when (or immediately if already) aborted.

        Checking ``is_set`` under the same lock that ``set`` holds when it
        snapshots callbacks closes the race where a concurrent ``set`` could
        otherwise leave a just-appended callback un-invoked.
        """
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)
                return
        self._invoke(callback)

    @staticmethod
    def _invoke(callback: Callable[[], None]) -> None:
        try:
            callback()
        except Exception:  # noqa: BLE001 — one callback must not block the rest
            logger.exception("abort callback failed")
