"""Single-writer mutation gate shared by a parent run and its children (design §Mutation Gate).

Patch-first apply, ``save_memory`` and ``run_command`` share one gate per parent
run: at most one mutation section — propose -> approval -> apply/execute — runs
at any moment across the parent and all concurrent children. Reads and ordinary
state tools never acquire it, so multiple agents can keep reading in parallel.

The gate serializes the *whole* section (not just apply) so a proposal's diff is
computed against the file state that will actually be present at approval/apply
time. Combined with the existing content-hash stale-write check in
``PatchService.apply``, a second concurrent writer cannot silently clobber the
first one.
"""

from __future__ import annotations

import threading


class MutationGate:
    """A thread-safe context manager guarding one mutation section.

    Reentrant for the same thread (the parent tool chain may nest a command
    inside a patch section); blocking across threads. ``task_id`` is accepted
    for future diagnostics/debugging only — the gate itself does not distinguish
    tasks.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def __enter__(self) -> None:
        self._lock.acquire()

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore[no-untyped-def]
        self._lock.release()
