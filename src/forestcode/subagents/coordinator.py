"""Per-parent-run SubagentCoordinator: FIFO queue, bounded worker pool, wait/cancel/cleanup.

design §Scheduling and Waiting, §Cancellation Limit, §Runtime Types.

The coordinator is per-parent-run mutable state, not a module global. It owns:

- an explicit FIFO ``deque`` of queued task ids plus at most ``max_workers``
  running ``Future``s; a small daemon worker pool only hosts tasks that
  already hold a worker slot (its submit queue never provides FIFO
  semantics — FIFO is the coordinator's explicit deque, per design);
- ``Condition``-based waiting (no busy polling) and a daemon watchdog thread
  that enforces per-task timeouts without polling;
- cooperative cancellation: ``AbortSignal`` per task; ``cancelling`` is
  non-terminal and keeps occupying the slot until the worker actually exits,
  so a non-interruptible sync model request can never free a slot early.

External callbacks (event emission, executor submit, confirm bridge) are
always invoked *outside* the state lock, per design; state transitions are
applied under the lock and produce immutable ``SubagentTaskSnapshot`` values.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from queue import Queue
from types import MappingProxyType
from typing import Protocol

from forestcode.core.abort import Aborted, AbortSignal
from forestcode.core.events import EventSink
from forestcode.core.types import RunEvent

from .types import (
    DEFAULT_TASK_TIMEOUT_SECONDS,
    TERMINAL_STATUSES,
    SubagentCancelReason,
    SubagentRequest,
    SubagentResult,
    SubagentStatus,
    SubagentTaskSnapshot,
    generate_task_id,
    is_terminal_status,
    is_valid_task_id,
    transition_allowed,
)

logger = logging.getLogger(__name__)

# Stable event name consumed by the terminal renderer (design §Terminal and
# Observability); Step 4 adds ``subagent_tool_call_started/finished`` from the
# child EventSink wrapper.
EVENT_SUBAGENT_STATUS_CHANGED = "subagent_status_changed"

# Payload bounds: summaries/errors are excerpts, never full prompts/instructions.
MAX_SUMMARY_CHARS = 200
MAX_ERROR_CHARS = 2_000
# Bounded wait/join budgets (design: never busy-wait, never join forever).
MAX_WAIT_SECONDS = 60.0
CLEANUP_JOIN_SECONDS = 1.5
WATCHDOG_JOIN_SECONDS = 1.0


class CoordinatorClosedError(RuntimeError):
    """``delegate`` was called after ``cleanup``.

    The tool layer converts this into a normal tool failure; the coordinator
    never silently drops a delegation.
    """


class UnknownTaskError(ValueError):
    """``wait`` referenced a task id this run does not know.

    Design §Model Tool Contract: an unknown id is a tool argument error, so the
    coordinator rejects it instead of silently skipping it.
    """


@dataclass(frozen=True, slots=True)
class WaitOutcome:
    """Result of one ``wait`` call (design §Model Tool Contract).

    ``snapshots`` holds the current snapshot of every target task ordered by
    queue position; ``results`` maps task ids to *newly* handed-over final
    results. A repeated wait on an already-delivered task yields no entry.
    """

    timed_out: bool
    snapshots: tuple[SubagentTaskSnapshot, ...]
    results: Mapping[str, SubagentResult]

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))


class ChildRunnerProtocol(Protocol):
    """Runs one child to completion; implemented in Step 4 by the real factory.

    Raising reports a child error (``failed(reason=child_error)``); returning a
    ``SubagentResult`` reports completion. Cancellation is cooperative: the
    runner checks ``abort`` at bounded checkpoints. A request already blocked in
    a synchronous model HTTP call may return late — the coordinator discards
    late results and keeps the slot until this method actually returns.
    """

    def run(
        self, request: SubagentRequest, *, abort: AbortSignal
    ) -> SubagentResult: ...


class _SubmitPending:
    """Slot placeholder between queue-pop and ``executor.submit``.

    Occupies one entry of ``_running`` so the physical worker cap (at most
    ``max_workers`` actual running tasks) holds even during the submit window.
    """

    __slots__ = ()


@dataclass(slots=True)
class _TaskEntry:
    """Mutable per-task state; always accessed under the coordinator state lock."""

    request: SubagentRequest
    abort: AbortSignal = field(default_factory=AbortSignal)
    status: SubagentStatus = "queued"
    reason: SubagentCancelReason | None = None
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    queue_position: int | None = None
    timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS
    deadline: float | None = None
    # Timeout clock pause while the child waits for approval (design: queued and
    # waiting_approval time does not count toward the deadline; it resumes after
    # the approval exits). Set while status == waiting_approval.
    timeout_pause_started_at: float | None = None
    summary: str | None = None
    error: str | None = None
    delivered: bool = False
    result: SubagentResult | None = None


def _excerpt(text: str | None, limit: int) -> str | None:
    """Bound a payload field; ``None`` stays ``None`` (keeps event payloads neutral)."""
    if text is None:
        return None
    if len(text) <= limit:
        return text
    marker = f"\n...<truncated {len(text) - limit} chars>"
    keep = max(limit - len(marker), 0)
    return text[:keep] + marker


class SubagentCoordinator:
    """One parent run's child tasks: delegate / wait / list / cancel / cleanup.

    All mutable state lives under a single ``RLock``; the ``Condition`` built on
    it is the only wake-up channel for waiters and the watchdog (no busy
    polling). The ThreadPoolExecutor is created lazily on first delegate so a
    run that never delegates spawns no threads (Compatibility: the single-agent
    path keeps its current behavior and overhead).
    """

    def __init__(
        self,
        runner: ChildRunnerProtocol,
        *,
        max_workers: int = 6,
        events: EventSink | None = None,
        session_dir: Path | None = None,
        default_wait_timeout: float = 30.0,
        default_task_timeout_seconds: float = DEFAULT_TASK_TIMEOUT_SECONDS,
        id_factory: Callable[[], str] = generate_task_id,
        cleanup_join_seconds: float = CLEANUP_JOIN_SECONDS,
        confirm_cancel: Callable[[str], None] | None = None,
        snapshot_listener: Callable[[tuple[SubagentTaskSnapshot, ...]], None]
        | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if not 0 <= default_wait_timeout <= MAX_WAIT_SECONDS:
            raise ValueError(
                f"default_wait_timeout must be within [0, {MAX_WAIT_SECONDS}] seconds"
            )
        if default_task_timeout_seconds <= 0:
            raise ValueError("default_task_timeout_seconds must be positive")
        if cleanup_join_seconds < 0:
            raise ValueError("cleanup_join_seconds must be non-negative")
        self._runner = runner
        self._max_workers = max_workers
        self._events = events
        # Parent session root for child transcripts; resolved once so events and
        # Step 4's child SessionStore share one canonical path. The coordinator
        # itself never writes transcripts.
        self._session_dir = (
            Path(session_dir).resolve() if session_dir is not None else None
        )
        self._default_wait_timeout = default_wait_timeout
        self._default_task_timeout_seconds = default_task_timeout_seconds
        self._id_factory = id_factory
        self._cleanup_join_seconds = cleanup_join_seconds
        self._confirm_cancel = confirm_cancel
        self._snapshot_listener = snapshot_listener

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        # Serializes slot-filling decisions and executor submits; reentrant so an
        # inline done callback (instant worker) can safely re-enter drain.
        self._submit_lock = threading.RLock()
        self._drain_in_progress = False
        self._tasks: dict[str, _TaskEntry] = {}
        self._pending: deque[str] = deque()
        self._running: dict[str, Future | _SubmitPending] = {}
        self._closed = False
        self._closed_reason: SubagentCancelReason | None = None
        self._next_queue_position = 1
        self._pool: _DaemonWorkerPool | None = None
        self._watchdog: threading.Thread | None = None

    # -- public surface -----------------------------------------------------
    @property
    def max_workers(self) -> int:
        return self._max_workers

    def has_active_children(self) -> bool:
        """True while any task is not yet terminal (queued/running/approval/cancelling).

        Part of the narrow protocol the parent AgentLoop consumes (design
        §Architectural Boundary); the AgentLoop still calls ``cleanup`` on every
        termination path so the watchdog/pool always stop, but this query lets
        callers avoid further work when a run already reached the terminal set.
        """
        with self._lock:
            return any(
                not is_terminal_status(entry.status) for entry in self._tasks.values()
            )

    @property
    def session_dir(self) -> Path | None:
        return self._session_dir

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def delegate(
        self, request: SubagentRequest, *, timeout_seconds: float | None = None
    ) -> SubagentTaskSnapshot:
        """Register one child task and start it if a worker slot is free.

        The task id is always generated by the coordinator's ``id_factory`` and
        validated against the safe charset; any id carried by the incoming
        request is replaced (R10: the model can never choose a task id).
        ``timeout_seconds`` is fixed at enqueue time (the tool layer resolves it
        from the agent config); ``None`` falls back to the coordinator default.
        Raises ``CoordinatorClosedError`` after cleanup.
        """
        task_id = self._id_factory()
        if not is_valid_task_id(task_id):
            raise ValueError(f"id_factory produced invalid task id {task_id!r}")
        if timeout_seconds is None:
            timeout_seconds = self._default_task_timeout_seconds
        if timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {timeout_seconds}")
        with self._lock:
            if self._closed:
                raise CoordinatorClosedError("subagent coordinator is closed")
            if task_id in self._tasks:
                raise ValueError(f"duplicate task id {task_id!r}")
            request = replace(request, task_id=task_id)
            entry = _TaskEntry(
                request=request,
                created_at=time.monotonic(),
                queue_position=self._next_queue_position,
                timeout_seconds=timeout_seconds,
            )
            self._next_queue_position += 1
            self._tasks[task_id] = entry
            self._pending.append(task_id)
            snapshot = self._snapshot(entry)
        self._ensure_worker_threads()
        self._emit_status(snapshot)  # queued
        self._drain()
        return snapshot

    def wait(
        self,
        task_ids: list[str] | None = None,
        timeout: float | None = None,
        abort: AbortSignal | None = None,
    ) -> WaitOutcome:
        """Wait until any target task reaches a terminal state (bounded).

        ``task_ids=None`` selects every not-yet-delivered task; an explicit
        empty list or an unknown id is an argument error (the tool layer
        validates the model's arguments against the same contract). ``timeout``
        is bounded and only ends this wait — it never cancels tasks. Newly
        terminal completed/failed tasks hand their final body over once
        (``delivered`` flips); repeated waits return the snapshot with
        ``delivered=True`` and no result entry. A run with no tasks returns an
        empty outcome and lets the tool layer report the error.
        """
        if timeout is None:
            timeout = self._default_wait_timeout
        if not 0 <= timeout <= MAX_WAIT_SECONDS:
            raise ValueError(
                f"wait timeout must be within [0, {MAX_WAIT_SECONDS}] seconds"
            )
        if abort is not None:
            abort.raise_if_aborted()
            abort.on_abort(self._notify_waiters)
        with self._lock:
            if task_ids is None:
                targets = [
                    tid for tid, entry in self._tasks.items() if not entry.delivered
                ]
                if not targets and self._tasks:
                    # Every task already delivered (repeated wait): still report
                    # the current snapshots so the tool layer can answer with
                    # ``result_omitted`` instead of an empty/timed-out outcome
                    # (design §Model Tool Contract). No waiting, no delivery.
                    targets = list(self._tasks)
            else:
                if not task_ids:
                    raise ValueError("wait task_ids must be a non-empty list")
                targets = list(dict.fromkeys(task_ids))
                unknown = [tid for tid in targets if tid not in self._tasks]
                if unknown:
                    raise UnknownTaskError(f"unknown task id(s): {', '.join(unknown)}")
            if not targets:
                return WaitOutcome(timed_out=True, snapshots=(), results={})
            deadline = time.monotonic() + timeout
            timed_out = False
            while True:
                if abort is not None:
                    abort.raise_if_aborted()
                if any(is_terminal_status(self._tasks[tid].status) for tid in targets):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                self._condition.wait(timeout=remaining)
            # Deliver new terminal results first, then snapshot, so every
            # snapshot returned by this wait already reflects the delivered
            # flag (a repeat wait on the same task reports delivered=True).
            results: dict[str, SubagentResult] = {}
            for tid in targets:
                entry = self._tasks[tid]
                if entry.status in ("completed", "failed") and not entry.delivered:
                    entry.delivered = True
                    if entry.result is not None:
                        results[tid] = replace(entry.result, delivered=True)
                    else:
                        results[tid] = SubagentResult(
                            task_id=tid,
                            agent_name=entry.request.agent_name,
                            delivered=True,
                        )
            snapshots = [
                self._snapshot(self._tasks[tid])
                for tid in sorted(
                    targets, key=lambda t: self._tasks[t].queue_position or 0
                )
            ]
            # Results are ordered by queue position too, so a single wait that
            # delivers several tasks has a stable, predictable order (design:
            # structured snapshots ordered by queue position). Completion order
            # across separate waits is inherently concurrent — FIFO governs
            # start/refill order, not delivery order.
            results = {
                tid: results[tid]
                for tid in sorted(
                    results, key=lambda t: self._tasks[t].queue_position or 0
                )
            }
        return WaitOutcome(
            timed_out=timed_out, snapshots=tuple(snapshots), results=results
        )

    def _notify_waiters(self) -> None:
        """Wake Condition waits when the parent turn is aborted."""
        with self._condition:
            self._condition.notify_all()

    def list(self) -> tuple[SubagentTaskSnapshot, ...]:
        """All task snapshots of this run, ordered by queue position."""
        return self._ordered_snapshots()

    def approval_started(self, task_id: str) -> None:
        """Mark one running child as blocked on the central approval queue.

        The child confirm bridge calls this immediately before enqueueing its
        ticket.  Timeout accounting pauses here; event emission stays outside
        the coordinator lock.  A concurrently cancelled task is left in
        ``cancelling`` and must not be moved back to an active state.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None or entry.status != "running":
                return
            snapshot = self._transition(task_id, "waiting_approval")
        self._emit_status(snapshot)

    def approval_finished(self, task_id: str) -> None:
        """Resume timeout accounting after one child approval resolves.

        Cancellation may resolve the ticket with ``Aborted`` while this call
        unwinds.  In that race the task is already ``cancelling`` and remains
        there until the physical worker exits.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None or entry.status != "waiting_approval":
                return
            snapshot = self._transition(task_id, "running")
        self._emit_status(snapshot)

    def cancel(self, task_id: str) -> SubagentTaskSnapshot | None:
        """Cancel one task; ``None`` for an unknown id (the tool layer errors).

        Queued tasks are removed from the FIFO immediately and reach terminal
        ``cancelled``; running / waiting_approval tasks move to ``cancelling``
        (non-terminal — the worker keeps its slot until it actually exits), get
        their ``AbortSignal`` set, and their approval tickets (Step 3 wiring)
        are released outside the lock. Terminal tasks are idempotent.
        """
        abort_entry: _TaskEntry | None = None
        emitted: SubagentTaskSnapshot | None = None
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry is None:
                return None
            if entry.status == "queued":
                self._pending.remove(task_id)
                emitted = self._transition(task_id, "cancelled", "requested")
            elif entry.status in ("running", "waiting_approval"):
                emitted = self._transition(task_id, "cancelling", "requested")
                abort_entry = entry
            else:
                # Already cancelling or terminal: idempotent, keep the reason.
                return self._snapshot(entry)
        if abort_entry is not None:
            self._fire_abort(abort_entry)
        if emitted is not None:
            self._emit_status(emitted)
        return emitted

    def cleanup(
        self, reason: SubagentCancelReason = "parent_finished"
    ) -> tuple[SubagentTaskSnapshot, ...]:
        """Cancel everything and release the worker pool. Idempotent.

        Stops accepting new tasks, cancels queued tasks immediately, moves
        running / waiting_approval tasks to ``cancelling`` and sets their
        aborts. The join is bounded (``cleanup_join_seconds``): workers stuck in
        a sync model request are not joined to completion; their late results
        are discarded and the executor shuts down non-blocking. Returns the
        final snapshots for the bridge to persist as the recent run.
        """
        to_abort: list[_TaskEntry] = []
        emitted: list[SubagentTaskSnapshot] = []
        with self._submit_lock:
            with self._lock:
                if self._closed:
                    return self._ordered_snapshots()
                self._closed = True
                self._closed_reason = reason
                for task_id in list(self._pending):
                    self._pending.remove(task_id)
                    emitted.append(self._transition(task_id, "cancelled", reason))
                for task_id in list(self._running):
                    entry = self._tasks[task_id]
                    if entry.status in ("running", "waiting_approval"):
                        emitted.append(self._transition(task_id, "cancelling", reason))
                        to_abort.append(entry)
                    elif entry.status == "cancelling":
                        # A watchdog/cancel already declared a reason; keep it.
                        to_abort.append(entry)
                self._condition.notify_all()
            for entry in to_abort:
                self._fire_abort(entry)
            for snapshot in emitted:
                self._emit_status(snapshot)
            # Bounded wait for the workers that honor the abort to actually exit
            # (slot release happens in the done callback). No busy polling.
            deadline = time.monotonic() + self._cleanup_join_seconds
            while True:
                with self._lock:
                    if not self._running:
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    self._condition.wait(timeout=remaining)
            if self._pool is not None:
                self._pool.shutdown()
        self._stop_watchdog()
        return self._ordered_snapshots()

    # -- scheduling ---------------------------------------------------------
    def _drain(self) -> None:
        """Fill free worker slots from the FIFO queue (strict FIFO).

        Serialized by ``_submit_lock``: only one thread decides candidates and
        submits at a time, so a task can never be double-submitted and
        ``len(_running) <= max_workers`` always holds. Executor submit happens
        *outside* the state lock; a submit failure is rolled back through a
        narrow state transition. Reentrant with a guard so an inline done
        callback (instant worker) cannot recurse.
        """
        with self._submit_lock:
            if self._drain_in_progress:
                # The outer drain loop re-checks the freed slot on its next pass.
                return
            self._drain_in_progress = True
            try:
                while True:
                    with self._lock:
                        if (
                            self._closed
                            or len(self._running) >= self._max_workers
                            or not self._pending
                        ):
                            return
                        task_id = self._pending.popleft()
                        entry = self._tasks[task_id]
                        self._running[task_id] = _SubmitPending()
                        snapshot = self._transition(task_id, "running")
                    pool = self._ensure_pool()
                    try:
                        future = pool.submit(
                            self._runner.run, entry.request, abort=entry.abort
                        )
                    except Exception as exc:  # noqa: BLE001 - executor shutdown / creation failure
                        with self._lock:
                            self._running.pop(task_id, None)
                            if entry.status == "cancelling":
                                snapshot = self._transition(task_id, "cancelled")
                            elif self._closed:
                                reason = self._closed_reason
                                if reason is None:
                                    reason = "parent_finished"
                                snapshot = self._transition(
                                    task_id, "cancelled", reason
                                )
                            else:
                                message = str(exc)
                                if not message:
                                    message = exc.__class__.__name__
                                entry.error = _excerpt(message, MAX_ERROR_CHARS)
                                snapshot = self._transition(
                                    task_id, "failed", "child_error"
                                )
                        self._emit_status(snapshot)
                        continue
                    with self._lock:
                        self._running[task_id] = future
                        # Re-check: a concurrent cancel may have moved the task
                        # to cancelling during the submit window; skip the stale
                        # "running" event so per-task order stays consistent.
                        emit_running = entry.status == "running"
                    if emit_running:
                        self._emit_status(snapshot)
                    future.add_done_callback(self._on_child_done)
            finally:
                self._drain_in_progress = False

    def _ensure_worker_threads(self) -> None:
        """Lazily create the worker pool and watchdog (first delegate only)."""
        with self._submit_lock:
            self._ensure_pool()
            if self._watchdog is None:
                watchdog = threading.Thread(
                    target=self._watchdog_loop, name="subagent-watchdog", daemon=True
                )
                self._watchdog = watchdog
                watchdog.start()

    def _ensure_pool(self) -> _DaemonWorkerPool:
        """Lazily create the daemon worker pool and return it (idempotent).

        The pool runs each task on its own daemon thread: a worker stuck in a
        non-interruptible sync model request after a bounded cleanup must never
        hang CLI exit; its late result is discarded anyway (design
        §Cancellation Limit). ``ThreadPoolExecutor`` was not used because its
        workers are non-daemon and it exposes no daemon control on the Python
        versions this project runs on.
        """
        if self._pool is None:
            self._pool = _DaemonWorkerPool(self._max_workers)
        return self._pool

    def _on_child_done(self, future: Future) -> None:
        """Finalize one finished worker: free its slot, pick the terminal state.

        Runs on the worker thread (or inline from ``add_done_callback``). A
        result from a cancelled/closed task is always discarded and a late
        callback never overrides the declared cancel reason. Refill happens via
        ``_drain`` outside the state lock.
        """
        with self._lock:
            task_id = self._task_id_for_future(future)
            if task_id is None:
                return
            entry = self._tasks[task_id]
            self._running.pop(task_id, None)
            try:
                result = future.result()
            except Aborted:
                # The runner hit a cancellation checkpoint: the coordinator has
                # already moved the task to ``cancelling``, so this branch is
                # defensive; the error is never surfaced for cancelled tasks.
                result = None
                error = "cancelled"
            except Exception as exc:  # noqa: BLE001 - worker raised -> child error
                result = None
                message = str(exc)
                if not message:
                    message = exc.__class__.__name__
                error = _excerpt(message, MAX_ERROR_CHARS)
            else:
                error = None
            if entry.status in ("running", "waiting_approval"):
                if entry.status == "waiting_approval":
                    # A worker finishing while its approval is still pending is
                    # anomalous (Step 3 resolves tickets before the child can
                    # finish); normalize through the legal edge first so the
                    # final transition below can never be a guard violation.
                    self._transition(task_id, "running")
                if result is not None:
                    entry.result = result
                    entry.summary = _excerpt(result.final_text, MAX_SUMMARY_CHARS)
                    snapshot = self._transition(task_id, "completed")
                else:
                    entry.error = error
                    entry.summary = _excerpt(error, MAX_SUMMARY_CHARS)
                    snapshot = self._transition(task_id, "failed", "child_error")
            elif entry.status == "cancelling":
                snapshot = self._transition(task_id, "cancelled")
            else:
                snapshot = None  # already terminal: duplicate callback
        if snapshot is not None:
            self._emit_status(snapshot)
        self._drain()

    def _task_id_for_future(self, future: Future) -> str | None:
        """Map a done future back to its task id (identity match; skips sentinels)."""
        for task_id, registered in self._running.items():
            if registered is future:
                return task_id
        return None

    # -- timeout watchdog ---------------------------------------------------
    def _watchdog_loop(self) -> None:
        """Enforce running deadlines without busy polling.

        Waits on the condition until the nearest running deadline (or until
        notified), then moves overdue tasks to ``cancelling(reason=timeout)``
        and sets their aborts. The worker keeps its slot until it actually
        exits; the done callback then finalizes ``cancelled(reason=timeout)``.
        """
        while True:
            with self._lock:
                if self._closed:
                    return
                now = time.monotonic()
                overdue = [
                    entry
                    for entry in self._tasks.values()
                    if entry.status == "running"
                    and entry.deadline is not None
                    and entry.deadline <= now
                ]
                if overdue:
                    emitted = []
                    for entry in overdue:
                        emitted.append(
                            self._transition(
                                entry.request.task_id, "cancelling", "timeout"
                            )
                        )
                    self._condition.notify_all()
                else:
                    next_deadline = min(
                        (
                            entry.deadline
                            for entry in self._tasks.values()
                            if entry.status == "running" and entry.deadline is not None
                        ),
                        default=None,
                    )
                    if next_deadline is None:
                        self._condition.wait()
                    else:
                        self._condition.wait(timeout=max(next_deadline - now, 0.0))
                    continue
            for entry in overdue:
                self._fire_abort(entry)
            for snapshot in emitted:
                self._emit_status(snapshot)

    def _stop_watchdog(self) -> None:
        watchdog = self._watchdog
        self._watchdog = None
        if watchdog is not None:
            watchdog.join(timeout=WATCHDOG_JOIN_SECONDS)
            if watchdog.is_alive():
                logger.warning(
                    "subagent watchdog did not exit in time; leaving daemon thread"
                )

    # -- state machine ------------------------------------------------------
    def _transition(
        self,
        task_id: str,
        to_status: SubagentStatus,
        reason: SubagentCancelReason | None = None,
    ) -> SubagentTaskSnapshot:
        """Apply one validated state transition and build the new snapshot.

        Must be called with the state lock held. Raises ``ValueError`` on an
        illegal transition (a guard violation, e.g. ``queued -> completed``);
        terminal transitions record ``finished_at``, entering ``running``
        records ``started_at`` and the watchdog deadline.
        """
        entry = self._tasks[task_id]
        if not transition_allowed(entry.status, to_status):
            raise ValueError(
                f"subagent task {task_id}: illegal transition "
                f"{entry.status!r} -> {to_status!r}"
            )
        entry.status = to_status
        if to_status == "running":
            if entry.started_at is None:
                entry.started_at = time.monotonic()
                # Timeout clock starts when the task first starts running;
                # queued time is not counted.
                entry.deadline = time.monotonic() + entry.timeout_seconds
            elif entry.timeout_pause_started_at is not None:
                # Resumed from waiting_approval: extend the deadline by the
                # paused interval (approval wait time is not counted).
                if entry.deadline is None:
                    entry.deadline = time.monotonic() + entry.timeout_seconds
                else:
                    entry.deadline += time.monotonic() - entry.timeout_pause_started_at
                entry.timeout_pause_started_at = None
        elif to_status == "waiting_approval" and entry.timeout_pause_started_at is None:
            entry.timeout_pause_started_at = time.monotonic()
        if to_status in TERMINAL_STATUSES:
            entry.finished_at = time.monotonic()
        if reason is not None:
            entry.reason = reason
        self._condition.notify_all()
        return self._snapshot(entry)

    def _snapshot(self, entry: _TaskEntry) -> SubagentTaskSnapshot:
        return SubagentTaskSnapshot(
            task_id=entry.request.task_id,
            agent_name=entry.request.agent_name,
            status=entry.status,
            created_at=entry.created_at,
            started_at=entry.started_at,
            finished_at=entry.finished_at,
            queue_position=entry.queue_position,
            summary=entry.summary,
            error=entry.error,
            cancel_reason=entry.reason,
            delivered=entry.delivered,
        )

    def _ordered_snapshots(self) -> tuple[SubagentTaskSnapshot, ...]:
        with self._lock:
            snapshots = [self._snapshot(entry) for entry in self._tasks.values()]
        snapshots.sort(key=lambda snapshot: snapshot.queue_position or 0)
        return tuple(snapshots)

    # -- external callbacks (always outside the state lock) -----------------
    def _fire_abort(self, entry: _TaskEntry) -> None:
        """Set the child's abort and release its approval tickets (Step 3 wiring).

        Both are external callbacks, so this runs outside the state lock; a
        failing confirm bridge must not break cancellation.
        """
        entry.abort.set()
        if self._confirm_cancel is not None:
            try:
                self._confirm_cancel(entry.request.task_id)
            except Exception:  # one bridge failure must not block cancel
                logger.exception(
                    "confirm_cancel failed for task %s", entry.request.task_id
                )

    def _emit_status(self, snapshot: SubagentTaskSnapshot) -> None:
        """Neutral, bounded status event; never leaks keys or full instructions."""
        payload: dict[str, object] = {
            "task_id": snapshot.task_id,
            "agent_name": snapshot.agent_name,
            "status": snapshot.status,
        }
        if snapshot.cancel_reason is not None:
            payload["reason"] = snapshot.cancel_reason
        if snapshot.summary is not None:
            payload["summary"] = snapshot.summary
        self._emit(RunEvent(type=EVENT_SUBAGENT_STATUS_CHANGED, payload=payload))
        if self._snapshot_listener is not None:
            try:
                self._snapshot_listener(self._ordered_snapshots())
            except Exception:
                logger.exception("subagent snapshot listener failed")

    def _emit(self, event: RunEvent) -> None:
        if self._events is None:
            return
        try:
            self._events.emit(event)
        except Exception:  # a sink failure must not corrupt scheduling
            logger.exception("subagent event sink failed for %s", event.type)


class _DaemonWorkerPool:
    """Minimal fixed-size daemon worker pool.

    Replaces ``ThreadPoolExecutor`` (non-daemon workers; no daemon control on
    the Python versions this project runs on). The coordinator already enforces
    the slot cap and FIFO through its own ``_running``/``_pending`` state, so
    the pool only needs to run a task on a daemon thread and report the outcome
    through a standard ``concurrent.futures.Future`` (done callbacks run on the
    worker thread, matching ThreadPoolExecutor semantics).

    Daemon workers guarantee that a child stuck in a non-interruptible sync
    model request can never hang CLI exit; the coordinator discards late
    results anyway (design §Cancellation Limit).
    """

    def __init__(
        self, max_workers: int, thread_name_prefix: str = "subagent-worker"
    ) -> None:
        self._name_prefix = thread_name_prefix
        self._queue: Queue = Queue()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._shutdown = False
        for index in range(max_workers):
            thread = threading.Thread(
                target=self._worker,
                name=f"{self._name_prefix}-{index + 1}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                return
            future, fn, args, kwargs = item
            try:
                if future.set_running_or_notify_cancel():
                    try:
                        result = fn(*args, **kwargs)
                    except BaseException as exc:  # noqa: BLE001 - Future must capture Aborted
                        future.set_exception(exc)
                    else:
                        future.set_result(result)
            finally:
                self._queue.task_done()

    def submit(
        self, fn: Callable[..., object], *args: object, **kwargs: object
    ) -> Future:
        """Queue ``fn`` on one of the fixed daemon workers."""
        with self._lock:
            if self._shutdown:
                raise RuntimeError("worker pool is shut down")
            future: Future = Future()
            self._queue.put((future, fn, args, kwargs))
        return future

    def shutdown(self) -> None:
        """Stop accepting new tasks; running daemon workers finish on their own.

        The coordinator already performed a bounded join before calling this, so
        any worker still alive is stuck in a non-interruptible sync request; as
        a daemon it can never block interpreter exit.
        """
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            workers = len(self._threads)
        for _ in range(workers):
            self._queue.put(None)
