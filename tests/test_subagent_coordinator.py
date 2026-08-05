"""Coordinator, FIFO scheduling, cancellation, timeouts and persistence helpers.

design §Scheduling and Waiting, §Cancellation Limit, §State Machine, §Persistence.
Offline only: no network, no model calls. All concurrency assertions are driven
by Events/Barriers and bounded waits — no ``time.sleep``-based races. A tiny
amount of real time is inherent to the watchdog deadline mechanism (it is a wall
clock), so timeout tests use generous bounded waits and Events to observe it.
"""

from __future__ import annotations

import itertools
import threading
import time
import unittest
from collections.abc import Callable
from pathlib import Path

from forestcode.core.abort import Aborted, AbortSignal
from forestcode.core.events import InMemoryEventSink
from forestcode.subagents import (
    EVENT_SUBAGENT_STATUS_CHANGED,
    SubagentCoordinator,
    UnknownTaskError,
    WaitOutcome,
    child_transcript_dir,
)
from forestcode.subagents.coordinator import CoordinatorClosedError
from forestcode.subagents.types import (
    SubagentRequest,
    SubagentResult,
)


def _seq_ids() -> Callable[[], str]:
    counter = itertools.count(1)
    return lambda: f"sub-{next(counter):04d}"


def _request(name: str = "helper") -> SubagentRequest:
    return SubagentRequest(
        task_id="unused",
        agent_name=name,
        description=f"desc {name}",
        prompt=f"prompt {name}",
    )


class _TaskGate:
    """Per-task control point: the test decides when a worker starts/exits."""

    __slots__ = ("abort_seen", "release", "started")

    def __init__(self) -> None:
        self.started = threading.Event()
        self.abort_seen = threading.Event()
        self.release = threading.Event()


class _BlockingRunner:
    """Blocks each child until its gate is released; records abort observation.

    ``ignore_abort=True`` simulates a runner stuck in a non-interruptible sync
    model request: it never registers an abort callback, so it keeps blocking
    until released even after cancellation (the coordinator must keep the slot
    and discard its late result).
    """

    def __init__(self, *, ignore_abort: bool = False) -> None:
        self.ignore_abort = ignore_abort
        self._gates: dict[str, _TaskGate] = {}
        self._lock = threading.Lock()
        self._started_order: list[str] = []
        self.completed: dict[str, SubagentResult] = {}
        self.active = 0
        self.max_active = 0

    def gate(self, task_id: str) -> _TaskGate:
        with self._lock:
            return self._gates.setdefault(task_id, _TaskGate())

    def started_order(self) -> list[str]:
        with self._lock:
            return list(self._started_order)

    def run(self, request: SubagentRequest, *, abort: AbortSignal) -> SubagentResult:
        gate = self.gate(request.task_id)
        if not self.ignore_abort:
            abort.on_abort(gate.abort_seen.set)
        with self._lock:
            self._started_order.append(request.task_id)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        gate.started.set()
        gate.release.wait()  # the test controls when this worker actually exits
        result = SubagentResult(
            task_id=request.task_id,
            agent_name=request.agent_name,
            final_text=f"done:{request.task_id}",
            turn_count=1,
            tool_count=2,
        )
        with self._lock:
            self.completed[request.task_id] = result
            self.active -= 1
        return result


class _FailRunner:
    """Raises immediately: simulates ``failed(reason=child_error)``."""

    def __init__(self, message: str = "child exploded") -> None:
        self.message = message
        self.started = threading.Event()

    def run(self, request: SubagentRequest, *, abort: AbortSignal) -> SubagentResult:
        self.started.set()
        raise RuntimeError(self.message)


class _BoomEventSink:
    """Event sink that fails on every emit; scheduling must survive it."""

    def emit(self, event) -> None:  # type: ignore[no-untyped-def]
        raise RuntimeError("sink exploded")


class SubagentCoordinatorTest(unittest.TestCase):
    def make_coordinator(
        self,
        runner,
        *,
        max_workers: int = 6,
        events=None,
        **kwargs,
    ) -> SubagentCoordinator:
        kwargs.setdefault("cleanup_join_seconds", 0.2)
        coordinator = SubagentCoordinator(
            runner,
            max_workers=max_workers,
            events=events,
            id_factory=_seq_ids(),
            default_wait_timeout=2.0,
            **kwargs,
        )
        self.addCleanup(coordinator.cleanup)
        return coordinator

    def wait_terminal(
        self, coordinator, task_id: str, timeout: float = 5.0
    ) -> WaitOutcome:
        """Wait until one task is terminal; fails the test on timeout."""
        outcome = coordinator.wait([task_id], timeout=timeout)
        self.assertFalse(outcome.timed_out, f"task {task_id} did not finish in time")
        return outcome

    # -- concurrency cap and FIFO ------------------------------------------
    def test_concurrency_cap_and_fifo_refill(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=6)
        for index in range(8):
            coordinator.delegate(_request(f"agent-{index}"))
        # Exactly 6 running, 2 queued, queue positions 1..8 stable.
        snapshots = {s.task_id: s for s in coordinator.list()}
        running = [s for s in snapshots.values() if s.status == "running"]
        queued = [s for s in snapshots.values() if s.status == "queued"]
        self.assertEqual(len(running), 6)
        self.assertEqual(len(queued), 2)
        self.assertEqual(
            [s.queue_position for s in snapshots.values()], list(range(1, 9))
        )
        # Let the first two started workers exit: the two queued tasks (7, 8)
        # fill the freed slots in FIFO order.
        for tid in ("sub-0001", "sub-0002"):
            self.assertTrue(runner.gate(tid).started.wait(5))
            runner.gate(tid).release.set()
        self.assertTrue(runner.gate("sub-0007").started.wait(5))
        self.assertTrue(runner.gate("sub-0008").started.wait(5))
        started = runner.started_order()
        self.assertEqual(started[:6], [f"sub-{i:04d}" for i in range(1, 7)])
        self.assertEqual(set(started[6:]), {"sub-0007", "sub-0008"})
        # Physical cap still holds: 6 running/queued mix, never more workers.
        self.assertLessEqual(
            len([s for s in coordinator.list() if s.status == "running"]), 6
        )
        self.assertLessEqual(runner.max_active, 6)

    def test_fifo_delivery_order(self):
        """All tasks eventually deliver; FIFO governs start/refill, not the
        completion order of concurrently running workers (covered by
        test_concurrency_cap_and_fifo_refill), so assert the delivered set.
        """
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=6)
        for index in range(8):
            coordinator.delegate(_request(f"agent-{index}"))
        for tid in [f"sub-{i:04d}" for i in range(1, 9)]:
            runner.gate(tid).release.set()
        collected: dict[str, SubagentResult] = {}
        deadline = time.monotonic() + 5
        while len(collected) < 8 and time.monotonic() < deadline:
            outcome = coordinator.wait(None, timeout=0.5)
            collected.update(outcome.results)
        self.assertEqual(sorted(collected), [f"sub-{i:04d}" for i in range(1, 9)])

    def test_wait_is_woken_by_parent_abort(self):
        runner = _BlockingRunner(ignore_abort=True)
        coordinator = self.make_coordinator(runner, max_workers=1)
        delegated = coordinator.delegate(_request())
        self.assertTrue(runner.gate(delegated.task_id).started.wait(5))
        abort = AbortSignal()
        outcome: list[str] = []

        def wait() -> None:
            try:
                coordinator.wait([delegated.task_id], timeout=60, abort=abort)
            except Aborted:
                outcome.append("aborted")

        thread = threading.Thread(target=wait)
        thread.start()
        abort.set()
        thread.join(5)
        self.assertEqual(outcome, ["aborted"])
        runner.gate(delegated.task_id).release.set()

    # -- cancel -------------------------------------------------------------
    def test_queued_cancel_is_immediate_and_keeps_positions(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=6)
        for index in range(8):
            coordinator.delegate(_request(f"agent-{index}"))
        snapshot = coordinator.cancel("sub-0007")
        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.status, "cancelled")
        self.assertEqual(snapshot.cancel_reason, "requested")
        # Task 8 is still queued at its original position; slot usage unchanged.
        snapshots = {s.task_id: s for s in coordinator.list()}
        self.assertEqual(snapshots["sub-0008"].status, "queued")
        self.assertEqual(snapshots["sub-0008"].queue_position, 8)
        self.assertEqual(
            len([s for s in snapshots.values() if s.status == "running"]), 6
        )
        # A cancelled task is terminal: wait returns its snapshot without results.
        outcome = coordinator.wait(["sub-0007"], timeout=0)
        self.assertFalse(outcome.timed_out)
        self.assertEqual(outcome.snapshots[0].status, "cancelled")
        self.assertEqual(outcome.results, {})
        # Unknown ids return None (the tool layer converts to a tool failure).
        self.assertIsNone(coordinator.cancel("sub-9999"))

    def test_running_cancel_keeps_slot_until_worker_exits(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"))
        gate = runner.gate("sub-0001")
        self.assertTrue(gate.started.wait(5))
        snapshot = coordinator.cancel("sub-0001")
        assert snapshot is not None
        self.assertEqual(snapshot.status, "cancelling")
        self.assertEqual(snapshot.cancel_reason, "requested")
        # The worker observed the abort but has not exited yet: the slot must
        # stay occupied, so a new task can only queue.
        self.assertTrue(gate.abort_seen.wait(5))
        coordinator.delegate(_request("b"))
        self.assertEqual(coordinator.list()[1].status, "queued")
        # Let the worker exit: only then does it free the slot and refill FIFO.
        gate.release.set()
        self.assertTrue(runner.gate("sub-0002").started.wait(5))
        outcome = self.wait_terminal(coordinator, "sub-0001")
        self.assertEqual(outcome.snapshots[0].status, "cancelled")
        self.assertEqual(outcome.snapshots[0].cancel_reason, "requested")
        self.assertEqual(outcome.results, {})  # cancelled tasks never deliver

    def test_non_cooperative_runner_result_is_discarded(self):
        runner = _BlockingRunner(ignore_abort=True)
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"))
        gate = runner.gate("sub-0001")
        self.assertTrue(gate.started.wait(5))
        snapshot = coordinator.cancel("sub-0001")
        assert snapshot is not None
        self.assertEqual(snapshot.status, "cancelling")
        # The runner never registered an abort callback (sync-model analogue).
        self.assertFalse(gate.abort_seen.wait(0.1))
        # Worker finally exits; its returned result must be discarded.
        gate.release.set()
        outcome = self.wait_terminal(coordinator, "sub-0001")
        self.assertEqual(outcome.snapshots[0].status, "cancelled")
        self.assertEqual(outcome.results, {})
        self.assertIn("sub-0001", runner.completed)  # the result existed…

    # -- timeout ------------------------------------------------------------
    def test_timeout_moves_to_cancelling_then_cancelled(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"), timeout_seconds=0.1)
        gate = runner.gate("sub-0001")
        self.assertTrue(gate.started.wait(5))
        # The watchdog fires at the deadline: cancelling(timeout), then abort.
        self.assertTrue(gate.abort_seen.wait(5))
        snapshot = coordinator.list()[0]
        self.assertEqual(snapshot.status, "cancelling")
        self.assertEqual(snapshot.cancel_reason, "timeout")
        # cancelling is non-terminal and still occupies the slot.
        coordinator.delegate(_request("b"))
        self.assertEqual(coordinator.list()[1].status, "queued")
        # Only the worker's actual exit finalizes cancelled(timeout) + refill.
        gate.release.set()
        self.assertTrue(runner.gate("sub-0002").started.wait(5))
        outcome = self.wait_terminal(coordinator, "sub-0001")
        self.assertEqual(outcome.snapshots[0].status, "cancelled")
        self.assertEqual(outcome.snapshots[0].cancel_reason, "timeout")
        self.assertEqual(outcome.results, {})

    def test_queued_time_does_not_count_toward_deadline(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"), timeout_seconds=0.1)
        coordinator.delegate(_request("b"), timeout_seconds=0.1)
        gate_a = runner.gate("sub-0001")
        self.assertTrue(gate_a.started.wait(5))
        # A times out while B has been queued for longer than B's own deadline.
        self.assertTrue(gate_a.abort_seen.wait(5))
        self.assertEqual(coordinator.list()[1].status, "queued")  # B unaffected
        gate_a.release.set()
        self.assertTrue(runner.gate("sub-0002").started.wait(5))
        self.assertEqual(coordinator.list()[1].status, "running")

    # -- wait contract ------------------------------------------------------
    def test_wait_timeout_zero_is_instant_snapshot_and_never_cancels(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"))
        self.assertTrue(runner.gate("sub-0001").started.wait(5))
        outcome = coordinator.wait(["sub-0001"], timeout=0)
        self.assertTrue(outcome.timed_out)
        self.assertEqual(outcome.snapshots[0].status, "running")
        self.assertEqual(outcome.results, {})
        # The task keeps running: the wait did not cancel it.
        self.assertEqual(coordinator.list()[0].status, "running")

    def test_wait_delivers_once_and_repeat_wait_omits_body(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"))
        gate = runner.gate("sub-0001")
        self.assertTrue(gate.started.wait(5))
        gate.release.set()
        outcome = self.wait_terminal(coordinator, "sub-0001")
        result = outcome.results.get("sub-0001")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.final_text, "done:sub-0001")
        self.assertEqual(result.turn_count, 1)
        self.assertEqual(result.tool_count, 2)
        self.assertTrue(result.delivered)
        self.assertTrue(outcome.snapshots[0].delivered)
        # A repeated wait on the delivered task returns the snapshot with
        # delivered=True and no body (result_omitted semantics).
        repeat = coordinator.wait(["sub-0001"], timeout=0)
        self.assertFalse(repeat.timed_out)
        self.assertEqual(repeat.results, {})
        self.assertTrue(repeat.snapshots[0].delivered)

    def test_wait_none_returns_on_first_terminal_only(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=2)
        coordinator.delegate(_request("a"))
        coordinator.delegate(_request("b"))
        self.assertTrue(runner.gate("sub-0002").started.wait(5))
        runner.gate("sub-0002").release.set()
        outcome = coordinator.wait(None, timeout=5)
        self.assertFalse(outcome.timed_out)
        self.assertEqual(list(outcome.results), ["sub-0002"])  # only B delivered
        self.assertNotIn("sub-0001", outcome.results)
        # The undelivered task is still running; a snapshot wait times out.
        instant = coordinator.wait(None, timeout=0)
        self.assertTrue(instant.timed_out)
        runner.gate("sub-0001").release.set()
        outcome = self.wait_terminal(coordinator, "sub-0001")
        self.assertEqual(list(outcome.results), ["sub-0001"])

    def test_wait_rejects_empty_list_and_unknown_ids(self):
        coordinator = self.make_coordinator(_FailRunner())
        with self.assertRaises(ValueError):
            coordinator.wait([], timeout=0)
        with self.assertRaises(UnknownTaskError):
            coordinator.wait(["sub-9999"], timeout=0)

    def test_failed_runner_produces_failed_result(self):
        runner = _FailRunner("kaboom")
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"))
        self.assertTrue(runner.started.wait(5))
        outcome = self.wait_terminal(coordinator, "sub-0001")
        snapshot = outcome.snapshots[0]
        self.assertEqual(snapshot.status, "failed")
        self.assertEqual(snapshot.cancel_reason, "child_error")
        self.assertIn("kaboom", snapshot.error or "")
        result = outcome.results["sub-0001"]
        self.assertTrue(result.delivered)

    def test_wait_timeout_bounds(self):
        coordinator = self.make_coordinator(_FailRunner())
        with self.assertRaises(ValueError):
            coordinator.wait(None, timeout=-1)
        with self.assertRaises(ValueError):
            coordinator.wait(None, timeout=61)

    # -- empty run ----------------------------------------------------------
    def test_empty_run_list_and_wait(self):
        coordinator = self.make_coordinator(_FailRunner())
        self.assertEqual(coordinator.list(), ())
        outcome = coordinator.wait(None, timeout=0)
        self.assertTrue(outcome.timed_out)
        self.assertEqual(outcome.snapshots, ())
        self.assertEqual(outcome.results, {})

    # -- cleanup ------------------------------------------------------------
    def test_cleanup_cancels_queued_and_running(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"))
        coordinator.delegate(_request("b"))
        self.assertTrue(runner.gate("sub-0001").started.wait(5))
        final = coordinator.cleanup("parent_finished")
        by_id = {s.task_id: s for s in final}
        self.assertEqual(by_id["sub-0001"].status, "cancelling")
        self.assertEqual(by_id["sub-0001"].cancel_reason, "parent_finished")
        self.assertEqual(by_id["sub-0002"].status, "cancelled")
        self.assertEqual(by_id["sub-0002"].cancel_reason, "parent_finished")
        with self.assertRaises(CoordinatorClosedError):
            coordinator.delegate(_request("c"))
        runner.gate("sub-0001").release.set()
        outcome = self.wait_terminal(coordinator, "sub-0001")
        self.assertEqual(outcome.snapshots[0].status, "cancelled")
        self.assertEqual(outcome.snapshots[0].cancel_reason, "parent_finished")
        # Idempotent: a second cleanup returns the same terminal snapshots.
        again = coordinator.cleanup("parent_finished")
        self.assertEqual({s.task_id for s in again}, {"sub-0001", "sub-0002"})

    def test_cleanup_is_bounded_with_stuck_worker(self):
        runner = _BlockingRunner(ignore_abort=True)
        coordinator = self.make_coordinator(
            runner, max_workers=1, cleanup_join_seconds=0.1
        )
        coordinator.delegate(_request("a"))
        self.assertTrue(runner.gate("sub-0001").started.wait(5))
        started = time.monotonic()
        final = coordinator.cleanup("parent_aborted")
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 1.0, "cleanup must not join a stuck worker forever")
        self.assertEqual(final[0].status, "cancelling")  # worker never exited
        self.assertEqual(final[0].cancel_reason, "parent_aborted")
        runner.gate("sub-0001").release.set()
        outcome = self.wait_terminal(coordinator, "sub-0001")
        self.assertEqual(outcome.snapshots[0].status, "cancelled")
        self.assertEqual(outcome.snapshots[0].cancel_reason, "parent_aborted")

    def test_cleanup_empty_run_is_idempotent(self):
        coordinator = self.make_coordinator(_FailRunner())
        self.assertEqual(coordinator.cleanup("parent_finished"), ())
        self.assertEqual(coordinator.cleanup("parent_finished"), ())
        with self.assertRaises(CoordinatorClosedError):
            coordinator.delegate(_request("a"))

    # -- state machine guard ------------------------------------------------
    def test_illegal_transition_is_rejected_without_state_change(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1)
        coordinator.delegate(_request("a"))
        coordinator.delegate(_request("b"))  # stays queued behind a
        with coordinator._lock:
            with self.assertRaises(ValueError):
                coordinator._transition("sub-0002", "completed")
            self.assertEqual(coordinator._tasks["sub-0002"].status, "queued")
            coordinator.cancel("sub-0001")  # running -> cancelling
            with self.assertRaises(ValueError):
                coordinator._transition("sub-0001", "completed")
            self.assertEqual(coordinator._tasks["sub-0001"].status, "cancelling")

    # -- events -------------------------------------------------------------
    def test_status_events_per_task_order(self):
        sink = InMemoryEventSink()
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1, events=sink)
        coordinator.delegate(_request("a"))
        coordinator.delegate(_request("b"))
        self.assertTrue(runner.gate("sub-0001").started.wait(5))
        runner.gate("sub-0001").release.set()
        outcome = self.wait_terminal(coordinator, "sub-0001")
        self.assertFalse(outcome.timed_out)
        runner.gate("sub-0002").release.set()
        self.wait_terminal(coordinator, "sub-0002")
        events = [e for e in sink.events if e.type == EVENT_SUBAGENT_STATUS_CHANGED]
        by_task: dict[str, list[str]] = {}
        for event in events:
            task_id = event.payload["task_id"]
            by_task.setdefault(task_id, []).append(event.payload["status"])
        self.assertEqual(by_task["sub-0001"], ["queued", "running", "completed"])
        self.assertEqual(by_task["sub-0002"], ["queued", "running", "completed"])
        self.assertIn("agent_name", events[0].payload)

    def test_cancel_and_timeout_events_carry_reason(self):
        sink = InMemoryEventSink()
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1, events=sink)
        coordinator.delegate(_request("a"))
        self.assertTrue(runner.gate("sub-0001").started.wait(5))
        coordinator.cancel("sub-0001")
        runner.gate("sub-0001").release.set()
        self.wait_terminal(coordinator, "sub-0001")
        events = [e for e in sink.events if e.type == EVENT_SUBAGENT_STATUS_CHANGED]
        cancelling = next(e for e in events if e.payload["status"] == "cancelling")
        cancelled = next(e for e in events if e.payload["status"] == "cancelled")
        self.assertEqual(cancelling.payload["reason"], "requested")
        self.assertEqual(cancelled.payload["reason"], "requested")

    def test_event_sink_failure_does_not_corrupt_scheduling(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(
            runner, max_workers=1, events=_BoomEventSink()
        )
        coordinator.delegate(_request("a"))
        self.assertTrue(runner.gate("sub-0001").started.wait(5))
        runner.gate("sub-0001").release.set()
        outcome = self.wait_terminal(coordinator, "sub-0001")
        self.assertEqual(outcome.snapshots[0].status, "completed")
        self.assertIn("sub-0001", outcome.results)

    # -- constructor and id factory ----------------------------------------
    def test_id_factory_validation(self):
        coordinator = SubagentCoordinator(
            _FailRunner(),
            max_workers=1,
            id_factory=lambda: "BAD ID!",
            default_wait_timeout=2.0,
        )
        with self.assertRaises(ValueError):
            coordinator.delegate(_request("a"))

    def test_duplicate_id_factory_output_is_rejected(self):
        coordinator = SubagentCoordinator(
            _FailRunner(),
            max_workers=1,
            id_factory=lambda: "sub-dup",
            default_wait_timeout=2.0,
        )
        self.addCleanup(coordinator.cleanup)
        coordinator.delegate(_request("a"))
        with self.assertRaises(ValueError):
            coordinator.delegate(_request("b"))

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            SubagentCoordinator(_FailRunner(), max_workers=0)
        with self.assertRaises(ValueError):
            SubagentCoordinator(_FailRunner(), default_wait_timeout=999)
        with self.assertRaises(ValueError):
            SubagentCoordinator(_FailRunner(), default_task_timeout_seconds=0)

    def test_delegate_timeout_validation(self):
        coordinator = self.make_coordinator(_FailRunner())
        with self.assertRaises(ValueError):
            coordinator.delegate(_request("a"), timeout_seconds=-1)

    def test_delegate_replaces_request_task_id(self):
        runner = _BlockingRunner()
        coordinator = self.make_coordinator(runner, max_workers=1)
        snapshot = coordinator.delegate(_request("a"))
        self.assertEqual(snapshot.task_id, "sub-0001")  # coordinator-owned id
        self.assertEqual(snapshot.agent_name, "a")  # agent name preserved
        self.assertEqual(snapshot.status, "queued")
        runner.gate("sub-0001").release.set()
        self.wait_terminal(coordinator, "sub-0001")


class SubagentPersistenceTest(unittest.TestCase):
    def test_child_transcript_dir_structure(self):
        root = Path("/tmp/workspace")
        self.assertEqual(
            child_transcript_dir(root, "default"),
            root / ".forestcode" / "subagents" / "default" / "sessions",
        )

    def test_child_transcript_dir_rejects_unsafe_session_ids(self):
        for bad in (
            "",
            ".",
            "..",
            "a/b",
            "a\\b",
            "a:b",
            "a*b",
            "a<b",
            "a>b",
            "a|b",
            "a?b",
        ):
            with self.subTest(session_id=bad), self.assertRaises(ValueError):
                child_transcript_dir(Path("/tmp"), bad)

if __name__ == "__main__":
    unittest.main()
