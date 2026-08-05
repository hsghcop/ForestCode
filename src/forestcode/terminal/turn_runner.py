"""TurnRunner: runs one backend turn on a worker thread, UI on the main thread.

Discipline (plan §7.2): the worker thread runs backend logic only; all UI —
``renderer``, ``rich.Live``, the prompt_toolkit confirm menu — stays on the main
thread. They communicate over a single thread-safe ``queue.Queue`` that carries
``event`` / ``confirm`` / ``done`` items in FIFO order (so started → confirm →
finished stay ordered).

Confirmation is a blocking thread-bridge: the worker calls the confirm proxy,
which puts a ``confirm`` request on the queue and blocks on an ``Event``; the
main-thread pump renders the menu and replies, unblocking the worker.

Ctrl+C is cooperative: the first one sets the ``AbortSignal`` (which closes
connections / kills child processes via ``on_abort``) and replies ABORTED to any
in-flight or queued confirm so the worker cannot deadlock; the second forces
``exit 130``.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from queue import Empty
from typing import TYPE_CHECKING, Any, Protocol

from forestcode.context import ContextFragment
from forestcode.core.abort import Aborted, AbortSignal
from forestcode.core.types import RunEvent
from forestcode.skills import SkillSnapshot
from forestcode.tools import ApprovalRequest

from .renderer import TerminalRenderer

if TYPE_CHECKING:
    from .bridge import InputOutcome, TurnExecution
    from .confirm import ConfirmationController

logger = logging.getLogger(__name__)

# Sentinel reply meaning "cancelled" — the proxy turns it into Aborted inside
# the worker rather than faking a user "No" (plan §7.2 评审3 / decision #36).
ABORTED = object()

# Short poll so Ctrl+C is responsive on Windows and the Live region can refresh
# (plan §7.2 评审3 / decision #34).
_POLL_SECONDS = 0.1
_JOIN_TIMEOUT_SECONDS = 5.0


class TurnBridge(Protocol):
    """The slice of BackendBridge that TurnRunner depends on.

    Structural: a test double only needs ``run_one_turn`` and
    ``render_turn_epilogue`` (plan §7.2 keeps the worker/UI boundary behind the
    bridge; TurnRunner never touches the rest).
    """

    def run_one_turn(
        self,
        user_task: str,
        *,
        sink: Callable[[RunEvent], None],
        confirm: Callable[[ApprovalRequest], bool],
        abort: AbortSignal | None = None,
        transient_fragments: tuple[ContextFragment, ...] = (),
        skills_snapshot: SkillSnapshot | None = None,
        launch_context: Any = None,
        confirm_proxy: ConfirmProxy | None = None,
    ) -> TurnExecution: ...

    def render_turn_epilogue(self, execution: TurnExecution) -> None: ...


@dataclass(slots=True)
class _ConfirmTicket:
    request: ApprovalRequest
    _event: threading.Event
    _result: Any = False
    task_id: str | None = None
    _reply_lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def answered(self) -> bool:
        """True once ``reply`` ran (main-thread pump skips answered tickets)."""
        return self._event.is_set()

    def reply(self, value: Any) -> None:
        with self._reply_lock:
            if self._event.is_set():
                return
            self._result = value
            self._event.set()

    def wait(self) -> Any:
        self._event.wait()
        return self._result


class ConfirmProxy:
    """Worker-side confirm callable that bridges to the main thread (plan §7.2).

    ``task_id`` optionally tags a ticket as belonging to one subagent child
    (design §Mutation Gate and Approval / R6). The tag lives on the ticket, not
    on ``ApprovalRequest`` — the approval contract stays policy-neutral.
    ``cancel_task(task_id)`` resolves only that child's current and queued
    tickets to ABORTED (the worker's ``wait`` then raises ``Aborted``), so
    cancelling one child can never drain another child's tickets. Ctrl+C still
    replies ABORTED to every ticket via the main-thread pump.
    """

    def __init__(self, q: queue.Queue) -> None:
        self._q = q
        self._tickets: dict[str, list[_ConfirmTicket]] = {}
        self._lock = threading.Lock()

    def __call__(
        self,
        request: ApprovalRequest,
        *,
        task_id: str | None = None,
        abort: AbortSignal | None = None,
    ) -> bool:
        ticket = _ConfirmTicket(
            request=request, _event=threading.Event(), task_id=task_id
        )
        if task_id is not None:
            with self._lock:
                self._tickets.setdefault(task_id, []).append(ticket)
            if abort is not None:
                abort.on_abort(lambda: self._cancel_ticket(task_id, ticket))
        self._q.put(("confirm", ticket))
        try:
            result = ticket.wait()
        finally:
            if task_id is not None:
                self._remove_ticket(task_id, ticket)
        if result is ABORTED:
            raise Aborted()
        return bool(result)

    def _cancel_ticket(self, task_id: str, ticket: _ConfirmTicket) -> None:
        with self._lock:
            tickets = self._tickets.get(task_id)
            if tickets is None or ticket not in tickets:
                return
        ticket.reply(ABORTED)

    def _remove_ticket(self, task_id: str, ticket: _ConfirmTicket) -> None:
        with self._lock:
            tickets = self._tickets.get(task_id)
            if tickets is None:
                return
            if ticket in tickets:
                tickets.remove(ticket)
            if not tickets:
                self._tickets.pop(task_id, None)

    def cancel_task(self, task_id: str) -> None:
        """Resolve one child's current and queued tickets to ABORTED (R6).

        Idempotent; a ticket already answered keeps its answer. Other children's
        tickets are untouched. Thread-safe: called from the coordinator when a
        child is cancelled.
        """
        with self._lock:
            tickets = self._tickets.pop(task_id, [])
        for ticket in tickets:
            ticket.reply(ABORTED)


class TurnRunner:
    """Runs ``bridge.run_one_turn`` on a worker thread and pumps its events."""

    def __init__(
        self,
        renderer: TerminalRenderer,
        confirm_controller: ConfirmationController,
        *,
        join_timeout: float = _JOIN_TIMEOUT_SECONDS,
    ) -> None:
        self._renderer = renderer
        self._confirm = confirm_controller
        self._join_timeout = join_timeout

    def run(
        self,
        bridge: TurnBridge,
        task: str,
        *,
        transient_fragments: tuple[ContextFragment, ...] = (),
        skills_snapshot: SkillSnapshot | None = None,
        launch_context: Any = None,
    ) -> InputOutcome:
        q: queue.Queue = queue.Queue()
        abort = AbortSignal()
        proxy = ConfirmProxy(q)
        box: dict[str, Any] = {}

        def worker() -> None:
            try:
                box["result"] = bridge.run_one_turn(
                    task,
                    sink=lambda event: q.put(("event", event)),
                    confirm=proxy,
                    abort=abort,
                    transient_fragments=transient_fragments,
                    skills_snapshot=skills_snapshot,
                    launch_context=launch_context,
                    confirm_proxy=proxy,
                )
            except Aborted:
                box["aborted"] = True
            except BaseException as exc:  # noqa: BLE001 - surfaced to main thread
                box["error"] = exc
            finally:
                q.put(("done", None))

        thread = threading.Thread(target=worker, name="forestcode-turn", daemon=True)
        self._renderer.begin_turn()
        thread.start()
        interrupts = 0
        pending: _ConfirmTicket | None = None
        try:
            while True:
                try:
                    kind, payload = q.get(timeout=_POLL_SECONDS)
                except Empty:
                    continue
                if kind == "event":
                    self._render_event(payload)
                elif kind == "confirm":
                    if payload.answered:
                        # Already resolved by cancel_task (targeted child
                        # cancel) or a drain — do not show a stale dialog.
                        pending = None
                        continue
                    pending = payload
                    try:
                        decision = self._confirm.confirm(payload.request)
                    except KeyboardInterrupt:
                        # In-flight confirm hit by Ctrl+C: reply ABORTED so the
                        # worker unblocks, then run the cancel path (plan §7.2).
                        interrupts += 1
                        abort.set()
                        payload.reply(ABORTED)
                        pending = None
                        if interrupts >= 2:
                            raise SystemExit(130) from None
                        continue
                    payload.reply(decision)
                    pending = None
                elif kind == "done":
                    break
        except KeyboardInterrupt:
            interrupts += 1
            if interrupts >= 2:
                raise SystemExit(130) from None
            abort.set()
            if pending is not None:
                pending.reply(ABORTED)
                pending = None
            self._drain_pending_confirms(q)
            # Keep pumping until the worker observes the abort and finishes.
            self._drain_until_done(q, abort)
        finally:
            self._renderer.end_turn()

        thread.join(timeout=self._join_timeout)
        return self._finish(bridge, box)

    # -- helpers -----------------------------------------------------------
    def _render_event(self, event: Any) -> None:
        try:
            self._renderer.on_event(event)
        except Exception:
            logger.exception("event render error")

    def _drain_pending_confirms(self, q: queue.Queue) -> None:
        """Reply ABORTED to any confirms already queued (plan §7.2)."""
        drained: list[tuple[str, Any]] = []
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "confirm":
                    payload.reply(ABORTED)
                else:
                    drained.append((kind, payload))
        except Empty:
            pass
        for item in drained:
            q.put(item)

    def _drain_until_done(self, q: queue.Queue, abort: AbortSignal) -> None:
        interrupts = 0
        while True:
            try:
                kind, payload = q.get(timeout=_POLL_SECONDS)
            except Empty:
                continue
            except KeyboardInterrupt:
                interrupts += 1
                if interrupts >= 1:
                    raise SystemExit(130) from None
                continue
            if kind == "confirm":
                payload.reply(ABORTED)
            elif kind == "done":
                return
            # events during teardown are dropped (the turn is cancelled)

    def _finish(self, bridge: TurnBridge, box: dict[str, Any]) -> InputOutcome:
        from .bridge import InputOutcome

        if box.get("aborted"):
            self._renderer.render_warning("⊘ 已中断 (turn cancelled)")
            return InputOutcome(action="continue")
        error = box.get("error")
        if error is not None:
            return InputOutcome(
                action="error", exit_code=1, error=f"Agent error: {error}"
            )
        execution: TurnExecution | None = box.get("result")
        if execution is None:
            return InputOutcome(
                action="error", exit_code=1, error="Agent error: no turn result"
            )
        bridge.render_turn_epilogue(execution)
        return execution.outcome
