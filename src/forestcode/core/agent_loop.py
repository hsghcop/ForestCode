"""Core orchestration loop for one user task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

from .abort import Aborted, AbortSignal
from .context_builder import ContextBuilder
from .events import EventSink
from .run_state import RunState
from .stop_policy import StopPolicy
from .tool_executor import ToolExecutor
from .turn_processor import TurnProcessor
from .types import ModelClient, RunEvent

if TYPE_CHECKING:
    # SubagentCancelReason lives in the subagents package; the import is
    # type-checking only so the core runtime never hard-depends on subagents
    # (annotation strings via ``from __future__ import annotations``).
    from forestcode.subagents.types import SubagentCancelReason

logger = logging.getLogger(__name__)


class SubagentCoordinatorProtocol(Protocol):
    """The narrow slice of SubagentCoordinator the AgentLoop consumes.

    design §Architectural Boundary: the loop only queries whether children are
    still active and asks for cleanup on termination — it never manages threads,
    the FIFO queue or the terminal. ``cleanup`` is idempotent (design §Scheduling
    and Waiting, step 5).
    """

    def has_active_children(self) -> bool: ...

    def cleanup(self, reason: SubagentCancelReason) -> None: ...


class RunRecorder(Protocol):
    def record_run(self, state: RunState) -> None:
        """Record a completed or failed run."""


class CompactionController(Protocol):
    def maybe_major_compact(self, model_input, state: RunState) -> bool:
        """Compact session history before a model request when needed."""
        ...

    def maybe_normal_compact(self, state: RunState) -> bool:
        """Compact session history after a run has been recorded."""
        ...


class AgentLoop:
    def __init__(
        self,
        model: ModelClient,
        context_builder: ContextBuilder,
        turn_processor: TurnProcessor,
        tool_executor: ToolExecutor,
        events: EventSink,
        stop_policy: StopPolicy,
        memory_manager: RunRecorder | None = None,
        compaction_controller: CompactionController | None = None,
        abort: AbortSignal | None = None,
        subagents: SubagentCoordinatorProtocol | None = None,
    ) -> None:
        self.model = model
        self.context_builder = context_builder
        self.turn_processor = turn_processor
        self.tool_executor = tool_executor
        self.events = events
        self.stop_policy = stop_policy
        self.memory_manager = memory_manager
        self.compaction_controller = compaction_controller
        # Per-turn cancellation token (plan §11-B3). None => no cancellation,
        # behavior identical to before.
        self._abort = abort
        # Per-run subagent coordinator (design §Architectural Boundary). None on
        # the single-agent path => no coordinator interaction, no overhead.
        self._subagents = subagents

    def _check_abort(self) -> None:
        if self._abort is not None:
            self._abort.raise_if_aborted()

    def _aborted(self) -> bool:
        return self._abort is not None and self._abort.is_set()

    def run(self, user_task: str) -> RunState:
        state = RunState.start(user_task)
        self.events.emit(RunEvent("run_started", {"user_task": user_task}))
        # Subagent cleanup reason per design §State machine: normal finish,
        # parent aborted (Ctrl+C / model-side Aborted) or any other failure.
        # The finally guarantees cleanup on every exit path — including the
        # all-children-terminal case, which must still stop the coordinator's
        # watchdog and worker pool (design §Cancellation Limit / §Scheduling).
        cleanup_reason: SubagentCancelReason = "parent_finished"
        try:
            result = self._run_loop(state)
            if result.final_text is None or result.error is not None:
                cleanup_reason = "parent_failed"
            return result
        except Aborted:
            # Cooperative cancellation (plan §7): mark aborted and surface a
            # run_cancelled event, then re-raise so the worker (TurnRunner)
            # unwinds. record_run is intentionally NOT called — the turn did
            # not complete (plan §7.4).
            state.fail("aborted")
            self.events.emit(RunEvent("run_cancelled", {"turns": state.turns}))
            cleanup_reason = "parent_aborted"
            raise
        except BaseException:
            # Any other failure (model adapter error, tool crash, ...): children
            # must be logically cancelled with parent_failed; the exception is
            # re-raised untouched.
            cleanup_reason = "parent_failed"
            raise
        finally:
            self._cleanup_subagents(cleanup_reason)

    def _cleanup_subagents(self, reason: SubagentCancelReason) -> None:
        """Cancel leftover children on this run's termination path (idempotent).

        ``cleanup`` is a required coordination side effect but must never hide
        the already-determined run outcome; a coordinator failure is logged and
        isolated (spec §可选副作用).
        """
        if self._subagents is None:
            return
        try:
            self._subagents.cleanup(reason)
        except Exception:
            logger.exception("subagent cleanup failed (reason=%s)", reason)

    def _run_loop(self, state: RunState) -> RunState:
        while not self.stop_policy.should_stop(state):
            self._check_abort()  # checkpoint ①: loop top
            model_input = self.context_builder.build(state)
            if (
                self.compaction_controller is not None and not self._aborted()
            ):  # ②: before major compact
                try:
                    compacted = self.compaction_controller.maybe_major_compact(
                        model_input, state
                    )
                except Exception as exc:  # noqa: BLE001 - compaction must not hide run outcome
                    compacted = False
                    self.events.emit(
                        RunEvent(
                            "session_compaction_failed",
                            {"kind": "major", "error": str(exc)},
                        )
                    )
                if compacted:
                    self.events.emit(
                        RunEvent("session_compaction_finished", {"kind": "major"})
                    )
                    model_input = self.context_builder.build(state)

            self._check_abort()  # checkpoint ③: before model request
            self.events.emit(
                RunEvent("model_request_started", {"turn": state.turns + 1})
            )
            model_output = self.model.complete(model_input, abort=self._abort)
            self.events.emit(
                RunEvent("model_response_received", {"turn": state.turns + 1})
            )
            self._check_abort()  # checkpoint ④: after model response
            for artifact in model_output.assistant_turn.reasoning_artifacts:
                if artifact.visible and artifact.display_text:
                    self.events.emit(
                        RunEvent(
                            "assistant_reasoning_received",
                            {
                                "provider": artifact.provider,
                                "kind": artifact.kind,
                                "text": artifact.display_text,
                            },
                        )
                    )

            state.turns += 1
            turn_result = self.turn_processor.process(model_output, state)
            for event in turn_result.events:
                self.events.emit(event)

            if turn_result.final_text is not None:
                state.add_assistant_turn(model_output.assistant_turn)
                state.finish(turn_result.final_text)
                self.events.emit(
                    RunEvent(
                        "assistant_text_received", {"text": turn_result.final_text}
                    )
                )
                self.events.emit(RunEvent("run_finished", {"turns": state.turns}))
                if self._record_run(state):
                    self._normal_compact(state)
                return state

            if not turn_result.tool_calls:
                state.fail("model returned neither final_text nor tool_calls")
                self.events.emit(RunEvent("run_failed", {"error": state.error}))
                if self._record_run(state):
                    self._normal_compact(state)
                return state

            state.add_assistant_turn(model_output.assistant_turn)
            for call in turn_result.tool_calls:
                self._check_abort()  # checkpoint ⑤: before each tool
                self.events.emit(
                    RunEvent(
                        "tool_call_started",
                        {
                            "tool_call_id": call.id,
                            "tool_name": call.name,
                            # B1: raw arguments; the frontend extracts the key
                            # display arg per tool (plan §4.1). Backend stays
                            # presentation-agnostic.
                            "arguments": call.arguments,
                        },
                    )
                )

                result = self.tool_executor.execute(call, state)
                state.add_tool_result(result)
                self.events.emit(
                    RunEvent(
                        "tool_call_finished",
                        {
                            "tool_call_id": call.id,
                            "tool_name": call.name,
                            "ok": result.ok,
                            "summary": result.summary,
                            # B2: structured metrics passthrough. The frontend
                            # formats data (lines/diff/exit_code, command
                            # stdout/stderr) per plan §4.2/§5. No generic
                            # `content` field — that would let the renderer dump
                            # arbitrary tool output and blur display ownership.
                            "data": result.data,
                        },
                    )
                )

        state.fail("max turns reached")
        self.events.emit(RunEvent("run_failed", {"error": state.error}))
        if self._record_run(state):
            self._normal_compact(state)
        return state

    def _record_run(self, state: RunState) -> bool:
        if self.memory_manager is None:
            return False
        try:
            self.memory_manager.record_run(state)
        except Exception as exc:  # noqa: BLE001 - memory recording must not hide run outcome
            self.events.emit(RunEvent("memory_record_failed", {"error": str(exc)}))
            return False
        return True

    def _normal_compact(self, state: RunState) -> None:
        # Compaction is an optional optimization; skip it once cancelled (§7).
        if self.compaction_controller is None or self._aborted():
            return
        try:
            compacted = self.compaction_controller.maybe_normal_compact(state)
        except Exception as exc:  # noqa: BLE001 - compaction must not hide run outcome
            self.events.emit(
                RunEvent(
                    "session_compaction_failed", {"kind": "normal", "error": str(exc)}
                )
            )
            return
        if compacted:
            self.events.emit(
                RunEvent("session_compaction_finished", {"kind": "normal"})
            )
