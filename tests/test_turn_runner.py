from __future__ import annotations

import io
import threading
import unittest
from collections.abc import Callable
from typing import Any

from forestcode.core.abort import Aborted, AbortSignal
from forestcode.core.types import RunEvent
from forestcode.terminal.bridge import InputOutcome, TurnExecution
from forestcode.terminal.confirm import ConfirmationController
from forestcode.terminal.input import StdinInputController
from forestcode.terminal.renderer import PlainRenderer
from forestcode.terminal.turn_runner import ABORTED, ConfirmProxy, TurnRunner
from forestcode.tools import ApprovalRequest


class _StubBridge:
    """Minimal bridge stand-in driving run_one_turn behavior for tests."""

    def __init__(self, behavior):
        self._behavior = behavior
        self.epilogues = []

    def run_one_turn(
        self,
        user_task: str,
        *,
        sink: Callable[[RunEvent], None],
        confirm: Callable[[ApprovalRequest], bool],
        abort: AbortSignal | None = None,
        transient_fragments: tuple[Any, ...] = (),
        skills_snapshot: Any = None,
        launch_context: Any = None,
        confirm_proxy: Any = None,
    ) -> TurnExecution:
        return self._behavior(user_task, sink, confirm, abort)

    def render_turn_epilogue(self, execution):
        self.epilogues.append(execution)


def _patch(path="a.txt"):
    return ApprovalRequest(
        kind="patch", tool_name="edit_file", preview="diff", path=path
    )


def _runner(confirm_inputs=()):
    out = io.StringIO()
    renderer = PlainRenderer(out, out)
    controller = ConfirmationController(renderer, StdinInputController(lambda _p: ""))
    return TurnRunner(renderer, controller, join_timeout=2.0), renderer, out


class TurnRunnerTest(unittest.TestCase):
    def test_events_are_rendered_in_order(self):
        def behavior(task, sink, confirm, abort):
            sink(
                RunEvent(
                    "tool_call_started",
                    {"tool_name": "read_file", "tool_call_id": "c1"},
                )
            )
            sink(
                RunEvent(
                    "tool_call_finished",
                    {"tool_name": "read_file", "tool_call_id": "c1", "ok": True},
                )
            )
            return TurnExecution(outcome=InputOutcome(action="continue"))

        runner, _renderer, out = _runner()
        outcome = runner.run(_StubBridge(behavior), "go")
        self.assertEqual(outcome.action, "continue")
        text = out.getvalue()
        self.assertLess(text.index("read_file"), text.index("read_file ok"))

    def test_confirm_thread_bridge_blocks_and_replies(self):
        seen = {}

        def behavior(task, sink, confirm, abort):
            decision = confirm(_patch())
            seen["decision"] = decision
            return TurnExecution(outcome=InputOutcome(action="continue"))

        out = io.StringIO()
        renderer = PlainRenderer(out, out)
        # chooser always says yes
        controller = ConfirmationController(
            renderer, StdinInputController(lambda _p: ""), chooser=lambda _r: "yes"
        )
        runner = TurnRunner(renderer, controller, join_timeout=2.0)
        outcome = runner.run(_StubBridge(behavior), "go")
        self.assertEqual(outcome.action, "continue")
        self.assertTrue(seen["decision"])

    def test_error_surfaced_as_outcome(self):
        def behavior(task, sink, confirm, abort):
            raise RuntimeError("kaboom")

        runner, _renderer, _out = _runner()
        outcome = runner.run(_StubBridge(behavior), "go")
        self.assertEqual(outcome.action, "error")
        error = outcome.error
        assert error is not None
        self.assertIn("kaboom", error)

    def test_aborted_worker_yields_cancel_outcome(self):
        def behavior(task, sink, confirm, abort):
            raise Aborted()

        runner, _renderer, out = _runner()
        outcome = runner.run(_StubBridge(behavior), "go")
        self.assertEqual(outcome.action, "continue")
        self.assertIn("已中断", out.getvalue())

    def test_epilogue_called_on_success(self):
        execution = TurnExecution(
            outcome=InputOutcome(action="continue"), recorded_message="rec"
        )

        def behavior(task, sink, confirm, abort):
            return execution

        bridge = _StubBridge(behavior)
        runner, _renderer, _out = _runner()
        runner.run(bridge, "go")
        self.assertEqual(bridge.epilogues, [execution])

    def test_confirm_proxy_aborted_sentinel_raises(self):
        import queue

        q: queue.Queue = queue.Queue()
        proxy = ConfirmProxy(q)
        result = {}

        def worker():
            try:
                proxy(_patch())
            except Aborted:
                result["aborted"] = True

        t = threading.Thread(target=worker)
        t.start()
        # main side: pull the ticket and reply ABORTED
        kind, ticket = q.get(timeout=2.0)
        self.assertEqual(kind, "confirm")
        ticket.reply(ABORTED)
        t.join(timeout=2.0)
        self.assertTrue(result.get("aborted"))


if __name__ == "__main__":
    unittest.main()
