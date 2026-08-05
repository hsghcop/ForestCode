import threading
import time
import unittest
from datetime import UTC, datetime
from pathlib import Path

from forestcode.core import (
    AgentLoop,
    ContextBuilder,
    InMemoryEventSink,
    MaxTurnsStopPolicy,
    ModelOutput,
    ToolCall,
    ToolExecutor,
    TurnProcessor,
)
from forestcode.core.abort import Aborted, AbortSignal
from forestcode.tools import CommandService
from forestcode.tools.types import CommandProposal


def _build_loop(model, tool_executor, signal, max_turns=10):
    events = InMemoryEventSink()
    loop = AgentLoop(
        model=model,
        context_builder=ContextBuilder(),
        turn_processor=TurnProcessor(),
        tool_executor=tool_executor,
        events=events,
        stop_policy=MaxTurnsStopPolicy(max_turns=max_turns),
        abort=signal,
    )
    return loop, events


class _AbortingModel:
    """Sets the abort signal during complete(), then returns a tool call."""

    def __init__(self, signal: AbortSignal) -> None:
        self._signal = signal
        self.calls = 0

    def complete(self, model_input, *, abort=None):
        self.calls += 1
        self._signal.set()
        return ModelOutput(tool_calls=[ToolCall(id="c1", name="noop", arguments={})])


class _CountingModel:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, model_input, *, abort=None):
        self.calls += 1
        return ModelOutput(text="done")


class AgentLoopAbortTest(unittest.TestCase):
    def test_abort_during_model_stops_before_tools(self):
        signal = AbortSignal()
        ran: list[str] = []
        executor = ToolExecutor({"noop": lambda args: ran.append("ran") or "ok"})
        model = _AbortingModel(signal)
        loop, events = _build_loop(model, executor, signal)

        with self.assertRaises(Aborted):
            loop.run("hi")

        self.assertEqual(model.calls, 1)
        self.assertEqual(ran, [])  # checkpoint ④ stops before tool execution
        self.assertIn("run_cancelled", [e.type for e in events.events])

    def test_preset_abort_skips_model_entirely(self):
        signal = AbortSignal()
        signal.set()
        model = _CountingModel()
        loop, events = _build_loop(model, ToolExecutor({}), signal)

        with self.assertRaises(Aborted):
            loop.run("hi")

        self.assertEqual(model.calls, 0)  # checkpoint ① stops at loop top
        self.assertIn("run_cancelled", [e.type for e in events.events])

    def test_no_abort_runs_normally(self):
        # Regression guard: abort=None path is unaffected.
        model = _CountingModel()
        loop, _ = _build_loop(model, ToolExecutor({}), None)
        state = loop.run("hi")
        self.assertEqual(state.final_text, "done")


def _proposal(command: str, cwd: Path, timeout: int = 30) -> CommandProposal:
    return CommandProposal(
        id="cmd1",
        command=command,
        cwd=cwd,
        timeout=timeout,
        shell_label="test",
        is_dangerous=False,
        display=command,
        status="proposed",
        tool_call_id="t1",
        created_at=datetime.now(UTC).isoformat(),
    )


class CommandCancelTest(unittest.TestCase):
    def test_blocking_path_unchanged_without_abort(self):
        import sys

        proposal = _proposal(f'{sys.executable} -c "print(1)"', Path.cwd())
        CommandService().execute(proposal)  # abort=None
        self.assertEqual(proposal.status, "executed")
        self.assertEqual(proposal.exit_code, 0)

    def test_abort_terminates_running_command(self):
        import sys

        signal = AbortSignal()
        proposal = _proposal(
            f'{sys.executable} -c "import time; time.sleep(20)"',
            Path.cwd(),
            timeout=30,
        )

        def killer():
            time.sleep(0.4)
            signal.set()

        threading.Thread(target=killer, daemon=True).start()
        start = time.monotonic()
        with self.assertRaises(Aborted):
            CommandService().execute(proposal, abort=signal)
        # Should return well before the 20s sleep / 30s timeout.
        self.assertLess(time.monotonic() - start, 10)


if __name__ == "__main__":
    unittest.main()
