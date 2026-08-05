"""Permission composition, mutation gate and per-child approval cancellation (R5/R6/R7).

design §Permission Composition, §Mutation Gate and Approval. Offline only.
- ``ToolVisibilityPolicy``: profile -> allow -> deny -> no-subagent-tools, with
  the parent-visible catalog as the hard ceiling.
- ``MutationGate``: serializes the full patch/save-memory/command section; the
  stale content-hash check then rejects a second writer.
- ``ConfirmProxy.cancel_task``: cancelling one child resolves only its own
  tickets; other children and the untagged parent ticket are untouched.
"""

from __future__ import annotations

import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path

from forestcode.context.tool_catalog import ToolVisibilityPolicy
from forestcode.core.abort import Aborted, AbortSignal
from forestcode.core.run_state import RunState
from forestcode.core.tool_executor import ToolExecutor
from forestcode.core.types import ToolCall
from forestcode.subagents.coordinator import SubagentCoordinator
from forestcode.subagents.types import SubagentRequest, SubagentResult
from forestcode.terminal.turn_runner import ConfirmProxy
from forestcode.tools import (
    ApprovalRequest,
    MutationGate,
    PatchService,
    ReadStateStore,
    ToolRuntimeServices,
    create_builtin_tool_registry,
)

# Parent-visible catalog the filter is applied to (hard ceiling). Includes the
# delegation tools, which must never survive into a child catalog.
PARENT_VISIBLE = frozenset(
    {
        "list_files",
        "glob_files",
        "grep_text",
        "read_file",
        "get_file_info",
        "read_session_history",
        "load_skill",
        "edit_file",
        "write_file",
        "save_memory",
        "run_command",
        "write_todos",
        "delegate_task",
        "wait_subagents",
        "list_subagents",
        "cancel_subagent",
    }
)

READ_TOOLS = frozenset(
    {
        "list_files",
        "glob_files",
        "grep_text",
        "read_file",
        "get_file_info",
        "read_session_history",
        "load_skill",
    }
)


class ToolVisibilityPolicyTest(unittest.TestCase):
    def test_research_is_read_only(self):
        policy = ToolVisibilityPolicy(PARENT_VISIBLE, "research")
        self.assertEqual(policy.visible_names, READ_TOOLS)

    def test_verify_adds_command(self):
        policy = ToolVisibilityPolicy(PARENT_VISIBLE, "verify")
        self.assertEqual(policy.visible_names, READ_TOOLS | {"run_command"})

    def test_edit_adds_patch_tools(self):
        policy = ToolVisibilityPolicy(PARENT_VISIBLE, "edit")
        self.assertEqual(
            policy.visible_names,
            READ_TOOLS | {"edit_file", "write_file", "save_memory"},
        )

    def test_full_inherits_parent_visible_except_delegation(self):
        policy = ToolVisibilityPolicy(PARENT_VISIBLE, "full")
        self.assertEqual(
            policy.visible_names,
            PARENT_VISIBLE
            - {"delegate_task", "wait_subagents", "list_subagents", "cancel_subagent"},
        )

    def test_allow_can_expand_but_not_beyond_parent(self):
        # read_session_history is not in the parent set here: allow cannot
        # introduce it.
        parent = PARENT_VISIBLE - {"read_session_history"}
        policy = ToolVisibilityPolicy(
            parent, "research", allow=("read_session_history",)
        )
        self.assertNotIn("read_session_history", policy.visible_names)
        # run_command can be added to research via explicit allow (parent has it)
        policy = ToolVisibilityPolicy(parent, "research", allow=("run_command",))
        self.assertIn("run_command", policy.visible_names)

    def test_deny_always_wins(self):
        policy = ToolVisibilityPolicy(PARENT_VISIBLE, "edit", deny=("edit_file",))
        self.assertNotIn("edit_file", policy.visible_names)
        # deny removes delegation tools that would be removed anyway
        policy = ToolVisibilityPolicy(
            PARENT_VISIBLE, "full", deny=("list_files", "grep_text")
        )
        self.assertNotIn("list_files", policy.visible_names)
        self.assertNotIn("grep_text", policy.visible_names)
        self.assertIn("read_file", policy.visible_names)

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            ToolVisibilityPolicy(PARENT_VISIBLE, "admin")  # type: ignore[arg-type]

    def test_filter_applies_to_tool_definitions(self):
        registry = create_builtin_tool_registry()
        names = {tool.name for tool in registry.list_tools()}
        parent = names - {
            "delegate_task",
            "wait_subagents",
            "list_subagents",
            "cancel_subagent",
        }
        policy = ToolVisibilityPolicy(parent, "research")
        filtered = policy.filter(registry.list_tools())
        self.assertTrue(filtered)
        for tool in filtered:
            self.assertIn(tool.name, READ_TOOLS)


class MutationGateTest(unittest.TestCase):
    def test_serializes_across_threads(self):
        gate = MutationGate()
        a_entered = threading.Event()
        a_release = threading.Event()
        b_entered = threading.Event()

        def worker_a() -> None:
            with gate:
                a_entered.set()
                a_release.wait()

        def worker_b() -> None:
            with gate:
                b_entered.set()

        ta = threading.Thread(target=worker_a)
        tb = threading.Thread(target=worker_b)
        ta.start()
        self.assertTrue(a_entered.wait(5))
        tb.start()
        # b must not enter while a holds the gate
        self.assertFalse(b_entered.wait(0.2))
        a_release.set()
        self.assertTrue(b_entered.wait(5))
        ta.join(5)
        tb.join(5)

    def test_reentrant_for_same_thread(self):
        gate = MutationGate()
        # Same thread may re-enter the RLock-backed gate; both entries must be
        # released before another thread can acquire it.
        gate.__enter__()
        gate.__enter__()
        gate.__exit__(None, None, None)
        entered = threading.Event()

        def worker() -> None:
            with gate:
                entered.set()

        t = threading.Thread(target=worker)
        t.start()
        self.assertFalse(entered.wait(0.2))  # still held by this thread
        gate.__exit__(None, None, None)
        self.assertTrue(entered.wait(5))
        t.join(5)


class MutationGateToolExecutorTest(unittest.TestCase):
    """Two concurrent writers on the same file: serialized sections, stale
    hash rejected — no silent clobber (AC6)."""

    def _executor(
        self,
        root: Path,
        gate: MutationGate,
        confirm=None,
        abort: AbortSignal | None = None,
    ) -> ToolExecutor:
        store = ReadStateStore()
        patch_service = PatchService(read_state_store=store)
        runtime = ToolRuntimeServices(
            read_state_store=store,
            patch_service=patch_service,
            confirm=confirm or (lambda _request: True),
            mutation_gate=gate,
        )
        return ToolExecutor(
            create_builtin_tool_registry(),
            workspace_root=root,
            runtime=runtime,
            abort=abort,
        )

    def _read(self, executor: ToolExecutor, path: str) -> None:
        executor.execute(
            ToolCall(
                id=f"read_{path}",
                name="read_file",
                arguments={"path": path, "offset": 0, "limit": 20_000},
            ),
            RunState.start("test"),
        )

    def test_second_writer_rejected_after_first_applies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "app.py"
            target.write_text("x = 1\n", encoding="utf-8")
            gate = MutationGate()
            # Two children with independent ReadStateStore/PatchService
            executor_a = self._executor(root, gate)
            executor_b = self._executor(root, gate)

            # Both read the same original content first
            self._read(executor_a, "app.py")
            self._read(executor_b, "app.py")

            first = executor_a.execute(
                ToolCall(
                    id="edit-a",
                    name="edit_file",
                    arguments={
                        "path": "app.py",
                        "old_text": "x = 1",
                        "new_text": "x = 2",
                    },
                ),
                RunState.start("test"),
            )
            self.assertTrue(first.ok)
            self.assertEqual(target.read_text(encoding="utf-8"), "x = 2\n")

            # Child B's proposal was based on the old read state; the section is
            # gated so it proposes against the new file state and the stale
            # hash check rejects the edit.
            second = executor_b.execute(
                ToolCall(
                    id="edit-b",
                    name="edit_file",
                    arguments={
                        "path": "app.py",
                        "old_text": "x = 1",
                        "new_text": "x = 3",
                    },
                ),
                RunState.start("test"),
            )
            self.assertFalse(second.ok)
            self.assertIn("changed", second.error or "")
            self.assertEqual(target.read_text(encoding="utf-8"), "x = 2\n")

    def test_sequential_reads_are_not_gated(self):
        """Concurrent read-only tools proceed without the gate (no deadlock)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "app.py").write_text("x = 1\n", encoding="utf-8")
            gate = MutationGate()
            executor = self._executor(root, gate)
            entered = threading.Event()
            release = threading.Event()

            def hold_gate() -> None:
                with gate:
                    entered.set()
                    release.wait()

            holder = threading.Thread(target=hold_gate)
            holder.start()
            self.assertTrue(entered.wait(5))
            result = executor.execute(
                ToolCall(
                    id="read",
                    name="read_file",
                    arguments={"path": "app.py", "offset": 0, "limit": 20_000},
                ),
                RunState.start("test"),
            )
            self.assertTrue(result.ok)  # read is not blocked by the held gate
            release.set()
            holder.join(5)

    def test_cancel_during_approval_prevents_patch_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            abort = AbortSignal()

            def confirm(_request) -> bool:
                abort.set()
                return True

            executor = self._executor(
                root, MutationGate(), confirm=confirm, abort=abort
            )
            with self.assertRaises(Aborted):
                executor.execute(
                    ToolCall(
                        id="write",
                        name="write_file",
                        arguments={"path": "cancelled.txt", "content": "no"},
                    ),
                    RunState.start("test"),
                )
            self.assertFalse((root / "cancelled.txt").exists())


class ConfirmProxyCancelTest(unittest.TestCase):
    def _proxy(self) -> tuple[ConfirmProxy, queue.Queue]:
        q: queue.Queue = queue.Queue()
        return ConfirmProxy(q), q

    def _call(
        self, proxy: ConfirmProxy, task_id: str | None, results: dict, key: str
    ) -> None:
        try:
            proxy(
                ApprovalRequest(
                    kind="patch", tool_name="edit_file", preview="diff", path="a.txt"
                ),
                task_id=task_id,
            )
            results[key] = "ok"
        except Aborted:
            results[key] = "aborted"

    def test_cancel_task_releases_only_its_own_ticket(self):
        proxy, q = self._proxy()
        results: dict[str, str] = {}

        ta = threading.Thread(target=self._call, args=(proxy, "child-a", results, "a"))
        tb = threading.Thread(target=self._call, args=(proxy, "child-b", results, "b"))
        ta.start()
        tb.start()

        tickets: dict[str, object] = {}
        for _ in range(2):
            kind, ticket = q.get(timeout=5)
            self.assertEqual(kind, "confirm")
            tickets[ticket.task_id] = ticket  # type: ignore[attr-defined]

        # Cancel child-a only: its worker unblocks with Aborted, b keeps waiting.
        proxy.cancel_task("child-a")
        ta.join(5)
        self.assertEqual(results["a"], "aborted")
        self.assertNotIn("b", results)  # b still pending

        # Approve b normally: it completes.
        tickets["child-b"].reply(True)  # type: ignore[attr-defined]
        tb.join(5)
        self.assertEqual(results["b"], "ok")

    def test_untagged_ticket_unaffected_by_cancel_task(self):
        proxy, q = self._proxy()
        results: dict[str, str] = {}
        # parent (untagged) ticket + one child ticket
        tp = threading.Thread(target=self._call, args=(proxy, None, results, "parent"))
        tc = threading.Thread(
            target=self._call, args=(proxy, "child", results, "child")
        )
        tp.start()
        tc.start()
        tickets = {}
        for _ in range(2):
            _kind, ticket = q.get(timeout=5)
            tickets[ticket.task_id] = ticket  # type: ignore[attr-defined]

        proxy.cancel_task("child")
        tc.join(5)
        self.assertEqual(results["child"], "aborted")
        self.assertNotIn("parent", results)

        tickets[None].reply(False)  # type: ignore[index]
        tp.join(5)
        self.assertEqual(results["parent"], "ok")  # bool(False) is not Aborted

    def test_cancel_task_idempotent_and_unknown_id_safe(self):
        proxy, q = self._proxy()
        proxy.cancel_task("no-such-child")  # no-op
        proxy.cancel_task("no-such-child")  # still no-op

        results: dict[str, str] = {}
        tc = threading.Thread(target=self._call, args=(proxy, "child", results, "c"))
        tc.start()
        _kind, _ticket = q.get(timeout=5)
        proxy.cancel_task("child")
        proxy.cancel_task("child")  # idempotent: ticket already resolved
        tc.join(5)
        self.assertEqual(results["c"], "aborted")

    def test_already_aborted_child_never_blocks_or_shows_stale_approval(self):
        proxy, q = self._proxy()
        abort = AbortSignal()
        abort.set()
        result: list[str] = []

        def call() -> None:
            try:
                proxy(
                    ApprovalRequest(
                        kind="patch", tool_name="edit_file", preview="diff", path="a"
                    ),
                    task_id="child",
                    abort=abort,
                )
            except Aborted:
                result.append("aborted")

        thread = threading.Thread(target=call)
        thread.start()
        thread.join(5)
        self.assertEqual(result, ["aborted"])
        kind, ticket = q.get(timeout=5)
        self.assertEqual(kind, "confirm")
        self.assertTrue(ticket.answered)
        self.assertEqual(proxy._tickets, {})


class _BlockingRunner:
    def __init__(self, result: SubagentResult | None = None) -> None:
        self.result = result or SubagentResult(
            task_id="unused", agent_name="a", final_text="done"
        )
        self.release = threading.Event()
        self.started = threading.Event()

    def run(self, request: SubagentRequest, *, abort) -> SubagentResult:  # type: ignore[no-untyped-def]
        self.started.set()
        self.release.wait()
        return SubagentResult(
            task_id=request.task_id,
            agent_name=request.agent_name,
            final_text="done",
            turn_count=1,
        )


class ApprovalTimeoutSemanticsTest(unittest.TestCase):
    """waiting_approval time must not count toward the task deadline (design)."""

    def test_approval_pause_resumes_deadline(self):
        runner = _BlockingRunner()
        coordinator = SubagentCoordinator(
            runner,
            max_workers=1,
            default_task_timeout_seconds=0.25,
            id_factory=lambda: "t1",
        )
        self.addCleanup(coordinator.cleanup)
        coordinator.delegate(
            SubagentRequest(task_id="x", agent_name="a", description="d", prompt="p")
        )
        self.assertTrue(runner.started.wait(5))
        # Simulate the child entering approval (Step 4 wires this transition).
        with coordinator._lock:
            coordinator._transition("t1", "waiting_approval")
        # Wait past the original deadline while paused (wall-clock sleep is
        # inherent to a deadline test): the task must NOT time out while
        # waiting for approval.
        time.sleep(0.4)
        with coordinator._lock:
            entry = coordinator._tasks["t1"]
            self.assertEqual(entry.status, "waiting_approval")
            coordinator._transition("t1", "running")
            self.assertEqual(entry.status, "running")
        # Resumed and finished immediately: if approval time had counted toward
        # the deadline, the watchdog would have moved it to cancelling(timeout)
        # already and this wait would report cancelled, not completed.
        runner.release.set()
        outcome = coordinator.wait(["t1"], timeout=5)
        self.assertFalse(outcome.timed_out)
        self.assertEqual(outcome.snapshots[0].status, "completed")


if __name__ == "__main__":
    unittest.main()
