import unittest

from forestcode.core import (
    AgentLoop,
    ContextBuilder,
    FakeModelClient,
    InMemoryEventSink,
    MaxTurnsStopPolicy,
    ModelOutput,
    ToolCall,
    ToolExecutor,
    TurnProcessor,
)


def build_loop(
    model: FakeModelClient,
    tool_executor: ToolExecutor,
    max_turns: int = 10,
    memory_manager=None,
    compaction_controller=None,
    subagents=None,
):
    events = InMemoryEventSink()
    loop = AgentLoop(
        model=model,
        context_builder=ContextBuilder(),
        turn_processor=TurnProcessor(),
        tool_executor=tool_executor,
        events=events,
        stop_policy=MaxTurnsStopPolicy(max_turns=max_turns),
        memory_manager=memory_manager,
        compaction_controller=compaction_controller,
        subagents=subagents,
    )
    return loop, events


class AgentLoopTest(unittest.TestCase):
    def test_tool_call_then_final_text(self):
        model = FakeModelClient(
            [
                ModelOutput(
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="calculate",
                            arguments={"expression": "1 + 1"},
                        )
                    ]
                ),
                ModelOutput(text="结果是 2"),
            ]
        )
        tools = ToolExecutor({"calculate": lambda args: "2"})
        loop, events = build_loop(model, tools)

        state = loop.run("计算 1 + 1")

        self.assertEqual(state.final_text, "结果是 2")
        self.assertIsNone(state.error)
        self.assertEqual(state.turns, 2)
        self.assertEqual(len(state.tool_calls), 1)
        self.assertEqual(len(state.tool_results), 1)
        self.assertTrue(state.tool_results[0].ok)
        self.assertEqual(state.tool_results[0].content, "2")
        self.assertEqual([event.type for event in events.events][-1], "run_finished")
        self.assertIn("assistant_text_received", [event.type for event in events.events])

        second_model_input = model.inputs[1]
        self.assertEqual(second_model_input.messages[-1].role, "tool_result")
        self.assertEqual(second_model_input.messages[-1].tool_call_id, "call_1")
        self.assertIn("2", second_model_input.messages[-1].content)

    def test_final_text_emits_assistant_text_received_before_run_finished(self):
        model = FakeModelClient([ModelOutput(text="done")])
        loop, events = build_loop(model, ToolExecutor({}))

        state = loop.run("task")

        self.assertEqual(state.final_text, "done")
        self.assertEqual(
            [event.type for event in events.events],
            [
                "run_started",
                "model_request_started",
                "model_response_received",
                "final_text_detected",
                "assistant_text_received",
                "run_finished",
            ],
        )
        self.assertEqual(events.events[-2].payload, {"text": "done"})

    def test_multiple_tool_calls_from_one_turn_keep_one_assistant_batch(self):
        model = FakeModelClient(
            [
                ModelOutput(
                    tool_calls=[
                        ToolCall(id="call_1", name="one", arguments={}),
                        ToolCall(id="call_2", name="two", arguments={}),
                    ]
                ),
                ModelOutput(text="done"),
            ]
        )
        loop, _events = build_loop(
            model,
            ToolExecutor(
                {
                    "one": lambda _args: "1",
                    "two": lambda _args: "2",
                }
            ),
        )

        state = loop.run("call both")

        assistant_tool_messages = [message for message in state.messages if message.role == "assistant" and message.tool_calls]
        self.assertEqual(len(assistant_tool_messages), 1)
        self.assertEqual([call.id for call in assistant_tool_messages[0].tool_calls], ["call_1", "call_2"])
        self.assertEqual([result.tool_call_id for result in state.tool_results], ["call_1", "call_2"])

    def test_missing_tool_returns_error_result_and_loop_continues(self):
        model = FakeModelClient(
            [
                ModelOutput(
                    tool_calls=[
                        ToolCall(
                            id="call_missing",
                            name="unknown_tool",
                            arguments={},
                        )
                    ]
                ),
                ModelOutput(text="工具不可用，已结束。"),
            ]
        )
        loop, events = build_loop(model, ToolExecutor({}))

        state = loop.run("调用未知工具")

        self.assertEqual(state.final_text, "工具不可用，已结束。")
        self.assertIsNone(state.error)
        self.assertFalse(state.tool_results[0].ok)
        self.assertEqual(state.tool_results[0].error, "Tool not found: unknown_tool")
        self.assertIn("run_finished", [event.type for event in events.events])

    def test_max_turns_failure(self):
        model = FakeModelClient(
            [
                ModelOutput(tool_calls=[ToolCall(id="call_1", name="noop", arguments={})]),
                ModelOutput(tool_calls=[ToolCall(id="call_2", name="noop", arguments={})]),
                ModelOutput(text="不会到这里"),
            ]
        )
        loop, events = build_loop(model, ToolExecutor({"noop": lambda args: "ok"}), max_turns=2)

        state = loop.run("一直调用工具")

        self.assertIsNone(state.final_text)
        self.assertEqual(state.error, "max turns reached")
        self.assertEqual(state.turns, 2)
        self.assertEqual(events.events[-1].type, "run_failed")

    def test_empty_model_output_fails_fast(self):
        model = FakeModelClient([ModelOutput()])
        loop, events = build_loop(model, ToolExecutor({}))

        state = loop.run("空输出")

        self.assertIsNone(state.final_text)
        self.assertEqual(state.error, "model returned neither final_text nor tool_calls")
        self.assertIn("empty_model_output", [event.type for event in events.events])

    def test_logical_failure_cleans_subagents_with_parent_failed(self):
        class Coordinator:
            def __init__(self) -> None:
                self.reasons = []

            def has_active_children(self):
                return True

            def cleanup(self, reason):
                self.reasons.append(reason)

        coordinator = Coordinator()
        loop, _events = build_loop(
            FakeModelClient([ModelOutput()]),
            ToolExecutor({}),
            subagents=coordinator,
        )

        state = loop.run("task")

        self.assertIsNotNone(state.error)
        self.assertEqual(coordinator.reasons, ["parent_failed"])

    def test_optional_memory_manager_records_finished_run(self):
        class RecordingMemoryManager:
            def __init__(self):
                self.states = []

            def record_run(self, state):
                self.states.append(state)

        memory_manager = RecordingMemoryManager()
        model = FakeModelClient([ModelOutput(text="done")])
        loop, events = build_loop(model, ToolExecutor({}), memory_manager=memory_manager)

        state = loop.run("task")

        self.assertEqual(memory_manager.states, [state])
        self.assertEqual([event.type for event in events.events][-1], "run_finished")

    def test_memory_manager_failure_does_not_hide_run_outcome(self):
        class FailingMemoryManager:
            def record_run(self, state):
                raise RuntimeError("disk full")

        model = FakeModelClient([ModelOutput(text="done")])
        loop, events = build_loop(model, ToolExecutor({}), memory_manager=FailingMemoryManager())

        state = loop.run("task")

        self.assertEqual(state.final_text, "done")
        self.assertIsNone(state.error)
        self.assertEqual(events.events[-1].type, "memory_record_failed")

    def test_major_compaction_runs_before_model_request_and_rebuilds(self):
        class OneShotCompaction:
            def __init__(self):
                self.calls = 0

            def maybe_major_compact(self, model_input, state):
                self.calls += 1
                return self.calls == 1

            def maybe_normal_compact(self, state):
                return False

        controller = OneShotCompaction()
        model = FakeModelClient([ModelOutput(text="done")])
        loop, events = build_loop(model, ToolExecutor({}), compaction_controller=controller)

        state = loop.run("task")

        self.assertEqual(state.final_text, "done")
        self.assertEqual(controller.calls, 1)
        self.assertEqual(len(model.inputs), 1)
        event_types = [event.type for event in events.events]
        self.assertLess(event_types.index("session_compaction_finished"), event_types.index("model_request_started"))
        self.assertEqual(events.events[event_types.index("session_compaction_finished")].payload, {"kind": "major"})

    def test_normal_compaction_runs_after_successful_record(self):
        class RecordingMemoryManager:
            def __init__(self):
                self.states = []

            def record_run(self, state):
                self.states.append(state)

        class NormalCompaction:
            def __init__(self):
                self.states = []

            def maybe_major_compact(self, model_input, state):
                return False

            def maybe_normal_compact(self, state):
                self.states.append(state)
                return True

        memory_manager = RecordingMemoryManager()
        controller = NormalCompaction()
        model = FakeModelClient([ModelOutput(text="done")])
        loop, events = build_loop(
            model,
            ToolExecutor({}),
            memory_manager=memory_manager,
            compaction_controller=controller,
        )

        state = loop.run("task")

        self.assertEqual(memory_manager.states, [state])
        self.assertEqual(controller.states, [state])
        self.assertEqual(events.events[-1].type, "session_compaction_finished")
        self.assertEqual(events.events[-1].payload, {"kind": "normal"})


if __name__ == "__main__":
    unittest.main()
