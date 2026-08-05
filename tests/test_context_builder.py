import tempfile
import unittest
from pathlib import Path

from forestcode.context import ContextBudget, ContextRequest, ToolCatalog
from forestcode.context.builder import ContextBuilder
from forestcode.context.providers import SessionContextProvider
from forestcode.core.run_state import RunState
from forestcode.core.types import AssistantTurn, ReasoningArtifact, ToolCall, ToolResult
from forestcode.memory import MemoryEntry, SessionCompressor, SessionStore, build_compaction_entry
from forestcode.tools import ToolDefinition, ToolRegistry, create_builtin_tool_registry


class ContextBuilderTest(unittest.TestCase):
    def test_builds_model_input_with_tool_schemas_and_metadata(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="read_file",
                    description="Read a file.",
                    input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
                    runner=lambda _context: "ok",
                )
            ]
        )
        state = RunState.start("read README")

        model_input = ContextBuilder(tool_catalog=ToolCatalog(registry)).build(state)

        self.assertEqual(model_input.tools[0]["function"]["name"], "read_file")
        self.assertEqual(model_input.metadata["selected_tools"], ["read_file"])
        self.assertTrue(model_input.system_prompt)

    def test_memory_write_prompt_is_derived_from_visible_save_memory_tool(self):
        registry = create_builtin_tool_registry()
        state = RunState.start("remember preference")

        model_input = ContextBuilder(
            tool_catalog=ToolCatalog(registry, read_only_only=False),
        ).build(state)

        self.assertIn("save_memory tool", model_input.system_prompt)
        self.assertIn("save_memory", model_input.metadata["selected_tools"])

    def test_memory_write_prompt_is_absent_when_save_memory_is_not_visible(self):
        registry = create_builtin_tool_registry(enable_memory_write=False)
        state = RunState.start("remember preference")

        model_input = ContextBuilder(
            tool_catalog=ToolCatalog(registry, read_only_only=False),
        ).build(state)

        self.assertNotIn("save_memory tool", model_input.system_prompt)
        self.assertNotIn("save_memory", model_input.metadata["selected_tools"])

    def test_orphan_current_run_tool_result_is_downgraded_to_user_summary(self):
        state = RunState.start("inspect")
        state.add_tool_result(
            ToolResult(
                tool_call_id="call_1",
                tool_name="read_file",
                ok=True,
                content="full content that should not be preferred",
                summary="short summary",
            )
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_input = ContextBuilder(workspace_root=Path(tmp)).build(state)

        self.assertEqual(model_input.messages[-1].role, "user")
        self.assertIn("Orphan tool result summary", model_input.messages[-1].content)
        self.assertEqual(len([message for message in model_input.messages if message.role == "tool_result"]), 0)

    def test_keeps_large_tool_group_together_when_it_exceeds_recent_message_cap(self):
        state = RunState.start("inspect")
        artifact = ReasoningArtifact(
            provider="deepseek",
            kind="reasoning_content",
            payload={"reasoning_content": "inspect many paths"},
            required_for_followup=True,
            visible=True,
            display_text="inspect many paths",
        )
        calls = [
            ToolCall(id=f"call_{index}", name="list_files", arguments={"path": "."})
            for index in range(25)
        ]
        state.add_assistant_turn(AssistantTurn(tool_calls=calls, reasoning_artifacts=[artifact]))
        for call in calls:
            state.add_tool_result(
                ToolResult(
                    tool_call_id=call.id,
                    tool_name=call.name,
                    ok=True,
                    content="file README.md",
                )
            )

        model_input = ContextBuilder(budget=ContextBudget(max_recent_messages=20)).build(state)

        self.assertEqual(model_input.messages[0].role, "user")
        assistant_messages = [message for message in model_input.messages if message.role == "assistant"]
        tool_results = [message for message in model_input.messages if message.role == "tool_result"]
        self.assertEqual(len(assistant_messages), 1)
        self.assertEqual(assistant_messages[0].tool_calls, calls)
        self.assertEqual(assistant_messages[0].reasoning_artifacts, [artifact])
        self.assertEqual(len(tool_results), 25)
        self.assertEqual({result.tool_call_id for result in tool_results}, {call.id for call in calls})

    def test_preserves_multiple_tool_turn_groups(self):
        state = RunState.start("inspect")
        first_artifact = ReasoningArtifact(
            provider="deepseek",
            kind="reasoning_content",
            payload={"reasoning_content": "inspect root"},
            required_for_followup=True,
        )
        second_artifact = ReasoningArtifact(
            provider="deepseek",
            kind="reasoning_content",
            payload={"reasoning_content": "inspect src"},
            required_for_followup=True,
        )
        first_call = ToolCall(id="call_1", name="list_files", arguments={"path": "."})
        second_call = ToolCall(id="call_2", name="list_files", arguments={"path": "src"})
        state.add_assistant_turn(AssistantTurn(tool_calls=[first_call], reasoning_artifacts=[first_artifact]))
        state.add_tool_result(ToolResult(tool_call_id="call_1", tool_name="list_files", ok=True, content="dir src"))
        state.add_assistant_turn(AssistantTurn(tool_calls=[second_call], reasoning_artifacts=[second_artifact]))
        state.add_tool_result(
            ToolResult(tool_call_id="call_2", tool_name="list_files", ok=True, content="dir src/forestcode")
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_input = ContextBuilder(workspace_root=Path(tmp)).build(state)

        messages = model_input.messages
        self.assertEqual([message.role for message in messages], ["user", "assistant", "tool_result", "assistant", "tool_result"])
        self.assertEqual(messages[1].reasoning_artifacts, [first_artifact])
        self.assertEqual(messages[3].reasoning_artifacts, [second_artifact])
        self.assertEqual(messages[2].tool_call_id, "call_1")
        self.assertEqual(messages[4].tool_call_id, "call_2")

    def test_preserves_tool_error_result_in_tool_group(self):
        state = RunState.start("read missing file")
        call = ToolCall(id="call_1", name="read_file", arguments={"path": "missing.md"})
        state.add_assistant_turn(AssistantTurn(tool_calls=[call]))
        state.add_tool_result(
            ToolResult(
                tool_call_id="call_1",
                tool_name="read_file",
                ok=False,
                content="",
                error="File not found",
            )
        )

        model_input = ContextBuilder().build(state)

        self.assertEqual(model_input.messages[-2].role, "assistant")
        self.assertEqual(model_input.messages[-1].role, "tool_result")
        self.assertEqual(model_input.messages[-1].tool_call_id, "call_1")
        self.assertIn("error:read_file:call_1:File not found", model_input.messages[-1].content)

    def test_reads_memory_md_and_marks_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MEMORY.md").write_text("abcdef", encoding="utf-8")
            state = RunState.start("task")

            model_input = ContextBuilder(
                workspace_root=root,
                budget=ContextBudget(max_memory_chars=3),
            ).build(state)

            self.assertIn("Long-term memory", model_input.messages[0].content)
            self.assertTrue(model_input.metadata["truncated"])
            self.assertIn("MEMORY.md", model_input.metadata["context_sources"])

    def test_reads_existing_session_entries_without_compacting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            store.append_entry("default", MemoryEntry(kind="message", role="user", content="previous task"))
            store.append_entry("default", MemoryEntry(kind="message", role="assistant", content="previous answer"))
            store.append_entry("default", MemoryEntry(kind="tool_result", content="ok:read_file:call_1:summary"))
            store.append_entry("default", MemoryEntry(kind="compaction", content="older history summary"))

            model_input = ContextBuilder(
                workspace_root=root,
                request=ContextRequest(session_id="default"),
                session_provider=SessionContextProvider(store),
            ).build(RunState.start("current task"))

            contents = [message.content for message in model_input.messages]
            self.assertIn("Session compaction summary:\nolder history summary", contents)
            self.assertIn("previous task", contents)
            self.assertIn("Historical assistant message:\nprevious answer", contents)
            self.assertIn("Historical tool result summary:\nok:read_file:call_1:summary", contents)
            historical_roles = [message.role for message in model_input.messages[:-1]]
            self.assertNotIn("assistant", historical_roles)
            self.assertNotIn("tool_result", historical_roles)
            self.assertEqual(model_input.metadata["session_entry_count"], 4)
            self.assertIn(".forestcode/sessions/default.jsonl", model_input.metadata["context_sources"])

    def test_boundary_driven_session_context_uses_latest_major_active_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            store.append_entry("default", MemoryEntry(kind="message", role="user", content="very old"))
            store.append_entry(
                "default",
                build_compaction_entry(
                    "normal before major",
                    compaction_kind="normal",
                    first_kept_entry_id="entry_000001",
                ),
            )
            store.append_entry(
                "default",
                build_compaction_entry(
                    "major summary",
                    compaction_kind="major",
                    first_kept_entry_id="entry_000004",
                ),
            )
            store.append_entry("default", MemoryEntry(kind="message", role="user", content="after major raw"))
            store.append_entry(
                "default",
                build_compaction_entry(
                    "normal after major",
                    compaction_kind="normal",
                    first_kept_entry_id="entry_000006",
                ),
            )
            store.append_entry("default", MemoryEntry(kind="message", role="user", content="after normal raw"))

            model_input = ContextBuilder(
                workspace_root=root,
                request=ContextRequest(session_id="default"),
                session_provider=SessionContextProvider(store),
            ).build(RunState.start("current task"))
            contents = [message.content for message in model_input.messages]

            self.assertTrue(any("This session is being continued" in content for content in contents))
            self.assertTrue(any("major summary" in content for content in contents))
            self.assertTrue(any("Continue from where it left off without asking" in content for content in contents))
            self.assertTrue(any("Resume directly — do not acknowledge the summary" in content for content in contents))
            normal_summary = [content for content in contents if content.startswith("Session compaction summary:\nnormal after major")][0]
            self.assertEqual(normal_summary, "Session compaction summary:\nnormal after major")
            self.assertNotIn("Continue from where it left off", normal_summary)
            self.assertIn("after normal raw", contents)
            self.assertFalse(any("normal before major" in content for content in contents))
            self.assertFalse(any("after major raw" == content for content in contents))
            stored = store.load("default")
            self.assertFalse(any("status" in entry.metadata for entry in stored.entries if entry.kind == "compaction"))

    def test_current_run_tool_results_are_microcompacted_by_budget(self):
        state = RunState.start("inspect")
        call = ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})
        state.add_assistant_turn(AssistantTurn(tool_calls=[call]))
        state.add_tool_result(
            ToolResult(
                tool_call_id="call_1",
                tool_name="read_file",
                ok=True,
                content="x" * 80,
            )
        )

        model_input = ContextBuilder(budget=ContextBudget(max_tool_result_chars=30)).build(state)

        tool_result = [message for message in model_input.messages if message.role == "tool_result"][0]
        self.assertLessEqual(len(tool_result.content), 60)
        self.assertIn("<truncated", tool_result.content)

    def test_historical_tool_results_are_microcompacted_by_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            store.append_entry("default", MemoryEntry(kind="tool_result", content="ok:read_file:call_1:" + "x" * 80))

            model_input = ContextBuilder(
                workspace_root=root,
                request=ContextRequest(session_id="default"),
                session_provider=SessionContextProvider(store),
                budget=ContextBudget(max_tool_result_chars=30),
            ).build(RunState.start("current task"))

            historical = [message for message in model_input.messages if "Historical tool result summary" in message.content][0]
            self.assertIn("<truncated", historical.content)

    def test_counts_tool_call_arguments_without_content(self):
        state = RunState.start("inspect")
        state.add_tool_calls([ToolCall(id="call_1", name="read_file", arguments={"path": "README.md"})])

        model_input = ContextBuilder().build(state)

        self.assertGreater(model_input.metadata["char_count"], len(model_input.system_prompt or ""))

    def test_preserves_current_run_reasoning_artifacts_and_counts_payload(self):
        state = RunState.start("inspect")
        artifact = ReasoningArtifact(
            provider="deepseek",
            kind="reasoning_content",
            payload={"reasoning_content": "thinking"},
            visible=True,
            display_text="thinking",
        )
        state.add_assistant_turn(AssistantTurn(text="done", reasoning_artifacts=[artifact]))

        model_input = ContextBuilder().build(state)

        assistant_message = model_input.messages[-1]
        self.assertEqual(assistant_message.reasoning_artifacts, [artifact])
        self.assertGreaterEqual(model_input.metadata["char_count"], len("thinking"))

    def test_over_context_budget_marks_major_compact_needed(self):
        state = RunState.start("x" * 200)

        model_input = ContextBuilder(budget=ContextBudget(max_context_chars=10)).build(state)

        self.assertTrue(model_input.metadata["needs_major_compact"])
        self.assertTrue(model_input.metadata["truncated"])

    def test_major_compaction_can_reduce_history_context_below_headroom(self):
        class StaticSummarizer:
            def summarize(self, entries, prior_summary):
                return "short major summary"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root)
            for index in range(5):
                role = "user" if index % 2 == 0 else "assistant"
                store.append_entry("default", MemoryEntry(kind="message", role=role, content=f"{index}-" + "x" * 250))
            budget = ContextBudget(max_context_chars=1_400, max_session_summary_chars=300, max_recent_messages=20)
            builder = ContextBuilder(
                workspace_root=root,
                request=ContextRequest(session_id="default"),
                session_provider=SessionContextProvider(store),
                budget=budget,
            )

            before = builder.build(RunState.start("current task"))
            compacted = SessionCompressor(
                store,
                "default",
                StaticSummarizer(),
                keep_recent_entries=1,
                compact_trigger_entries=2,
                max_summary_chars=300,
            ).maybe_major_compact()
            after = builder.build(RunState.start("current task"))

            self.assertTrue(before.metadata["needs_major_compact"])
            self.assertTrue(compacted)
            self.assertLess(after.metadata["char_count"], before.metadata["char_count"])
            self.assertFalse(after.metadata["needs_major_compact"])


if __name__ == "__main__":
    unittest.main()
