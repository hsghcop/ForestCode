"""Tests for skill context fragments and budget accounting (PRD R3, AC2, design §Context fragments)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forestcode.context import ContextBudget, ContextRequest, ContextFragment, ToolCatalog
from forestcode.context.builder import ContextBuilder
from forestcode.core.run_state import RunState
from forestcode.skills.catalog import (
    FRAGMENT_KIND_CATALOG,
    FRAGMENT_KIND_SKILL,
    SkillCatalogContextProvider,
    format_catalog,
    skill_body_fragment,
)
from forestcode.skills.loader import SkillLoader
from forestcode.skills.types import LoadedSkill, SkillDescriptor, SkillSnapshot


def _descriptor(name: str = "demo", description: str = "a demo skill") -> SkillDescriptor:
    return SkillDescriptor(
        name=name,
        description=description,
        root=Path("/nonexistent/skill"),
        entry_path=Path("/nonexistent/skill/SKILL.md"),
        source="project",
    )


class CatalogFormatTest(unittest.TestCase):
    def test_fixed_ac2_format_and_name_sorted(self):
        snapshot = SkillLoader().build_snapshot(
            project_root=Path("/nonexistent/.agents/skills"),
            user_root=Path("/nonexistent-user/.agents/skills"),
        )
        # fabricate a snapshot with two descriptors in name-sorted order
        ordered = SkillSnapshot(
            descriptors=(_descriptor("alpha", "first"), _descriptor("beta", "second"))
        )
        text = format_catalog(ordered)
        self.assertEqual(
            text,
            "Available skills (load with load_skill):\n"
            "- alpha: first\n"
            "- beta: second",
        )
        self.assertEqual(snapshot.descriptors, ())

    def test_catalog_provider_kind_and_label(self):
        snapshot = SkillSnapshot(descriptors=(_descriptor(),))
        fragment = SkillCatalogContextProvider().build(snapshot)
        self.assertEqual(fragment.kind, FRAGMENT_KIND_CATALOG)
        self.assertIn("Available skills (load with load_skill):", fragment.content)
        self.assertIn("- demo: a demo skill", fragment.content)

    def test_skill_body_fragment_kind_and_label(self):
        loaded = LoadedSkill(descriptor=_descriptor(), instructions="do stuff")
        fragment = skill_body_fragment(loaded)
        self.assertEqual(fragment.kind, FRAGMENT_KIND_SKILL)
        self.assertIn("demo", fragment.label)
        self.assertEqual(fragment.content, "do stuff")


class ContextBuilderSkillsTest(unittest.TestCase):
    def _builder(self, fragments=()):
        return ContextBuilder(
            request=ContextRequest(workspace_root=".", transient_fragments=fragments),
        )

    def test_catalog_fragment_goes_into_system_prompt(self):
        snapshot_fragment = SkillCatalogContextProvider().build(
            SkillSnapshot_for([("demo", "a demo skill")])
        )
        model_input = self._builder((snapshot_fragment,)).build(RunState.start("hi"))
        system_prompt = model_input.system_prompt
        assert system_prompt is not None
        self.assertIn("Available skills (load with load_skill):", system_prompt)
        self.assertIn("- demo: a demo skill", system_prompt)

    def test_skill_fragment_becomes_user_message_with_label(self):
        fragment = ContextFragment(
            kind=FRAGMENT_KIND_SKILL, label="Skill: demo", content="instructions body"
        )
        model_input = self._builder((fragment,)).build(RunState.start("hi"))
        message = model_input.messages[0]
        self.assertEqual(message.role, "user")
        self.assertIn("Skill: demo", message.content)
        self.assertIn("instructions body", message.content)

    def test_metadata_records_generic_source_only(self):
        fragments = (
            SkillCatalogContextProvider().build(SkillSnapshot_for([("demo", "desc")])),
            ContextFragment(kind=FRAGMENT_KIND_SKILL, label="Skill: demo", content="body"),
        )
        model_input = self._builder(fragments).build(RunState.start("hi"))
        self.assertIn("skills", model_input.metadata["context_sources"])
        serialized = str(model_input.metadata)
        self.assertNotIn("/", serialized.replace("context_sources", ""))
        self.assertNotIn("body", serialized)

    def test_char_count_includes_fragments(self):
        fragments = (
            SkillCatalogContextProvider().build(SkillSnapshot_for([("demo", "desc")])),
            ContextFragment(kind=FRAGMENT_KIND_SKILL, label="Skill: demo", content="x" * 100),
        )
        with_fragments = self._builder(fragments).build(RunState.start("hi"))
        without = self._builder(()).build(RunState.start("hi"))
        self.assertGreater(with_fragments.metadata["char_count"], without.metadata["char_count"])

    def test_no_skills_keeps_behavior_unchanged(self):
        plain = self._builder(()).build(RunState.start("hi"))
        system_prompt = plain.system_prompt
        if system_prompt is not None:
            self.assertEqual(system_prompt.find("Available skills"), -1)
        self.assertNotIn("skills", plain.metadata["context_sources"])


def SkillSnapshot_for(pairs):
    descriptors = tuple(_descriptor(name, description) for name, description in pairs)
    return SkillSnapshot(descriptors=descriptors)


class SkillResultBudgetTest(unittest.TestCase):
    def test_load_skill_result_uses_larger_budget(self):
        from forestcode.context.providers import SessionContextProvider
        from forestcode.core.types import Message

        budget = ContextBudget()
        provider = SessionContextProvider()
        long_body = "ok:load_skill:c1:" + "y" * 4000
        message = Message(role="tool_result", content=long_body, tool_call_id="c1")
        compacted = provider._compact_current_tool_result(message, budget)
        content = compacted.content
        assert content is not None
        # 4000 chars > default 2000 tool cap but < skill cap: must be preserved whole
        self.assertIn("y" * 4000, content)

    def test_regular_tool_result_still_truncated(self):
        from forestcode.context.providers import SessionContextProvider
        from forestcode.core.types import Message

        budget = ContextBudget()
        provider = SessionContextProvider()
        long_body = "ok:read_file:c1:" + "y" * 4000
        message = Message(role="tool_result", content=long_body, tool_call_id="c1")
        compacted = provider._compact_current_tool_result(message, budget)
        content = compacted.content
        assert content is not None
        self.assertIn("truncated", content)


class SkillBudgetContractTest(unittest.TestCase):
    """F1: the discovery-time loaded-skill bound plus the runtime wrapper must
    fit inside ContextBudget.max_skill_result_chars, and a maximum valid skill
    must reach the final model input complete — never truncated.
    """

    def test_budget_invariant_holds(self):
        from forestcode.skills.loader import (
            MAX_LOADED_SKILL_CHARS,
            SKILL_RESULT_WRAP_OVERHEAD,
        )

        budget = ContextBudget()
        self.assertLessEqual(
            MAX_LOADED_SKILL_CHARS + SKILL_RESULT_WRAP_OVERHEAD,
            budget.max_skill_result_chars,
        )

    def test_max_valid_skill_reaches_final_input_complete(self):
        # Build the largest skill that passes discovery, run it through the
        # real pipeline (load_skill -> ToolExecutor -> RunState ->
        # ContextProvider) and assert the final model input contains the full
        # body, description and resource list with no <truncated> marker.
        import forestcode.tools.builtin as builtin
        from forestcode.context.providers import SessionContextProvider
        from forestcode.core import ToolCall, ToolExecutor
        from forestcode.skills.loader import MAX_BODY_CHARS, MAX_LOADED_SKILL_CHARS
        from forestcode.tools import ToolRegistry, ToolRuntimeServices
        from forestcode.tools.skills import create_load_skill_tool

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace = base / "workspace"
            proj = workspace / ".agents" / "skills"
            skill_dir = proj / "big"
            skill_dir.mkdir(parents=True)
            body = "# Big skill\n" + ("x" * (MAX_BODY_CHARS - len("# Big skill\n") - 1))
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: big\ndescription: {'d' * 1500}\n---\n{body}",
                encoding="utf-8",
            )
            for index in range(20):
                (skill_dir / f"res{index:02d}.txt").write_text("r", encoding="utf-8")
            snapshot = SkillLoader().build_snapshot(
                project_root=proj, user_root=base / "nouser"
            )
            big = snapshot.get("big")
            self.assertIsNotNone(big, "skill must pass discovery bounds")

            builtin_registry = builtin.create_builtin_tool_registry()
            registry = ToolRegistry([create_load_skill_tool(snapshot)])
            for tool in builtin_registry.list_tools():
                registry.register(tool)
            runtime = ToolRuntimeServices()
            executor = ToolExecutor(registry, workspace_root=workspace, runtime=runtime)
            state = RunState.start("use big")
            result = executor.execute(
                ToolCall(id="call-big-skill-1", name="load_skill", arguments={"name": "big"}),
                state,
            )
            self.assertTrue(result.ok)
            self.assertLessEqual(len(result.content or ""), MAX_LOADED_SKILL_CHARS)
            # RunState wraps the transient result (state_only -> not recorded).
            state.add_tool_result(result)
            # ContextProvider must keep the full body for load_skill results.
            provider = SessionContextProvider()
            budget = ContextBudget()
            compacted = provider._compact_current_tool_result(state.messages[-1], budget)
            content = compacted.content or ""
            self.assertNotIn("<truncated>", content)
            self.assertIn(body, content)
            self.assertIn("d" * 1500, content)
            self.assertIn(".agents/skills/big/res19.txt", content)
            # The final message content carries no truncation marker at all.
            self.assertNotIn("<truncated>", content)
            self.assertIn(body, content)

    def test_oversized_skill_rejected_at_discovery_not_truncated(self):
        # A skill whose formatted output would exceed the budget is rejected at
        # discovery with a clear issue instead of being silently truncated in
        # the context layer (F1).
        import forestcode.skills.loader as loader_mod

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj = base / ".agents" / "skills"
            skill_dir = proj / "huge"
            skill_dir.mkdir(parents=True)
            body = "x" * loader_mod.MAX_BODY_CHARS
            long_desc = "d" * loader_mod.MAX_DESCRIPTION_CHARS
            (skill_dir / "SKILL.md").write_text(
                f"---\nname: huge\ndescription: {long_desc}\n---\n{body}",
                encoding="utf-8",
            )
            # 100 long resource names push the formatted size past the budget.
            for index in range(100):
                (skill_dir / f"resource-{index:03d}-very-long-name.txt").write_text("r", encoding="utf-8")
            snapshot = SkillLoader().build_snapshot(
                project_root=proj, user_root=base / "nouser"
            )
            self.assertIsNone(snapshot.get("huge"))
            codes = [issue.code for issue in snapshot.issues]
            self.assertTrue(
                any(code in ("loaded_too_large", "catalog_overflow", "catalog_desc_budget") for code in codes)
            )


if __name__ == "__main__":
    unittest.main()
