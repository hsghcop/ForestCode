from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from forestcode.config import AgentRuntimeConfig
from forestcode.core.fake_model import FakeModelClient
from forestcode.core.types import ModelOutput
from forestcode.memory import MemoryEntry, SessionStore
from forestcode.plan import PlanStore
from forestcode.skills import PendingSkillSelection, SkillRegistry
from forestcode.slash_commands import SlashContext
from forestcode.slash_handlers import build_builtin_slash_registry
from forestcode.tools import ToolRuntimeServices


def _context(root: Path, *, session_id: str | None = "default", inputs=None):
    stdout = io.StringIO()
    stderr = io.StringIO()
    store = SessionStore(root)
    return SlashContext(
        workspace_root=root,
        session_id=session_id,
        session_store=store,
        plan_store=PlanStore(),
        runtime=ToolRuntimeServices(),
        agent=AgentRuntimeConfig(),
        model=FakeModelClient([ModelOutput(text="summary")]),
        stdout=stdout,
        stderr=stderr,
        input_func=lambda prompt: next(inputs) if inputs is not None else "",
        registry=build_builtin_slash_registry(),
    )


def _cmd(ctx: SlashContext, name: str) -> Any:
    """Resolve a registered command; tests know it exists."""
    command = ctx.registry.get(name)
    if command is None:
        raise AssertionError(f"slash command not registered: {name}")
    return command


def _out(ctx: SlashContext) -> str:
    """The captured stdout text (tests always pass StringIO)."""
    return ctx.stdout.getvalue()  # type: ignore[attr-defined]


def _err(ctx: SlashContext) -> str:
    return ctx.stderr.getvalue()  # type: ignore[attr-defined]


def _make_skill(workspace: Path, name: str) -> None:
    skill_dir = workspace / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: desc\n---\nBODY", encoding="utf-8"
    )


class SlashHandlersTest(unittest.TestCase):
    def test_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _context(Path(tmp))
            result = _cmd(ctx, "exit").handler(ctx, "")

            self.assertEqual(result.action, "exit")

    def test_sessions_lists_meta_and_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root)

            _cmd(ctx, "sessions").handler(ctx, "")
            self.assertIn("no sessions yet", _out(ctx))

            ctx.session_store.append_entry(
                "s1", MemoryEntry(kind="message", role="user", content="hello")
            )
            ctx.session_store.update_meta("s1", title="Work")
            ctx.stdout = io.StringIO()
            _cmd(ctx, "sessions").handler(ctx, "")

            output = _out(ctx)
            self.assertIn("s1", output)
            self.assertIn("Work", output)
            self.assertIn("entries", output)

    def test_sessions_lists_legacy_json_without_migrating(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root, session_id=None)
            legacy = ctx.session_store._legacy_json_path("legacy")
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text(
                json.dumps({"session_id": "legacy", "entries": []}), encoding="utf-8"
            )

            _cmd(ctx, "sessions").handler(ctx, "")

            output = _out(ctx)
            self.assertIn("legacy", output)
            self.assertIn("(legacy)", output)
            self.assertTrue(legacy.exists())
            self.assertFalse(Path(str(legacy) + ".bak").exists())
            self.assertFalse(ctx.session_store._jsonl_path("legacy").exists())

    def test_switch_valid_invalid_and_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _context(Path(tmp), session_id="current")

            current = _cmd(ctx, "switch").handler(ctx, "current")
            invalid = _cmd(ctx, "switch").handler(ctx, "../etc")
            valid = _cmd(ctx, "switch").handler(ctx, "new-id")

            self.assertEqual(current.action, "continue")
            self.assertIn("already on current", _out(ctx))
            self.assertEqual(invalid.action, "continue")
            self.assertIn("Invalid session id", _err(ctx))
            self.assertEqual(valid.action, "switch_session")
            self.assertEqual(valid.new_session_id, "new-id")

    def test_delete_confirms_and_refuses_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root, session_id="current", inputs=iter(["n", "y"]))
            ctx.session_store.append_entry(
                "other", MemoryEntry(kind="message", role="user", content="hello")
            )
            path = ctx.session_store._jsonl_path("other")

            _cmd(ctx, "delete").handler(ctx, "current")
            self.assertIn("Cannot delete current session", _err(ctx))

            _cmd(ctx, "delete").handler(ctx, "other")
            self.assertTrue(path.exists())

            _cmd(ctx, "delete").handler(ctx, "other")
            self.assertFalse(path.exists())

    def test_delete_removes_legacy_json_and_backups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root, session_id=None, inputs=iter(["y"]))
            legacy = ctx.session_store._legacy_json_path("legacy")
            legacy.parent.mkdir(parents=True, exist_ok=True)
            paths = [
                legacy,
                Path(str(legacy) + ".bak"),
                Path(str(legacy) + ".bak.1"),
            ]
            for path in paths:
                path.write_text("legacy", encoding="utf-8")

            _cmd(ctx, "delete").handler(ctx, "legacy")

            for path in paths:
                self.assertFalse(path.exists(), str(path))

    def test_name_requires_session_and_updates_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            no_session = _context(root, session_id=None)
            _cmd(no_session, "name").handler(no_session, "Title")
            self.assertIn("requires --session or /switch first", _err(no_session))

            ctx = _context(root, session_id="default")
            _cmd(ctx, "name").handler(ctx, "My title")
            self.assertEqual(ctx.session_store.load("default").title, "My title")

    def test_compact_calls_compressor(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _context(Path(tmp))
            with patch(
                "forestcode.slash_handlers.SessionCompressor"
            ) as compressor_class:
                compressor_class.return_value.maybe_major_compact.return_value = True

                _cmd(ctx, "compact").handler(ctx, "")

            compressor_class.return_value.maybe_major_compact.assert_called_once_with()
            self.assertIn("Compact> compacted", _out(ctx))

    def test_compact_requires_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _context(Path(tmp), session_id=None)

            _cmd(ctx, "compact").handler(ctx, "")

            self.assertIn("requires --session or /switch first", _err(ctx))

    def test_skills_refreshes_before_selecting(self):
        """Legacy paths (cli.run_chat) never refresh before slash handling: the
        first /skills input must still discover existing skills instead of
        reporting "no skills found" (PRD R5 regression)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill(root, "demo")
            ctx = _context(root)
            ctx.skill_registry = SkillRegistry(root)  # never refreshed
            ctx.skill_pending = PendingSkillSelection()
            ctx.skill_selector = lambda _snapshot: "demo"

            _cmd(ctx, "skills").handler(ctx, "")

            self.assertNotIn("no skills found", _out(ctx))
            self.assertEqual(ctx.skill_pending.name, "demo")

    def test_skills_cancel_keeps_pending_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_skill(root, "demo")
            ctx = _context(root)
            ctx.skill_registry = SkillRegistry(root)
            ctx.skill_pending = PendingSkillSelection()
            ctx.skill_pending.replace("alpha")
            ctx.skill_selector = lambda _snapshot: None

            _cmd(ctx, "skills").handler(ctx, "")

            self.assertEqual(ctx.skill_pending.name, "alpha")

    def test_memory_prints_content_or_empty_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root)

            _cmd(ctx, "memory").handler(ctx, "")
            self.assertIn("no memory yet", _out(ctx))

            (root / "MEMORY.md").write_text("# Memory\n", encoding="utf-8")
            ctx.stdout = io.StringIO()
            _cmd(ctx, "memory").handler(ctx, "")
            self.assertIn("# Memory", _out(ctx))

    # -- /subagents --------------------------------------------------------
    def _agent_context(self, root: Path, names: tuple[str, ...] = ("helper",)):
        from forestcode.subagents.config_loader import AgentRegistry

        ctx = _context(root)
        agent_registry = AgentRegistry(root, user_root=root / "user-agents")
        for name in names:
            agent_dir = root / ".agents" / "subagents" / f"{name}.md"
            agent_dir.parent.mkdir(parents=True, exist_ok=True)
            agent_dir.write_text(
                f"---\nname: {name}\ndescription: desc\n---\nBODY", encoding="utf-8"
            )
        agent_registry.refresh()
        ctx.agent_registry = agent_registry
        from forestcode.subagents import PendingSubagentSelection

        ctx.subagent_pending = PendingSubagentSelection()
        ctx.subagent_selector = lambda snapshot: next(iter(snapshot.agents))
        return ctx

    def test_only_plural_subagents_command_is_registered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._agent_context(root)
            self.assertIsNone(ctx.registry.get("subagent"))
            self.assertIsNotNone(ctx.registry.get("subagents"))

    def test_subagents_sets_pending_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._agent_context(root)
            _cmd(ctx, "subagents").handler(ctx, "")
            self.assertEqual(ctx.subagent_pending.name, "helper")
            self.assertEqual(ctx.subagent_pending.marker_text(), "[Subagent: helper]")

    def test_subagents_cancel_keeps_existing_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._agent_context(root)
            ctx.subagent_pending.replace("existing")
            ctx.subagent_selector = lambda _snapshot: None
            _cmd(ctx, "subagents").handler(ctx, "")
            self.assertEqual(ctx.subagent_pending.name, "existing")

    def test_subagents_rejects_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._agent_context(root)
            _cmd(ctx, "subagents").handler(ctx, "helper")
            self.assertIn("Usage: /subagents", _err(ctx))
            self.assertIsNone(ctx.subagent_pending.name)

    def test_subagents_reports_empty_catalog(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._agent_context(root, names=())
            _cmd(ctx, "subagents").handler(ctx, "")
            self.assertIn("no subagents found", _out(ctx))


if __name__ == "__main__":
    unittest.main()
