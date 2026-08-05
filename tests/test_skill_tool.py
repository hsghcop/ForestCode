"""Tests for the load_skill tool through the real ToolExecutor pipeline (PRD R3, AC3)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forestcode.core import ToolCall, ToolExecutor
from forestcode.core.run_state import RunState
from forestcode.skills import SkillRegistry
from forestcode.skills.loader import MAX_RESOURCES, SkillLoader
from forestcode.skills.types import SkillSnapshot
from forestcode.tools import ToolRegistry, ToolRuntimeServices
from forestcode.tools.skills import create_load_skill_tool


def _snapshot_with_skill(
    name: str = "demo", body: str = "instructions", extra_files: int = 0
):
    """Yield a live temp workspace + snapshot; the dir stays alive until close()."""
    import tempfile as _tempfile

    tmp = _tempfile.TemporaryDirectory()
    base = Path(tmp.name)
    project = base / "workspace" / ".agents" / "skills"
    skill_dir = project / "s"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: a demo skill\n---\n{body}", encoding="utf-8"
    )
    for index in range(extra_files):
        (skill_dir / f"res{index}.txt").write_text("x", encoding="utf-8")
    snapshot = SkillLoader().build_snapshot(project_root=project, user_root=base / "nouser")
    return snapshot, tmp


class _SkillEnv:
    """Keeps the skill tempdir alive for the duration of one test."""

    def __init__(self, **kwargs):
        self.snapshot, self._tmp = _snapshot_with_skill(**kwargs)

    def close(self):
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class LoadSkillToolTest(unittest.TestCase):
    def test_schema_and_validator(self):
        with _SkillEnv() as env:
            tool = create_load_skill_tool(env.snapshot)
            self.assertEqual(tool.name, "load_skill")
            self.assertTrue(tool.is_read_only)
            self.assertEqual(tool.risk_level, "read_only")
            self.assertFalse(tool.persist_result)
            self.assertEqual(tool.validate_arguments({"name": "demo"}), {"name": "demo"})
            with self.assertRaises(ValueError):
                tool.validate_arguments({"name": "  "})

    def test_success_result_contains_body_and_resources(self):
        with _SkillEnv(body="# Body line", extra_files=2) as env:
            registry = ToolRegistry([create_load_skill_tool(env.snapshot)])
            with tempfile.TemporaryDirectory() as tmp:
                executor = ToolExecutor(registry, workspace_root=tmp)
                result = executor.execute(
                    ToolCall(id="c1", name="load_skill", arguments={"name": "demo"}),
                    RunState.start("t"),
                )
            self.assertTrue(result.ok)
            self.assertIn("# Body line", result.content)
            self.assertIn("res0.txt", result.content)
            self.assertIn("res1.txt", result.content)

    def test_transient_result_markers_on_success(self):
        with _SkillEnv(body="SECRET-BODY") as env:
            registry = ToolRegistry([create_load_skill_tool(env.snapshot)])
            with tempfile.TemporaryDirectory() as tmp:
                executor = ToolExecutor(registry, workspace_root=tmp)
                result = executor.execute(
                    ToolCall(id="c1", name="load_skill", arguments={"name": "demo"}),
                    RunState.start("t"),
                )
            self.assertTrue(result.ok)
            self.assertEqual(result.data.get("state_only"), True)
            self.assertEqual(result.data.get("transient"), True)

    def test_unknown_skill_returns_failed_result(self):
        with _SkillEnv() as env:
            registry = ToolRegistry([create_load_skill_tool(env.snapshot)])
            with tempfile.TemporaryDirectory() as tmp:
                executor = ToolExecutor(registry, workspace_root=tmp)
                result = executor.execute(
                    ToolCall(id="c1", name="load_skill", arguments={"name": "missing"}),
                    RunState.start("t"),
                )
            self.assertFalse(result.ok)
            self.assertIn("Unknown skill", result.error or "")
            # failures are transient too
            self.assertEqual(result.data.get("state_only"), True)

    def test_runs_without_path_declaration_and_does_not_touch_sandbox(self):
        with _SkillEnv() as env:
            tool = create_load_skill_tool(env.snapshot)
            self.assertEqual(tool.get_paths({"name": "demo"}), [])

    def test_resource_list_bounded(self):
        with _SkillEnv(extra_files=30) as env:
            registry = ToolRegistry([create_load_skill_tool(env.snapshot)])
            with tempfile.TemporaryDirectory() as tmp:
                executor = ToolExecutor(registry, workspace_root=tmp)
                result = executor.execute(
                    ToolCall(id="c1", name="load_skill", arguments={"name": "demo"}),
                    RunState.start("t"),
                )
            self.assertTrue(result.ok)
            self.assertLessEqual(result.content.count("\n- "), MAX_RESOURCES)

    def test_project_resource_path_readable_by_read_file(self):
        # F2 end-to-end: load_skill lists workspace-resolvable resource paths
        # and the existing read_file tool can read them directly.
        import forestcode.tools.builtin as builtin

        with _SkillEnv(body="# Body", extra_files=1) as env:
            workspace = env.snapshot.descriptors[0].root.parents[2]  # base/workspace
            builtin_registry = builtin.create_builtin_tool_registry()
            registry = ToolRegistry([create_load_skill_tool(env.snapshot)])
            for tool in builtin_registry.list_tools():
                registry.register(tool)
            runtime = ToolRuntimeServices()
            executor = ToolExecutor(registry, workspace_root=workspace, runtime=runtime)
            state = RunState.start("use demo")
            loaded = executor.execute(
                ToolCall(id="c1", name="load_skill", arguments={"name": "demo"}),
                state,
            )
            self.assertTrue(loaded.ok)
            self.assertIn(".agents/skills/s/res0.txt", loaded.content)
            # read_file resolves the workspace-relative path without approval.
            read = executor.execute(
                ToolCall(id="c2", name="read_file", arguments={"path": ".agents/skills/s/res0.txt"}),
                state,
            )
            self.assertTrue(read.ok)
            self.assertIn("x", read.content)

    def test_user_resource_path_uses_tilde_and_needs_approval(self):
        # F2: user-level resources are surfaced as ~ paths; the sandbox
        # expands them and routes them through the outside-workspace approval.
        import forestcode.tools.builtin as builtin
        import tempfile as _tf

        with _tf.TemporaryDirectory() as tmp:
            base = Path(tmp)
            user_root = base / "user" / ".agents" / "skills"
            skill_dir = user_root / "demo"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: demo\ndescription: user skill\n---\nuser instructions",
                encoding="utf-8",
            )
            (skill_dir / "secret.txt").write_text("user data", encoding="utf-8")
            snapshot = SkillLoader().build_snapshot(
                project_root=base / "ws" / ".agents" / "skills",
                user_root=user_root,
            )
            workspace = base / "ws"
            builtin_registry = builtin.create_builtin_tool_registry()
            registry = ToolRegistry([create_load_skill_tool(snapshot)])
            for tool in builtin_registry.list_tools():
                registry.register(tool)
            runtime = ToolRuntimeServices()
            executor = ToolExecutor(registry, workspace_root=workspace, runtime=runtime)
            state = RunState.start("use demo")
            loaded = executor.execute(
                ToolCall(id="c1", name="load_skill", arguments={"name": "demo"}),
                state,
            )
            self.assertTrue(loaded.ok)
            self.assertIn("~/.agents/skills/demo/secret.txt", loaded.content)
            # Reading a ~ resource outside the workspace requires approval.
            from forestcode.tools.permissions import PermissionManager

            executor._permission_manager = PermissionManager()
            read = executor.execute(
                ToolCall(id="c2", name="read_file", arguments={"path": "~/.agents/skills/demo/secret.txt"}),
                state,
            )
            self.assertFalse(read.ok)
            self.assertEqual(read.data.get("permission"), "ask")


if __name__ == "__main__":
    unittest.main()
