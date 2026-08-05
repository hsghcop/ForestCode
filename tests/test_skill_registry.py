"""Tests for SkillRegistry snapshot lifecycle (design §Skill domain)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from forestcode.skills import SkillRegistry
from forestcode.skills.loader import SkillLoader


def _write_skill(root: Path, rel_dir: str, name: str, description: str, body: str = "instructions") -> None:
    target = root / rel_dir
    target.mkdir(parents=True, exist_ok=True)
    (target / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )


def _registry(workspace: Path, user_root: Path) -> SkillRegistry:
    return SkillRegistry(workspace_root=workspace, user_root=user_root)


class SkillRegistryTest(unittest.TestCase):
    def test_refresh_returns_snapshot_and_sets_current(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace = base / "workspace"
            user = base / "user"
            project_skills = workspace / ".agents" / "skills"
            _write_skill(project_skills, "alpha", "alpha", "project skill")
            _write_skill(user / ".agents" / "skills", "beta", "beta", "user skill")

            registry = _registry(workspace, user / ".agents" / "skills")
            snapshot = registry.refresh()
            self.assertEqual([d.name for d in snapshot.descriptors], ["alpha", "beta"])
            self.assertIs(registry.snapshot(), snapshot)
            self.assertEqual(registry.list(), snapshot.descriptors)
            alpha = registry.get("alpha")
            assert alpha is not None
            self.assertEqual(alpha.source, "project")

    def test_no_skills_yields_empty_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            registry = _registry(base / "workspace", base / "user")
            snapshot = registry.refresh()
            self.assertEqual(snapshot.descriptors, ())
            self.assertIsNone(registry.get("missing"))
            self.assertIsNone(registry.load("missing"))

    def test_load_reads_validated_body_and_resources(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace = base / "workspace"
            project_skills = workspace / ".agents" / "skills"
            skill_dir = project_skills / "tool"
            _write_skill(project_skills, "tool", "tool", "a tool skill", body="# Do things")
            (skill_dir / "helper.py").write_text("print(1)", encoding="utf-8")

            registry = _registry(workspace, base / "user")
            registry.refresh()
            loaded = registry.load("tool")
            assert loaded is not None
            self.assertEqual(loaded.instructions, "# Do things")
            # F2: project-level resources are workspace-relative paths.
            self.assertEqual(loaded.resource_paths, (".agents/skills/tool/helper.py",))

    def test_snapshot_is_fixed_independent_of_later_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace = base / "workspace"
            project_skills = workspace / ".agents" / "skills"
            _write_skill(project_skills, "alpha", "alpha", "v1")

            registry = _registry(workspace, base / "user")
            first = registry.refresh()
            _write_skill(project_skills, "beta", "beta", "v2")
            registry.refresh()

            # The old snapshot stays immutable: no beta, and loading alpha still works.
            self.assertEqual([d.name for d in first.descriptors], ["alpha"])
            self.assertIsNone(first.get("beta"))
            assert first.get("alpha") is not None
            loaded = first.load("alpha")
            assert loaded is not None
            self.assertEqual(loaded.descriptor.description, "v1")

    def test_build_snapshot_independent_of_registry(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project_skills = base / "workspace" / ".agents" / "skills"
            _write_skill(project_skills, "alpha", "alpha", "direct")
            snapshot = SkillLoader().build_snapshot(
                project_root=project_skills, user_root=base / "nouser"
            )
            alpha = snapshot.get("alpha")
            assert alpha is not None
            self.assertEqual(alpha.description, "direct")


if __name__ == "__main__":
    unittest.main()
