"""Tests for skill discovery and validation (PRD R1/R2, design §Discovery)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from forestcode.skills.loader import (
    MAX_BODY_CHARS,
    MAX_CATALOG_SKILLS,
    MAX_DESCRIPTION_CHARS,
    MAX_RESOURCES,
    SkillLoader,
)


def _write_skill(root: Path, rel_dir: str, name: str, description: str, body: str = "instructions") -> Path:
    target = root / rel_dir
    target.mkdir(parents=True, exist_ok=True)
    entry = target / "SKILL.md"
    entry.write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n{body}",
        encoding="utf-8",
    )
    return entry


def SkillSnapshot_for(root: Path):
    return SkillLoader().build_snapshot(project_root=root, user_root=root.parent / "nouser")


class SkillDiscoveryTest(unittest.TestCase):
    def test_finds_project_and_user_skills(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj = base / "proj" / ".agents" / "skills"
            user = base / "user" / ".agents" / "skills"
            _write_skill(proj, "alpha", "alpha", "project skill")
            _write_skill(user, "beta", "beta", "user skill")
            snapshot = SkillLoader().build_snapshot(project_root=proj, user_root=user)
            names = [d.name for d in snapshot.descriptors]
            self.assertEqual(names, ["alpha", "beta"])
            alpha = snapshot.get("alpha")
            beta = snapshot.get("beta")
            assert alpha is not None
            assert beta is not None
            self.assertEqual(alpha.source, "project")
            self.assertEqual(beta.source, "user")

    def test_recursive_grouping_under_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj = base / ".agents" / "skills"
            _write_skill(proj, "group/sub/one", "one", "nested")
            _write_skill(proj, "two", "two", "top")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual([d.name for d in snapshot.descriptors], ["one", "two"])

    def test_stable_order_by_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj = base / ".agents" / "skills"
            for name in ["zebra", "alpha", "mike"]:
                _write_skill(proj, name, name, f"desc {name}")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual([d.name for d in snapshot.descriptors], ["alpha", "mike", "zebra"])

    def test_project_overrides_user_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj = base / "proj" / ".agents" / "skills"
            user = base / "user" / ".agents" / "skills"
            _write_skill(proj, "dup", "dup", "project version")
            _write_skill(user, "dup", "dup", "user version")
            snapshot = SkillLoader().build_snapshot(project_root=proj, user_root=user)
            self.assertEqual(len(snapshot.descriptors), 1)
            dup = snapshot.get("dup")
            assert dup is not None
            self.assertEqual(dup.source, "project")
            self.assertEqual(dup.description, "project version")

    def test_same_source_duplicate_is_ambiguous_and_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj = base / ".agents" / "skills"
            _write_skill(proj, "a/one", "dup", "first")
            _write_skill(proj, "b/two", "dup", "second")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual(snapshot.descriptors, ())
            codes = [issue.code for issue in snapshot.issues]
            self.assertIn("ambiguous", codes)

    def test_bad_file_does_not_block_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj = base / ".agents" / "skills"
            (proj / "bad").mkdir(parents=True)
            # dot is not allowed in skill names
            (proj / "bad" / "SKILL.md").write_text("---\nname: bad.name\n---\nbody", encoding="utf-8")
            _write_skill(proj, "good", "good", "fine")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual([d.name for d in snapshot.descriptors], ["good"])
            self.assertTrue(any(issue.code == "invalid_name" for issue in snapshot.issues))


class SkillValidationTest(unittest.TestCase):
    def test_uppercase_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            _write_skill(proj, "x", "BadName", "desc")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual(snapshot.descriptors, ())
            self.assertTrue(any(issue.code == "invalid_name" for issue in snapshot.issues))

    def test_missing_description_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            (proj / "x").mkdir(parents=True)
            (proj / "x" / "SKILL.md").write_text("---\nname: ok\n---\nbody", encoding="utf-8")
            snapshot = SkillSnapshot_for(proj)
            self.assertTrue(any(issue.code == "missing_description" for issue in snapshot.issues))

    def test_non_string_description_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            (proj / "x").mkdir(parents=True)
            (proj / "x" / "SKILL.md").write_text("---\nname: ok\ndescription: [1, 2]\n---\nbody", encoding="utf-8")
            snapshot = SkillSnapshot_for(proj)
            self.assertTrue(any(issue.code == "missing_description" for issue in snapshot.issues))

    def test_body_over_limit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            _write_skill(proj, "x", "big", "desc", body="x" * (MAX_BODY_CHARS + 1))
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual(snapshot.descriptors, ())
            self.assertTrue(any(issue.code == "body_too_large" for issue in snapshot.issues))

    def test_malformed_frontmatter_rejected_with_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            (proj / "x").mkdir(parents=True)
            (proj / "x" / "SKILL.md").write_text("---\nname: [bad\n---\nbody", encoding="utf-8")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual(snapshot.descriptors, ())
            self.assertTrue(any(issue.code == "frontmatter" for issue in snapshot.issues))

    def test_catalog_skill_count_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            for index in range(MAX_CATALOG_SKILLS + 5):
                name = f"skill{index:03d}"
                _write_skill(proj, f"d{index:03d}/{name}", name, "d")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual(len(snapshot.descriptors), MAX_CATALOG_SKILLS)
            self.assertTrue(any(issue.code == "catalog_overflow" for issue in snapshot.issues))

    def test_catalog_description_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            # Each description stays under the single-description cap but their
            # sum exceeds the catalog budget (F1 budget composition):
            # 6 x (MAX_DESCRIPTION_CHARS - 5) > MAX_CATALOG_DESC_CHARS.
            for index in range(6):
                name = f"s{index}"
                _write_skill(proj, name, name, "x" * (MAX_DESCRIPTION_CHARS - 5))
            snapshot = SkillSnapshot_for(proj)
            names = [d.name for d in snapshot.descriptors]
            self.assertNotIn("s5", names)
            self.assertTrue(any(issue.code == "catalog_desc_budget" for issue in snapshot.issues))

    def test_single_description_over_limit_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            _write_skill(proj, "x", "big", "x" * (MAX_DESCRIPTION_CHARS + 1))
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual(snapshot.descriptors, ())
            self.assertTrue(any(issue.code == "description_too_large" for issue in snapshot.issues))

    def test_non_utf8_skill_rejected_with_invalid_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            target = proj / "x"
            target.mkdir(parents=True)
            # Invalid UTF-8 byte sequence in the body.
            (target / "SKILL.md").write_bytes(b"---\nname: x\ndescription: d\n---\nbody \xff\xfe")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual(snapshot.descriptors, ())
            self.assertTrue(any(issue.code == "invalid_encoding" for issue in snapshot.issues))

    def test_body_over_limit_rejected_before_runtime_budget_check(self):
        # The body limit is checked before the formatted-result budget.
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            _write_skill(proj, "x", "big", "d" * 1000, body="x" * (MAX_BODY_CHARS + 1))
            snapshot = SkillSnapshot_for(proj)
            self.assertTrue(any(issue.code == "body_too_large" for issue in snapshot.issues))

    def test_formatted_skill_over_runtime_budget_rejected_at_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            entry = _write_skill(
                proj,
                "big",
                "big",
                "d" * MAX_DESCRIPTION_CHARS,
                body="x" * MAX_BODY_CHARS,
            )
            for index in range(MAX_RESOURCES):
                name = f"{index:02d}-" + ("r" * 180) + ".txt"
                (entry.parent / name).write_text("r", encoding="utf-8")

            snapshot = SkillSnapshot_for(proj)

            self.assertIsNone(snapshot.get("big"))
            self.assertTrue(
                any(issue.code == "loaded_too_large" for issue in snapshot.issues)
            )

    def test_load_rejects_resources_that_grow_past_budget_after_discovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            entry = _write_skill(
                proj,
                "big",
                "big",
                "d" * MAX_DESCRIPTION_CHARS,
                body="x" * MAX_BODY_CHARS,
            )
            snapshot = SkillSnapshot_for(proj)
            self.assertIsNotNone(snapshot.get("big"))

            # SKILL.md remains byte-identical, but newly added resource paths
            # would push format_loaded_skill beyond the runtime payload bound.
            for index in range(MAX_RESOURCES):
                name = f"{index:02d}-" + ("r" * 180) + ".txt"
                (entry.parent / name).write_text("r", encoding="utf-8")

            self.assertIsNone(snapshot.load("big"))

    def test_content_digest_fails_load_after_edit(self):
        # F4: a snapshot discovered before the file changed must not load new
        # content under the stale name/description.
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            entry = _write_skill(proj, "s", "s", "original")
            snapshot = SkillSnapshot_for(proj)
            loaded = snapshot.load("s")
            assert loaded is not None
            self.assertEqual(loaded.instructions, "instructions")
            # Rewrite the file after discovery; the fixed snapshot must refuse.
            entry.write_text("---\nname: other\ndescription: swapped\n---\nmalicious", encoding="utf-8")
            self.assertIsNone(snapshot.load("s"))


class SkillContainmentTest(unittest.TestCase):
    def test_symlinked_directory_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "outside"
            _write_skill(outside, "esc", "esc", "outside skill")
            proj = base / ".agents" / "skills"
            proj.mkdir(parents=True)
            try:
                os.symlink(outside, proj / "link", target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks not supported on this platform")
            _write_skill(proj, "real", "real", "inside")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual([d.name for d in snapshot.descriptors], ["real"])

    def test_symlinked_entry_escaping_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            outside = base / "outside"
            _write_skill(outside, "esc", "esc", "outside skill")
            proj = base / ".agents" / "skills"
            (proj / "x").mkdir(parents=True)
            try:
                os.symlink(outside / "esc" / "SKILL.md", proj / "x" / "SKILL.md")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks not supported on this platform")
            snapshot = SkillSnapshot_for(proj)
            self.assertEqual(snapshot.descriptors, ())
            self.assertTrue(any(issue.code == "escape" for issue in snapshot.issues))

    def test_issue_path_is_relative_not_absolute(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            (proj / "x").mkdir(parents=True)
            (proj / "x" / "SKILL.md").write_text("not frontmatter", encoding="utf-8")
            snapshot = SkillSnapshot_for(proj)
            issue = next(i for i in snapshot.issues)
            self.assertNotIn(str(Path(tmp)), issue.message)
            self.assertNotIn(str(Path.home()), issue.message)
            self.assertEqual(issue.path, "x/SKILL.md")


class SkillResourceListingTest(unittest.TestCase):
    def test_resources_exclude_entry_symlinks_include_hidden_files(self):
        # Design excludes SKILL.md, hidden directories, symlinks, and out-of-root
        # paths — hidden *files* are still valid resources (listing is relative).
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj = base / ".agents" / "skills"
            skill_dir = proj / "tool" / "skill"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\nname: tool\ndescription: d\n---\nbody", encoding="utf-8"
            )
            (skill_dir / "helper.py").write_text("print(1)", encoding="utf-8")
            (skill_dir / ".hidden.txt").write_text("x", encoding="utf-8")
            hidden_dir = skill_dir / ".secretdir"
            hidden_dir.mkdir()
            (hidden_dir / "inside.txt").write_text("x", encoding="utf-8")
            nested = skill_dir / "nested"
            nested.mkdir()
            (nested / "deep.md").write_text("d", encoding="utf-8")
            try:
                os.symlink(skill_dir / "helper.py", skill_dir / "linked.py")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks not supported on this platform")

            snapshot = SkillSnapshot_for(proj)
            loaded = snapshot.load("tool")
            assert loaded is not None
            # F2: resource paths are workspace-resolvable (project level), not
            # bare relative-to-skill-dir names. The skill dir is tool/skill, so
            # the display path preserves the grouping.
            self.assertEqual(
                loaded.resource_paths,
                (
                    ".agents/skills/tool/skill/.hidden.txt",
                    ".agents/skills/tool/skill/helper.py",
                    ".agents/skills/tool/skill/nested/deep.md",
                ),
            )

    def test_resources_capped_at_20(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            skill_dir = proj / "s"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nbody", encoding="utf-8")
            for index in range(30):
                (skill_dir / f"f{index:02d}.txt").write_text("x", encoding="utf-8")
            snapshot = SkillSnapshot_for(proj)
            loaded = snapshot.load("s")
            assert loaded is not None
            self.assertEqual(len(loaded.resource_paths), MAX_RESOURCES)

    def test_load_returns_full_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = Path(tmp) / ".agents" / "skills"
            _write_skill(proj, "s", "s", "desc", body="# My skill\n\nDo things.")
            snapshot = SkillSnapshot_for(proj)
            loaded = snapshot.load("s")
            assert loaded is not None
            self.assertEqual(loaded.instructions, "# My skill\n\nDo things.")
            self.assertEqual(loaded.descriptor.name, "s")


if __name__ == "__main__":
    unittest.main()
