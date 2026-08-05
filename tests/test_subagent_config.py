"""Tests for subagent configuration and domain contracts (design §Configuration Contract).

Offline only: no network, no model calls, no sleeps.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from forestcode.models.types import ModelAdapterError, ModelConfig
from forestcode.subagents import (
    MAX_COMBINED_CONTEXT_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_PROMPT_CHARS,
    PERMISSION_PROFILES,
    SUBAGENT_TOOLS,
    AgentConfig,
    AgentConfigLoader,
    AgentConfigSet,
    AgentRegistry,
    ModelOverride,
    ToolsSpec,
    effective_tool_names,
    generate_task_id,
    is_valid_task_id,
    resolve_child_model_config,
    transition_allowed,
)
from forestcode.subagents.config_loader import validate_agent_snapshot
from forestcode.subagents.types import (
    MAX_DEFAULT_SKILLS,
    MAX_INSTRUCTIONS_CHARS,
    MAX_MODEL_BASE_URL_CHARS,
    MAX_MODEL_FIELD_CHARS,
    MAX_TASK_TIMEOUT_SECONDS,
    MAX_TOOLS_ALLOW,
    MAX_TOOLS_DENY,
    MIN_TASK_TIMEOUT_SECONDS,
    SubagentStatus,
)

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

PARENT_MODEL = ModelConfig(
    api_type="openai",
    model="gpt-4o",
    base_url="https://api.openai.com/v1",
    api_key="parent-secret",
    timeout=60.0,
    reasoning_mode=None,
    reasoning_effort=None,
)


def _write_md(
    root: Path,
    rel: str,
    name: str,
    description: str,
    extra: str = "",
    body: str = "instructions",
) -> Path:
    """Write a Markdown agent config; ``extra`` is YAML lines before the closing ``---``."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}",
        encoding="utf-8",
    )
    return path


def _write_json(root: Path, rel: str, data: dict) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _build(project_root: Path, user_root: Path) -> AgentConfigSet:
    return AgentConfigLoader().build_set(project_root=project_root, user_root=user_root)


def _roots(tmp: Path) -> tuple[Path, Path]:
    """Return (project subagents dir, user subagents dir) under a tmp tree."""
    workspace = tmp / "workspace"
    user = tmp / "user_home"
    return workspace / ".agents" / "subagents", user / ".agents" / "subagents"


class AgentDiscoveryTest(unittest.TestCase):
    def test_finds_project_md_and_user_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "alpha.md", "alpha", "project md")
            _write_json(
                user,
                "beta.json",
                {"name": "beta", "description": "user json", "instructions": "body"},
            )
            snapshot = _build(proj, user)
            self.assertEqual(sorted(snapshot.agents), ["alpha", "beta"])
            alpha = snapshot.get("alpha")
            beta = snapshot.get("beta")
            assert alpha is not None
            assert beta is not None
            self.assertEqual(alpha.instructions, "instructions")
            self.assertEqual(beta.instructions, "body")

    def test_md_body_is_instructions_and_frontmatter_instructions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(
                proj, "a.md", "a", "d", extra="instructions: nope\n", body="# real body"
            )
            _write_md(proj, "b.md", "b", "d", body="# other body")
            snapshot = _build(proj, user)
            self.assertEqual([a.name for a in snapshot.agents.values()], ["b"])
            codes = [issue.code for issue in snapshot.issues]
            self.assertIn("instructions_in_frontmatter", codes)
            config = snapshot.get("b")
            assert config is not None
            self.assertEqual(config.instructions, "# other body")

    def test_json_requires_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_json(proj, "a.json", {"name": "a", "description": "d"})
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            self.assertTrue(
                any(issue.code == "missing_instructions" for issue in snapshot.issues)
            )

    def test_md_and_json_equivalent_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(
                proj,
                "reviewer.md",
                "reviewer",
                "code review helper",
                extra=(
                    "permission_profile: verify\n"
                    "tools:\n"
                    "  allow:\n"
                    "    - edit_file\n"
                    "  deny:\n"
                    "    - grep_text\n"
                    "default_skills:\n"
                    "  - review\n"
                    "  - review\n"
                    "model:\n"
                    "  api_type: deepseek\n"
                    "  model: deepseek-chat\n"
                    "  base_url: https://api.deepseek.com/v1\n"
                    "  timeout: 120\n"
                    "  api_key_env: DEEPSEEK_KEY\n"
                    "task_timeout_seconds: 900\n"
                ),
                body="Review carefully.",
            )
            _write_json(
                user,
                "reviewer.json",
                {
                    "name": "reviewer",
                    "description": "code review helper",
                    "instructions": "Review carefully.",
                    "permission_profile": "verify",
                    "tools": {"allow": ["edit_file"], "deny": ["grep_text"]},
                    "default_skills": ["review", "review"],
                    "model": {
                        "api_type": "deepseek",
                        "model": "deepseek-chat",
                        "base_url": "https://api.deepseek.com/v1",
                        "timeout": 120,
                        "api_key_env": "DEEPSEEK_KEY",
                    },
                    "task_timeout_seconds": 900,
                },
            )
            snapshot = _build(proj, user)
            self.assertEqual(
                len(snapshot.agents), 1
            )  # project wins, no ambiguity across roots
            config = snapshot.get("reviewer")
            assert config is not None
            self.assertEqual(config.permission_profile, "verify")
            self.assertEqual(
                config.tools, ToolsSpec(allow=("edit_file",), deny=("grep_text",))
            )
            self.assertEqual(config.default_skills, ("review",))
            assert config.model is not None
            self.assertEqual(config.model.api_type, "deepseek")
            self.assertEqual(config.model.model, "deepseek-chat")
            self.assertEqual(config.model.base_url, "https://api.deepseek.com/v1")
            self.assertEqual(config.model.timeout, 120.0)
            self.assertEqual(config.model.api_key_env, "DEEPSEEK_KEY")
            self.assertEqual(config.task_timeout_seconds, 900)
            self.assertEqual(config.instructions, "Review carefully.")

    def test_recursive_discovery_and_stable_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "group/sub/zeta.md", "zeta", "nested")
            _write_md(proj, "alpha.md", "alpha", "top")
            _write_json(
                proj,
                "group/mike.json",
                {"name": "mike", "description": "json nested", "instructions": "x"},
            )
            snapshot = _build(proj, user)
            self.assertEqual(
                [a.name for a in snapshot.agents.values()], ["alpha", "mike", "zeta"]
            )

    def test_project_overrides_user_same_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "dup.md", "dup", "project version")
            _write_md(user, "dup.md", "dup", "user version")
            snapshot = _build(proj, user)
            self.assertEqual(list(snapshot.agents), ["dup"])
            dup = snapshot.get("dup")
            assert dup is not None
            self.assertEqual(dup.description, "project version")
            self.assertFalse(
                any(issue.code == "ambiguous" for issue in snapshot.issues)
            )

    def test_same_source_md_json_duplicate_is_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "a/dup.md", "dup", "md version")
            _write_json(
                proj,
                "b/dup.json",
                {"name": "dup", "description": "json version", "instructions": "x"},
            )
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            self.assertTrue(any(issue.code == "ambiguous" for issue in snapshot.issues))

    def test_symlinked_file_not_followed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            outside = base / "outside.md"
            outside.write_text(
                "---\nname: evil\ndescription: d\n---\nbody", encoding="utf-8"
            )
            proj.mkdir(parents=True)
            os.symlink(outside, proj / "evil.md")
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})

    def test_file_too_large_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "big.md", "big", "d", body="x" * 64_001)
            _write_md(proj, "ok.md", "ok", "d")
            snapshot = _build(proj, user)
            self.assertEqual([a.name for a in snapshot.agents.values()], ["ok"])
            self.assertTrue(any(issue.code == "too_large" for issue in snapshot.issues))

    def test_invalid_utf8_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            path = proj / "bad.md"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"---\nname: bad\ndescription: d\n---\n\xff\xfe")
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            self.assertTrue(
                any(issue.code == "invalid_encoding" for issue in snapshot.issues)
            )

    def test_bad_file_does_not_block_others(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "bad.md", "bad.name", "d")  # dot not allowed in names
            _write_md(proj, "good.md", "good", "fine")
            snapshot = _build(proj, user)
            self.assertEqual([a.name for a in snapshot.agents.values()], ["good"])
            self.assertTrue(
                any(issue.code == "invalid_name" for issue in snapshot.issues)
            )

    def test_catalog_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            for i in range(101):
                _write_md(proj, f"agent{i:03d}.md", f"agent{i:03d}", "d")
            snapshot = _build(proj, user)
            self.assertEqual(len(snapshot.agents), 100)
            self.assertTrue(
                any(issue.code == "catalog_overflow" for issue in snapshot.issues)
            )

    def test_empty_roots_yield_empty_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            self.assertEqual(snapshot.issues, ())


class SchemaValidationTest(unittest.TestCase):
    def _issues_for(self, snapshot: AgentConfigSet) -> list[str]:
        return [issue.code for issue in snapshot.issues]

    def test_unknown_field_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "a.md", "a", "d", extra="bogus: 1\n")
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            self.assertIn("unknown_field", self._issues_for(snapshot))

    def test_invalid_name_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "upper.md", "BadName", "d")
            _write_md(proj, "dot.md", "bad.name", "d")
            _write_json(proj, "missing.json", {"description": "d", "instructions": "x"})
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            self.assertEqual(self._issues_for(snapshot).count("invalid_name"), 3)

    def test_description_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "empty.md", "empty", "   ")
            _write_md(proj, "long.md", "long", "d" * (MAX_DESCRIPTION_CHARS + 1))
            _write_md(proj, "ok.md", "ok", " d ")
            snapshot = _build(proj, user)
            self.assertEqual([a.name for a in snapshot.agents.values()], ["ok"])
            codes = self._issues_for(snapshot)
            self.assertIn("missing_description", codes)
            self.assertIn("description_too_large", codes)

    def test_instructions_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "empty.md", "empty", "d", body="   \n")
            _write_md(
                proj, "long.md", "long", "d", body="i" * (MAX_INSTRUCTIONS_CHARS + 1)
            )
            _write_md(proj, "ok.md", "ok", "d", body="i" * MAX_INSTRUCTIONS_CHARS)
            snapshot = _build(proj, user)
            self.assertEqual([a.name for a in snapshot.agents.values()], ["ok"])
            codes = self._issues_for(snapshot)
            self.assertIn("missing_instructions", codes)
            self.assertIn("instructions_too_large", codes)

    def test_tools_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "notmap.md", "notmap", "d", extra="tools: [edit_file]\n")
            _write_md(proj, "unknown.md", "unknown", "d", extra="tools:\n  foo: []\n")
            _write_md(
                proj,
                "badname.md",
                "badname",
                "d",
                extra="tools:\n  allow:\n    - Not_A_Tool\n",
            )
            _write_json(
                proj,
                "toomany.json",
                {
                    "name": "toomany",
                    "description": "d",
                    "instructions": "x",
                    "tools": {
                        "allow": [f"tool{i:02d}" for i in range(MAX_TOOLS_ALLOW + 1)]
                    },
                },
            )
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            codes = self._issues_for(snapshot)
            self.assertIn("invalid_tools", codes)  # notmap + unknown + badname
            self.assertIn("invalid_tool_name", codes)
            self.assertIn("tools_too_many", codes)

    def test_tools_within_limits_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            allow = [f"tool{i:02d}" for i in range(MAX_TOOLS_ALLOW)]
            deny = [f"tool{i:02d}" for i in range(MAX_TOOLS_DENY)]
            _write_json(
                proj,
                "max.json",
                {
                    "name": "max",
                    "description": "d",
                    "instructions": "x",
                    "tools": {"allow": allow, "deny": deny},
                },
            )
            snapshot = _build(proj, user)
            config = snapshot.get("max")
            assert config is not None
            self.assertEqual(len(config.tools.allow), MAX_TOOLS_ALLOW)
            self.assertEqual(len(config.tools.deny), MAX_TOOLS_DENY)

    def test_default_skills_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_json(
                proj,
                "toomany.json",
                {
                    "name": "toomany",
                    "description": "d",
                    "instructions": "x",
                    "default_skills": [f"s{i}" for i in range(MAX_DEFAULT_SKILLS + 1)],
                },
            )
            _write_json(
                proj,
                "bad.json",
                {
                    "name": "bad",
                    "description": "d",
                    "instructions": "x",
                    "default_skills": ["Not.A.Skill"],
                },
            )
            _write_json(
                proj,
                "notlist.json",
                {
                    "name": "notlist",
                    "description": "d",
                    "instructions": "x",
                    "default_skills": "review",
                },
            )
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            codes = self._issues_for(snapshot)
            self.assertIn("default_skills_too_many", codes)
            self.assertIn("invalid_skill_name", codes)
            self.assertIn("invalid_default_skills", codes)

    def test_model_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "notmap.md", "notmap", "d", extra="model: deepseek\n")
            _write_md(proj, "unknown.md", "unknown", "d", extra="model:\n  bogus: x\n")
            _write_md(
                proj,
                "long.md",
                "long",
                "d",
                extra=f"model:\n  api_type: {'x' * (MAX_MODEL_FIELD_CHARS + 1)}\n",
            )
            _write_md(
                proj,
                "longurl.md",
                "longurl",
                "d",
                extra=f"model:\n  base_url: {'x' * (MAX_MODEL_BASE_URL_CHARS + 1)}\n",
            )
            _write_md(proj, "badtm.md", "badtm", "d", extra="model:\n  timeout: 0\n")
            _write_md(
                proj, "booltm.md", "booltm", "d", extra="model:\n  timeout: true\n"
            )
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            self.assertEqual(self._issues_for(snapshot).count("invalid_model"), 6)

    def test_model_override_inherit_via_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(
                proj, "empty.md", "empty", "d", extra="model:\n  api_type: null\n"
            )
            snapshot = _build(proj, user)
            config = snapshot.get("empty")
            assert config is not None
            self.assertIsNone(config.model)

    def test_reasoning_null_is_explicit_disable_and_missing_is_inherit(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(
                proj,
                "disable.md",
                "disable",
                "d",
                extra="model:\n  reasoning_mode: null\n  reasoning_effort: null\n",
            )
            config = _build(proj, user).get("disable")
            assert config is not None and config.model is not None
            self.assertIsNone(config.model.reasoning_mode)
            self.assertIsNone(config.model.reasoning_effort)

    def test_invalid_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "a.md", "a", "d", extra="permission_profile: admin\n")
            snapshot = _build(proj, user)
            self.assertEqual(snapshot.agents, {})
            self.assertIn("invalid_profile", self._issues_for(snapshot))

    def test_timeout_out_of_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "zero.md", "zero", "d", extra="task_timeout_seconds: 0\n")
            _write_md(
                proj,
                "huge.md",
                "huge",
                "d",
                extra=f"task_timeout_seconds: {MAX_TASK_TIMEOUT_SECONDS + 1}\n",
            )
            _write_md(
                proj, "float.md", "float", "d", extra="task_timeout_seconds: 30.0\n"
            )
            _write_md(
                proj,
                "min.md",
                "min",
                "d",
                extra=f"task_timeout_seconds: {MIN_TASK_TIMEOUT_SECONDS}\n",
            )
            snapshot = _build(proj, user)
            self.assertEqual([a.name for a in snapshot.agents.values()], ["min"])
            min_config = snapshot.get("min")
            assert min_config is not None
            self.assertEqual(min_config.task_timeout_seconds, MIN_TASK_TIMEOUT_SECONDS)
            self.assertEqual(self._issues_for(snapshot).count("invalid_timeout"), 3)

    def test_defaults_applied(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            proj, user = _roots(base)
            _write_md(proj, "a.md", "a", "d")
            snapshot = _build(proj, user)
            config = snapshot.get("a")
            assert config is not None
            self.assertEqual(config.permission_profile, "research")
            self.assertEqual(config.tools, ToolsSpec())
            self.assertEqual(config.default_skills, ())
            self.assertIsNone(config.model)
            self.assertEqual(config.task_timeout_seconds, 600)


class ResolveModelConfigTest(unittest.TestCase):
    @staticmethod
    def _environ(vars: dict[str, str]):
        def lookup(name: str) -> str | None:
            return vars.get(name)

        return lookup

    def test_inherits_everything_without_override(self):
        resolved = resolve_child_model_config(PARENT_MODEL, None, self._environ({}))
        self.assertEqual(resolved, PARENT_MODEL)

    def test_field_by_field_override_keeps_key_when_provider_unchanged(self):
        override = ModelOverride(
            model="gpt-4o-mini", timeout=30.0, reasoning_mode="off"
        )
        resolved = resolve_child_model_config(PARENT_MODEL, override, self._environ({}))
        self.assertEqual(resolved.api_type, "openai")
        self.assertEqual(resolved.model, "gpt-4o-mini")
        self.assertEqual(resolved.base_url, PARENT_MODEL.base_url)
        self.assertEqual(resolved.timeout, 30.0)
        self.assertEqual(resolved.reasoning_mode, "off")
        self.assertEqual(resolved.api_key, "parent-secret")

    def test_explicit_null_disables_parent_reasoning(self):
        parent = ModelConfig(
            api_type=PARENT_MODEL.api_type,
            model=PARENT_MODEL.model,
            base_url=PARENT_MODEL.base_url,
            api_key=PARENT_MODEL.api_key,
            timeout=PARENT_MODEL.timeout,
            reasoning_mode="enabled",
            reasoning_effort="high",
        )
        resolved = resolve_child_model_config(
            parent,
            ModelOverride(reasoning_mode=None, reasoning_effort=None),
            self._environ({}),
        )
        self.assertIsNone(resolved.reasoning_mode)
        self.assertIsNone(resolved.reasoning_effort)

    def test_api_type_change_requires_api_key_env(self):
        override = ModelOverride(api_type="deepseek")
        with self.assertRaises(ModelAdapterError) as cm:
            resolve_child_model_config(PARENT_MODEL, override, self._environ({}))
        self.assertIn("api_key_env", str(cm.exception))

    def test_base_url_change_requires_api_key_env(self):
        override = ModelOverride(base_url="https://proxy.example.com/v1")
        with self.assertRaises(ModelAdapterError):
            resolve_child_model_config(PARENT_MODEL, override, self._environ({}))

    def test_api_key_env_missing_var_fails_with_name_only(self):
        override = ModelOverride(api_type="deepseek", api_key_env="DEEPSEEK_KEY")
        with self.assertRaises(ModelAdapterError) as cm:
            resolve_child_model_config(PARENT_MODEL, override, self._environ({}))
        message = str(cm.exception)
        self.assertIn("DEEPSEEK_KEY", message)
        self.assertNotIn("parent-secret", message)
        self.assertNotIn("sk-", message)

    def test_api_key_env_provides_key(self):
        override = ModelOverride(
            api_type="deepseek",
            base_url="https://api.deepseek.com/v1",
            api_key_env="DEEPSEEK_KEY",
        )
        resolved = resolve_child_model_config(
            PARENT_MODEL, override, self._environ({"DEEPSEEK_KEY": "sk-ds-123"})
        )
        self.assertEqual(resolved.api_key, "sk-ds-123")
        self.assertEqual(resolved.api_type, "deepseek")

    def test_api_key_env_wins_over_parent_key(self):
        override = ModelOverride(api_key_env="CHILD_KEY")
        resolved = resolve_child_model_config(
            PARENT_MODEL, override, self._environ({"CHILD_KEY": "sk-child-456"})
        )
        self.assertEqual(resolved.api_key, "sk-child-456")
        self.assertEqual(resolved.api_type, PARENT_MODEL.api_type)

    def test_empty_env_var_treated_as_missing(self):
        override = ModelOverride(api_key_env="EMPTY_KEY")
        with self.assertRaises(ModelAdapterError) as cm:
            resolve_child_model_config(
                PARENT_MODEL, override, self._environ({"EMPTY_KEY": ""})
            )
        self.assertIn("EMPTY_KEY", str(cm.exception))


class AgentSnapshotValidationTest(unittest.TestCase):
    def test_unknown_tools_and_missing_default_skills_are_excluded(self):
        snapshot = AgentConfigSet(
            {
                "valid": AgentConfig(name="valid", description="ok", instructions="i"),
                "bad-tool": AgentConfig(
                    name="bad-tool",
                    description="bad",
                    instructions="i",
                    tools=ToolsSpec(allow=("not_a_tool",)),
                ),
                "bad-skill": AgentConfig(
                    name="bad-skill",
                    description="bad",
                    instructions="i",
                    default_skills=("missing",),
                ),
            }
        )
        validated = validate_agent_snapshot(
            snapshot, valid_tool_names=frozenset({"read_file"}), skills_snapshot=None
        )
        self.assertEqual(tuple(validated.agents), ("valid",))
        self.assertEqual(
            {issue.code for issue in validated.issues},
            {"unknown_tool", "missing_default_skill"},
        )


class StateMachineTest(unittest.TestCase):
    def test_legal_transitions(self):
        legal: list[tuple[SubagentStatus, SubagentStatus]] = [
            ("queued", "running"),
            ("queued", "cancelled"),
            ("running", "waiting_approval"),
            ("running", "cancelling"),
            ("running", "completed"),
            ("running", "failed"),
            ("waiting_approval", "running"),
            ("waiting_approval", "cancelling"),
            ("cancelling", "cancelled"),
        ]
        for from_status, to_status in legal:
            self.assertTrue(
                transition_allowed(from_status, to_status),
                f"{from_status} -> {to_status}",
            )

    def test_terminal_idempotent(self):
        for status in ("completed", "failed", "cancelled"):
            self.assertTrue(transition_allowed(status, status), status)

    def test_illegal_transitions(self):
        illegal: list[tuple[SubagentStatus, SubagentStatus]] = [
            ("queued", "queued"),
            ("running", "running"),
            ("waiting_approval", "waiting_approval"),
            ("cancelling", "cancelling"),
            ("queued", "failed"),
            ("queued", "cancelling"),
            ("queued", "completed"),
            ("waiting_approval", "completed"),
            ("waiting_approval", "failed"),
            ("cancelling", "running"),
            ("cancelling", "completed"),
            ("cancelling", "failed"),
            ("cancelled", "running"),
            ("completed", "failed"),
            ("failed", "cancelled"),
        ]
        for from_status, to_status in illegal:
            self.assertFalse(
                transition_allowed(from_status, to_status),
                f"{from_status} -> {to_status}",
            )


class PermissionCompositionTest(unittest.TestCase):
    def test_profile_baselines(self):
        research = effective_tool_names("research", PARENT_VISIBLE)
        self.assertEqual(research, PERMISSION_PROFILES["research"])
        verify = effective_tool_names("verify", PARENT_VISIBLE)
        self.assertIn("run_command", verify)
        self.assertNotIn("edit_file", verify)
        self.assertNotIn("write_file", verify)
        self.assertNotIn("save_memory", verify)
        edit = effective_tool_names("edit", PARENT_VISIBLE)
        self.assertIn("edit_file", edit)
        self.assertIn("write_file", edit)
        self.assertIn("save_memory", edit)
        self.assertNotIn("run_command", edit)

    def test_full_is_parent_visible_minus_subagent_tools(self):
        full = effective_tool_names("full", PARENT_VISIBLE)
        self.assertEqual(full, PARENT_VISIBLE - SUBAGENT_TOOLS)

    def test_allow_extends_within_parent(self):
        result = effective_tool_names("research", PARENT_VISIBLE, allow=["edit_file"])
        self.assertIn("edit_file", result)

    def test_allow_cannot_introduce_unavailable_tool(self):
        result = effective_tool_names("research", PARENT_VISIBLE, allow=["mcp_invoke"])
        self.assertNotIn("mcp_invoke", result)

    def test_deny_always_removes(self):
        result = effective_tool_names(
            "full", PARENT_VISIBLE, deny=["read_file", "run_command"]
        )
        self.assertNotIn("read_file", result)
        self.assertNotIn("run_command", result)

    def test_deny_wins_over_explicit_allow(self):
        result = effective_tool_names(
            "research", PARENT_VISIBLE, allow=["edit_file"], deny=["edit_file"]
        )
        self.assertNotIn("edit_file", result)

    def test_subagent_tools_always_removed(self):
        result = effective_tool_names(
            "research", PARENT_VISIBLE, allow=["delegate_task", "wait_subagents"]
        )
        self.assertTrue(result.isdisjoint(SUBAGENT_TOOLS))
        full = effective_tool_names("full", PARENT_VISIBLE)
        self.assertTrue(full.isdisjoint(SUBAGENT_TOOLS))

    def test_parent_ceiling_always_applied(self):
        # profile tools not visible in the parent catalog cannot appear.
        result = effective_tool_names("edit", {"read_file"}, allow=["write_file"])
        self.assertEqual(result, frozenset({"read_file"}))

    def test_unknown_profile_raises(self):
        with self.assertRaises(ValueError):
            effective_tool_names("admin", PARENT_VISIBLE)

    def test_profiles_are_frozensets(self):
        for profile, tools in PERMISSION_PROFILES.items():
            self.assertIsInstance(tools, frozenset)
            self.assertIn(profile, ("research", "verify", "edit", "full"))


class ConstantsAndTaskIdTest(unittest.TestCase):
    def test_budget_constants(self):
        self.assertEqual(MAX_COMBINED_CONTEXT_CHARS, 24_000)
        self.assertEqual(MAX_PROMPT_CHARS, 16_000)
        self.assertEqual(MAX_DESCRIPTION_CHARS, 2_000)
        self.assertEqual(
            SUBAGENT_TOOLS,
            frozenset(
                {"delegate_task", "wait_subagents", "list_subagents", "cancel_subagent"}
            ),
        )

    def test_task_id_charset_prefix_and_uniqueness(self):
        ids = {generate_task_id() for _ in range(200)}
        self.assertEqual(len(ids), 200)
        for task_id in ids:
            self.assertTrue(is_valid_task_id(task_id))
            self.assertTrue(task_id.startswith("sub-"))
            self.assertRegex(task_id, r"^[a-z0-9_-]+$")

    def test_is_valid_task_id(self):
        self.assertTrue(is_valid_task_id("sub-1a2b3c"))
        self.assertTrue(is_valid_task_id("sub-abc_123"))
        self.assertFalse(is_valid_task_id(""))
        self.assertFalse(is_valid_task_id("SUB-1"))
        self.assertFalse(is_valid_task_id("../evil"))
        self.assertFalse(is_valid_task_id("sub-with space"))


class AgentRegistryTest(unittest.TestCase):
    def test_refresh_snapshot_and_get(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace = base / "workspace"
            user_root = base / "user_home" / ".agents" / "subagents"
            proj = workspace / ".agents" / "subagents"
            _write_md(proj, "alpha.md", "alpha", "project agent")
            _write_md(user_root, "beta.md", "beta", "user agent")
            registry = AgentRegistry(workspace_root=workspace, user_root=user_root)
            snapshot = registry.refresh()
            self.assertEqual(sorted(snapshot.agents), ["alpha", "beta"])
            self.assertIs(registry.snapshot(), snapshot)
            self.assertIsNotNone(registry.get("alpha"))
            self.assertIsNone(registry.get("missing"))

    def test_project_root_property(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            registry = AgentRegistry(
                workspace_root=base / "workspace", user_root=base / "user"
            )
            self.assertEqual(
                registry.project_root,
                (base / "workspace" / ".agents" / "subagents").resolve(),
            )

    def test_refresh_caches_last_issues(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            workspace = base / "workspace"
            user_root = base / "user_home" / ".agents" / "subagents"
            proj = workspace / ".agents" / "subagents"
            _write_md(proj, "bad.md", "bad.name", "d")
            registry = AgentRegistry(workspace_root=workspace, user_root=user_root)
            snapshot = registry.refresh()
            self.assertEqual(snapshot.agents, {})
            self.assertTrue(
                any(issue.code == "invalid_name" for issue in snapshot.issues)
            )


if __name__ == "__main__":
    unittest.main()
