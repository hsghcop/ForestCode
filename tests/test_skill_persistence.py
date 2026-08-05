"""Tests for the session boundary: skill bodies never reach session JSONL (PRD R7, AC3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from forestcode.config import AgentRuntimeConfig
from forestcode.core.fake_model import FakeModelClient
from forestcode.core.types import ModelOutput, ToolCall
from forestcode.memory import SessionStore
from forestcode.runtime.factory import build_agent_loop
from forestcode.skills import SkillRegistry
from forestcode.skills.catalog import SkillCatalogContextProvider, skill_body_fragment


def _make_skill(workspace: Path, name: str, body: str) -> None:
    skill_dir = workspace / ".agents" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: skill {name}\n---\n{body}", encoding="utf-8"
    )


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as _:
            continue
    return records


class SkillPersistenceTest(unittest.TestCase):
    def _run_loop(
        self,
        workspace: Path,
        session_id: str,
        model,
        user_task: str,
        transient_fragments=(),
        skills_snapshot=None,
    ):
        store = SessionStore(workspace)
        loop = build_agent_loop(
            model,
            workspace,
            agent=AgentRuntimeConfig(),
            session_id=session_id,
            enable_write_tools=True,
            session_store=store,
            transient_fragments=transient_fragments,
            skills_snapshot=skills_snapshot,
        )
        state = loop.run(user_task)
        return state, store

    def test_load_skill_result_is_not_persisted(self):
        SECRET = "TOP-SECRET-SKILL-BODY-12345"
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo", SECRET)
            registry = SkillRegistry(workspace)
            snapshot = registry.refresh()

            model = FakeModelClient(
                [
                    ModelOutput(
                        tool_calls=[
                            ToolCall(
                                id="c1", name="load_skill", arguments={"name": "demo"}
                            )
                        ]
                    ),
                    ModelOutput(text="done"),
                ]
            )
            state, store = self._run_loop(
                workspace, "s1", model, "use the skill", (), skills_snapshot=snapshot
            )

            # the current run sees the body in messages
            self.assertTrue(any(SECRET in (m.content or "") for m in state.messages))

            # ... but the recorded session must not contain it
            raw = _load_jsonl(store._jsonl_path("s1"))
            blob = json.dumps(raw, ensure_ascii=False)
            self.assertNotIn(SECRET, blob)
            self.assertNotIn("load_skill", blob)
            # run stats carry no body either
            runs = [r for r in raw if r.get("_t") == "run"]
            self.assertTrue(runs)
            self.assertNotIn(SECRET, json.dumps(runs, ensure_ascii=False))

    def test_manual_activation_fragment_is_not_persisted(self):
        SECRET = "MANUAL-BODY-NEVER-RECORDED"
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo", SECRET)
            registry = SkillRegistry(workspace)
            snapshot = registry.refresh()
            loaded = snapshot.load("demo")
            assert loaded is not None
            fragments = (
                SkillCatalogContextProvider().build(snapshot),
                skill_body_fragment(loaded),
            )
            model = FakeModelClient([ModelOutput(text="ok")])
            state, store = self._run_loop(workspace, "s2", model, "question", fragments)

            # model input contains the body...
            self.assertTrue(
                any(SECRET in (m.content or "") for m in model.inputs[0].messages)
            )

            # ...but the session file does not
            blob = json.dumps(_load_jsonl(store._jsonl_path("s2")), ensure_ascii=False)
            self.assertNotIn(SECRET, blob)

    def test_large_load_skill_result_stays_inline_not_written_to_disk(self):
        """Skill bodies above the file-ref threshold stay inline: nothing is
        spilled to .forestcode/tool-results/, the next model request sees the
        full body (no file reference), and JSONL still has no body (PRD R7)."""
        BIG = "B" * 9_000  # > ToolExecutor file-ref threshold (8_000)
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo", BIG)
            registry = SkillRegistry(workspace)
            snapshot = registry.refresh()

            model = FakeModelClient(
                [
                    ModelOutput(
                        tool_calls=[
                            ToolCall(
                                id="c1", name="load_skill", arguments={"name": "demo"}
                            )
                        ]
                    ),
                    ModelOutput(text="done"),
                ]
            )
            state, store = self._run_loop(
                workspace, "s-big", model, "use the skill", (), skills_snapshot=snapshot
            )

            # 1. no tool output was written under .forestcode/tool-results/
            tool_results = workspace / ".forestcode" / "tool-results"
            spilled = (
                [p for p in tool_results.rglob("*") if p.is_file()]
                if tool_results.exists()
                else []
            )
            self.assertEqual(spilled, [])

            # 2. the follow-up model request sees the full body inline, not a
            #    file reference (and the current run sees it in messages too)
            self.assertGreaterEqual(len(model.inputs), 2)
            follow_up = "".join(m.content or "" for m in model.inputs[1].messages)
            self.assertIn(BIG, follow_up)
            self.assertNotIn("[output written to", follow_up)
            self.assertTrue(any(BIG in (m.content or "") for m in state.messages))

            # 3. the recorded session still has no body
            raw = _load_jsonl(store._jsonl_path("s-big"))
            self.assertNotIn(BIG, json.dumps(raw, ensure_ascii=False))
            self.assertNotIn("load_skill", json.dumps(raw, ensure_ascii=False))
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _make_skill(workspace, "demo", "BODY")
            registry = SkillRegistry(workspace)
            snapshot = registry.refresh()
            loaded = snapshot.load("demo")
            assert loaded is not None
            fragments = (skill_body_fragment(loaded),)
            model = FakeModelClient([ModelOutput(text="ok")])
            state, store = self._run_loop(
                workspace, "s3", model, "please do it", fragments
            )
            self.assertEqual(state.user_task, "please do it")
            blob = json.dumps(_load_jsonl(store._jsonl_path("s3")), ensure_ascii=False)
            self.assertNotIn("BODY", blob)


if __name__ == "__main__":
    unittest.main()
