"""Tests for $skill-name token parsing and the pending one-shot selection (PRD R4-R6)."""

from __future__ import annotations

import unittest

from forestcode.skills.activation import parse_skill_token
from forestcode.skills.pending import PendingSkillSelection


class ParseSkillTokenTest(unittest.TestCase):
    def test_leading_token_removed_task_kept(self):
        result = parse_skill_token("$refactor 帮我重构这个文件")
        self.assertEqual(result.name, "refactor")
        self.assertEqual(result.task, "帮我重构这个文件")
        self.assertIsNone(result.error)

    def test_token_alone_is_user_error(self):
        result = parse_skill_token("$refactor")
        self.assertIsNone(result.name)
        self.assertIsNone(result.task)
        self.assertIsNotNone(result.error)

    def test_token_with_only_whitespace_is_user_error(self):
        result = parse_skill_token("$refactor   ")
        self.assertIsNotNone(result.error)

    def test_plain_text_untouched(self):
        result = parse_skill_token("refactor this file")
        self.assertIsNone(result.name)
        self.assertEqual(result.task, "refactor this file")

    def test_token_not_at_start_untouched(self):
        result = parse_skill_token("tell me about $refactor")
        self.assertIsNone(result.name)
        self.assertEqual(result.task, "tell me about $refactor")

    def test_uppercase_token_not_parsed(self):
        result = parse_skill_token("$Refactor please")
        self.assertIsNone(result.name)
        self.assertEqual(result.task, "$Refactor please")

    def test_underscore_and_dash_allowed(self):
        result = parse_skill_token("$my_skill-x do it")
        self.assertEqual(result.name, "my_skill-x")
        self.assertEqual(result.task, "do it")

    def test_digit_leading_name_allowed(self):
        result = parse_skill_token("$skill2 do it")
        self.assertEqual(result.name, "skill2")

    def test_punctuation_after_name_breaks_token(self):
        result = parse_skill_token("$skill. do it")
        self.assertIsNone(result.name)
        self.assertEqual(result.task, "$skill. do it")

    def test_newline_separated_token(self):
        result = parse_skill_token("$skill\ndo it")
        self.assertEqual(result.name, "skill")
        self.assertEqual(result.task, "do it")


class PendingSkillSelectionTest(unittest.TestCase):
    def test_replace_clear_and_marker(self):
        pending = PendingSkillSelection()
        self.assertIsNone(pending.marker_text())
        pending.replace("demo")
        self.assertEqual(pending.marker_text(), "[Skill: demo]")
        pending.replace("other")
        self.assertEqual(pending.marker_text(), "[Skill: other]")
        pending.clear()
        self.assertIsNone(pending.marker_text())

    def test_instances_are_independent(self):
        first = PendingSkillSelection()
        second = PendingSkillSelection()
        first.replace("a")
        self.assertIsNone(second.marker_text())


if __name__ == "__main__":
    unittest.main()
