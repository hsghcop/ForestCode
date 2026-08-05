"""Tests for the reusable single-select skill picker (PRD R5, design §Selection UI)."""

from __future__ import annotations

import io
import unittest

from forestcode.skills.types import SkillDescriptor, SkillSnapshot
from forestcode.terminal.skill_select import (
    _PLAIN_HEADER,
    _PT_HEADER,
    _pt_menu_step,
    make_numbered_skill_selector,
)


def _snapshot(names=("alpha", "beta")):
    descriptors = tuple(
        SkillDescriptor(
            name=name,
            description=f"desc {name}",
            root=__import__("pathlib").Path("/nonexistent"),
            entry_path=__import__("pathlib").Path(f"/nonexistent/{name}/SKILL.md"),
            source="project",
        )
        for name in names
    )
    return SkillSnapshot(descriptors=descriptors)


class NumberedSkillSelectorTest(unittest.TestCase):
    def _selector(self, answers):
        out = io.StringIO()
        inputs = iter(answers)

        def input_func(_prompt):
            return next(inputs)

        return make_numbered_skill_selector(input_func=input_func, stdout=out), out

    def test_prints_numbered_list_and_returns_choice(self):
        selector, out = self._selector(["1"])
        snapshot = _snapshot()
        self.assertEqual(selector(snapshot), "alpha")
        text = out.getvalue()
        self.assertIn("1. alpha — desc alpha", text)
        self.assertIn("2. beta — desc beta", text)

    def test_second_choice(self):
        selector, _out = self._selector(["2"])
        self.assertEqual(selector(_snapshot()), "beta")

    def test_empty_enter_cancels(self):
        selector, _out = self._selector([""])
        self.assertIsNone(selector(_snapshot()))

    def test_invalid_then_valid(self):
        selector, out = self._selector(["9", "oops", "2"])
        self.assertEqual(selector(_snapshot()), "beta")
        self.assertIn("invalid selection", out.getvalue())

    def test_single_skill(self):
        selector, _out = self._selector(["1"])
        self.assertEqual(selector(_snapshot(["only"])), "only")

    def test_plain_header_says_number_or_empty_to_cancel(self):
        # F6: the plain prompt must not claim Enter cancels — Enter (number)
        # confirms; empty input cancels.
        selector, out = self._selector(["1"])
        selector(_snapshot())
        self.assertIn(_PLAIN_HEADER, out.getvalue())
        self.assertIn("enter number, empty to cancel", _PLAIN_HEADER)
        self.assertNotIn("Enter cancels", _PLAIN_HEADER)


class PtMenuContractTest(unittest.TestCase):
    """F6: the arrow-key menu's Enter selects, Esc/Ctrl+C cancels (testable
    pure contract), and the header text says exactly that."""

    def test_header_claims_enter_selects_esc_cancels(self):
        self.assertIn("Enter select", _PT_HEADER)
        self.assertIn("Esc/Ctrl+C cancel", _PT_HEADER)
        self.assertNotIn("Enter cancels", _PT_HEADER)

    def test_enter_confirms_current_selection(self):
        index, action = _pt_menu_step(0, "enter", 3)
        self.assertEqual(action, "select")
        self.assertEqual(index, 0)

    def test_escape_cancels(self):
        index, action = _pt_menu_step(2, "escape", 3)
        self.assertEqual(action, "cancel")
        self.assertEqual(index, 2)

    def test_ctrl_c_cancels(self):
        _index, action = _pt_menu_step(1, "c-c", 3)
        self.assertEqual(action, "cancel")

    def test_up_down_move_and_wrap(self):
        index, action = _pt_menu_step(0, "down", 3)
        self.assertEqual((index, action), (1, ""))
        index, action = _pt_menu_step(2, "down", 3)
        self.assertEqual((index, action), (0, ""))  # wraps
        index, action = _pt_menu_step(0, "up", 3)
        self.assertEqual((index, action), (2, ""))  # wraps
        index, action = _pt_menu_step(1, "up", 3)
        self.assertEqual((index, action), (0, ""))

    def test_unknown_key_does_nothing(self):
        index, action = _pt_menu_step(1, "page-down", 3)
        self.assertEqual((index, action), (1, ""))


if __name__ == "__main__":
    unittest.main()
