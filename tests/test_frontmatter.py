"""Tests for shared YAML frontmatter parsing (design §Shared frontmatter)."""

import unittest

from forestcode.config.frontmatter import FrontmatterError, parse_frontmatter


class ParseFrontmatterTest(unittest.TestCase):
    def test_valid_frontmatter_returns_mapping_and_body(self):
        text = "---\nname: demo\ndescription: a skill\n---\n# Body\nline two"
        frontmatter, body = parse_frontmatter(text)
        self.assertEqual(frontmatter, {"name": "demo", "description": "a skill"})
        self.assertEqual(body, "# Body\nline two")

    def test_body_preserves_markdown_verbatim(self):
        text = "---\nname: x\n---\n**bold**\n\n- item\n```python\nprint(1)\n```"
        _, body = parse_frontmatter(text)
        self.assertEqual(body, "**bold**\n\n- item\n```python\nprint(1)\n```")

    def test_empty_frontmatter_is_empty_mapping(self):
        frontmatter, body = parse_frontmatter("---\n---\nbody")
        self.assertEqual(frontmatter, {})
        self.assertEqual(body, "body")

    def test_missing_leading_delimiter_raises(self):
        with self.assertRaises(FrontmatterError) as cm:
            parse_frontmatter("name: x\n---\nbody")
        self.assertIn("missing leading", str(cm.exception))

    def test_unterminated_frontmatter_raises(self):
        with self.assertRaises(FrontmatterError) as cm:
            parse_frontmatter("---\nname: x\nbody")
        self.assertIn("unterminated", str(cm.exception))

    def test_malformed_yaml_reports_line(self):
        with self.assertRaises(FrontmatterError) as cm:
            parse_frontmatter("---\nname: [unclosed\n---\nbody")
        exc = cm.exception
        self.assertIn("invalid YAML", str(exc))
        self.assertIsNotNone(exc.line)

    def test_non_mapping_frontmatter_rejected(self):
        with self.assertRaises(FrontmatterError) as cm:
            parse_frontmatter("---\n- a\n- b\n---\nbody")
        self.assertIn("mapping", str(cm.exception))

    def test_crlf_normalized(self):
        text = "---\r\nname: x\r\ndescription: y\r\n---\r\nbody line"
        frontmatter, body = parse_frontmatter(text)
        self.assertEqual(frontmatter["name"], "x")
        self.assertNotIn("\r", body)

    def test_yaml_tags_rejected_by_safe_load(self):
        with self.assertRaises(FrontmatterError):
            parse_frontmatter("---\n!!python/object:os.system {}\n---\nbody")


if __name__ == "__main__":
    unittest.main()
