import tempfile
import unittest
from pathlib import Path

from forestcode.memory import DirectMarkdownMemoryRetriever, MarkdownMemory
from forestcode.memory.markdown_memory import render_upserted_content


class MarkdownMemoryTest(unittest.TestCase):
    def test_missing_memory_returns_empty_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(MarkdownMemory(tmp).read(), "")

    def test_reads_memory_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "MEMORY.md").write_text("# MEMORY\nproject facts", encoding="utf-8")

            self.assertIn("project facts", DirectMarkdownMemoryRetriever(root).retrieve())

    def test_render_upserted_content_creates_template_for_missing_file(self):
        text = render_upserted_content("", "user-role", "user", "User is a backend engineer.")

        self.assertTrue(text.startswith("# Long-term Memory\n\n"))
        self.assertIn("## [user] user-role\nUser is a backend engineer.\n", text)

    def test_render_upserted_content_appends_new_section(self):
        existing = "# Long-term Memory\n\n## [user] user-role\nUser is a backend engineer.\n"

        text = render_upserted_content(existing, "terse-responses", "feedback", "Keep responses concise.")

        self.assertIn("## [user] user-role\nUser is a backend engineer.\n", text)
        self.assertIn("---\n\n## [feedback] terse-responses\nKeep responses concise.\n", text)

    def test_render_upserted_content_replaces_existing_name_across_type(self):
        existing = (
            "# Long-term Memory\n\n"
            "## [user] user-role\n"
            "Old content.\n\n"
            "---\n\n"
            "## [project] project-stack\n"
            "Python CLI.\n"
        )

        text = render_upserted_content(existing, "user-role", "feedback", "New content.")

        self.assertIn("## [feedback] user-role\nNew content.\n", text)
        self.assertNotIn("Old content.", text)
        self.assertIn("## [project] project-stack\nPython CLI.\n", text)

    def test_render_upserted_content_preserves_section_boundary(self):
        existing = (
            "# Long-term Memory\n\n"
            "## [user] first\n"
            "A\n\n"
            "---\n\n"
            "## [project] second\n"
            "B\n"
        )

        text = render_upserted_content(existing, "first", "user", "A2")

        self.assertIn("## [user] first\nA2\n\n---\n\n## [project] second\nB\n", text)
        self.assertIn("## [project] second\nB\n", text)

    def test_render_upserted_content_handles_empty_existing_text(self):
        text = render_upserted_content("", "project-decision", "project", "Use patch-first edits.")

        self.assertEqual(
            text,
            "# Long-term Memory\n\n## [project] project-decision\nUse patch-first edits.\n",
        )


if __name__ == "__main__":
    unittest.main()
