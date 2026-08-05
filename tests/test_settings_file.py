import tempfile
import unittest
from pathlib import Path

from forestcode.config import ConfigError
from forestcode.config.settings_file import create_template, load_settings, settings_to_env_keys


class SettingsFileTest(unittest.TestCase):
    def _write(self, temp_dir: str, text: str) -> Path:
        path = Path(temp_dir) / "settings.json"
        path.write_text(text, encoding="utf-8")
        return path

    def test_load_settings_skips_whole_line_comments_and_preserves_url(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(
                temp_dir,
                """
// comment
{
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v3"
}
""",
            )
            settings = load_settings(path)
        self.assertEqual(settings["base_url"], "https://api.deepseek.com")
        self.assertEqual(settings["model"], "deepseek-v3")

    def test_load_settings_rejects_top_level_array(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "[]")
            with self.assertRaisesRegex(ConfigError, "JSON object"):
                load_settings(path)

    def test_load_settings_rejects_invalid_json_with_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, '{"model": "x", // inline comment\n}')
            with self.assertRaisesRegex(ConfigError, "settings.json"):
                load_settings(path)

    def test_settings_to_env_keys_converts_scalars_and_skips_nulls(self):
        values = settings_to_env_keys(
            {
                "api_key": "secret",
                "auto_compact": False,
                "max_turns": 7,
                "timeout": 30.5,
                "reasoning_mode": None,
                "reasoning_effort": "",
            }
        )
        self.assertEqual(values["FORESTCODE_API_KEY"], "secret")
        self.assertEqual(values["FORESTCODE_AUTO_COMPACT"], "0")
        self.assertEqual(values["FORESTCODE_MAX_TURNS"], "7")
        self.assertEqual(values["FORESTCODE_TIMEOUT"], "30.5")
        self.assertNotIn("FORESTCODE_REASONING_MODE", values)
        self.assertNotIn("FORESTCODE_REASONING_EFFORT", values)

    def test_settings_to_env_keys_rejects_non_scalar_values(self):
        with self.assertRaisesRegex(ConfigError, "max_turns"):
            settings_to_env_keys({"max_turns": [1]})

    def test_create_template_creates_parent_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".forestcode" / "settings.json"
            create_template(path)
            self.assertTrue(path.exists())
            text = path.read_text(encoding="utf-8")
        self.assertIn("Only whole-line // comments are supported", text)
        self.assertIn('"base_url": "https://api.deepseek.com"', text)


if __name__ == "__main__":
    unittest.main()
