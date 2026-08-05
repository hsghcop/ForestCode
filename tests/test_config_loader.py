import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forestcode.config import ConfigError, load_config as _load_config_raw

_MODEL_ENV = {
    "FORESTCODE_MODEL": "deepseek-chat",
    "FORESTCODE_BASE_URL": "https://api.deepseek.com/v1",
    "FORESTCODE_API_KEY": "secret",
}


def _write_env(temp_dir: str, lines: list[str]) -> Path:
    path = Path(temp_dir) / ".env"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def load_config(*args, **kwargs):
    kwargs.setdefault("settings_file", None)
    return _load_config_raw(*args, **kwargs)


def _write_settings(temp_dir: str, text: str) -> Path:
    path = Path(temp_dir) / "settings.json"
    path.write_text(text, encoding="utf-8")
    return path


class DefaultsTest(unittest.TestCase):
    def test_defaults_when_only_model_present(self):
        with patch.dict(os.environ, _MODEL_ENV, clear=True):
            config = load_config(env_file=None)
        self.assertEqual(config.agent.runtime.max_turns, 10)
        self.assertEqual(config.agent.budget.max_context_chars, 40_000)
        self.assertEqual(config.agent.budget.max_recent_messages, 20)
        self.assertEqual(config.agent.budget.max_tool_result_chars, 2_000)
        self.assertEqual(config.agent.runtime.compact_trigger_entries, 40)
        self.assertTrue(config.agent.runtime.auto_compact)
        self.assertEqual(config.agent.tool_output_max_chars, 20_000)
        self.assertFalse(config.agent.features.enable_command_tools)
        self.assertTrue(config.agent.features.include_project_rules)


class ALayerPrecedenceTest(unittest.TestCase):
    def test_cli_beats_process_env_beats_file_beats_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = _write_env(temp_dir, ["FORESTCODE_MAX_TURNS=3"])
            # file gives 3
            with patch.dict(os.environ, _MODEL_ENV, clear=True):
                self.assertEqual(load_config(env_file=env_file).agent.runtime.max_turns, 3)
            # process env (7) overrides file (3)
            with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_MAX_TURNS": "7"}, clear=True):
                self.assertEqual(load_config(env_file=env_file).agent.runtime.max_turns, 7)
                # CLI (5) overrides process env (7)
                self.assertEqual(
                    load_config({"max_turns": 5}, env_file=env_file).agent.runtime.max_turns, 5
                )

    def test_zero_allowed_for_recent_messages_and_memory(self):
        with patch.dict(
            os.environ,
            {
                **_MODEL_ENV,
                "FORESTCODE_MAX_RECENT_MESSAGES": "0",
                "FORESTCODE_MAX_MEMORY_CHARS": "0",
                "FORESTCODE_MAX_TOOL_RESULT_CHARS": "0",
            },
            clear=True,
        ):
            config = load_config(env_file=None)
        self.assertEqual(config.agent.budget.max_recent_messages, 0)
        self.assertEqual(config.agent.budget.max_memory_chars, 0)
        self.assertEqual(config.agent.budget.max_tool_result_chars, 0)

    def test_compaction_env_keys_are_a_layer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = _write_env(
                temp_dir,
                [
                    "FORESTCODE_COMPACT_TRIGGER_ENTRIES=11",
                    "FORESTCODE_AUTO_COMPACT=0",
                    "FORESTCODE_MAX_TOOL_RESULT_CHARS=123",
                ],
            )
            with patch.dict(os.environ, _MODEL_ENV, clear=True):
                config = load_config(env_file=env_file)
            self.assertEqual(config.agent.runtime.compact_trigger_entries, 11)
            self.assertFalse(config.agent.runtime.auto_compact)
            self.assertEqual(config.agent.budget.max_tool_result_chars, 123)

            with patch.dict(
                os.environ,
                {
                    **_MODEL_ENV,
                    "FORESTCODE_COMPACT_TRIGGER_ENTRIES": "12",
                    "FORESTCODE_AUTO_COMPACT": "1",
                    "FORESTCODE_MAX_TOOL_RESULT_CHARS": "456",
                },
                clear=True,
            ):
                config = load_config(
                    {
                        "compact_trigger_entries": 13,
                        "auto_compact": False,
                        "max_tool_result_chars": 789,
                    },
                    env_file=env_file,
                )
            self.assertEqual(config.agent.runtime.compact_trigger_entries, 13)
            self.assertFalse(config.agent.runtime.auto_compact)
            self.assertEqual(config.agent.budget.max_tool_result_chars, 789)


class SettingsFilePrecedenceTest(unittest.TestCase):
    def test_settings_file_supplies_model_and_agent_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = _write_settings(
                temp_dir,
                """
{
  "api_key": "settings-secret",
  "model": "settings-model",
  "base_url": "https://settings.example/v1",
  "max_turns": 6,
  "enable_command_tools": true
}
""",
            )
            with patch.dict(os.environ, {}, clear=True):
                config = _load_config_raw(env_file=None, settings_file=settings_file)
        self.assertEqual(config.model.model, "settings-model")
        self.assertEqual(config.model.base_url, "https://settings.example/v1")
        self.assertEqual(config.agent.runtime.max_turns, 6)
        self.assertTrue(config.agent.features.enable_command_tools)

    def test_dotenv_overrides_settings_for_a_layer_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = _write_settings(
                temp_dir,
                """
{
  "api_key": "settings-secret",
  "model": "settings-model",
  "base_url": "https://settings.example/v1",
  "max_turns": 6
}
""",
            )
            env_file = _write_env(
                temp_dir,
                [
                    "FORESTCODE_MODEL=file-model",
                    "FORESTCODE_MAX_TURNS=8",
                ],
            )
            with patch.dict(os.environ, {}, clear=True):
                config = _load_config_raw(env_file=env_file, settings_file=settings_file)
        self.assertEqual(config.model.model, "file-model")
        self.assertEqual(config.agent.runtime.max_turns, 8)

    def test_dotenv_does_not_override_settings_for_command_tools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = _write_settings(
                temp_dir,
                """
{
  "api_key": "settings-secret",
  "model": "settings-model",
  "base_url": "https://settings.example/v1",
  "enable_command_tools": true
}
""",
            )
            env_file = _write_env(temp_dir, ["FORESTCODE_ENABLE_COMMAND_TOOLS=0"])
            with patch.dict(os.environ, {}, clear=True):
                config = _load_config_raw(env_file=env_file, settings_file=settings_file)
        self.assertTrue(config.agent.features.enable_command_tools)

    def test_process_env_and_cli_override_settings_for_command_tools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = _write_settings(
                temp_dir,
                """
{
  "api_key": "settings-secret",
  "model": "settings-model",
  "base_url": "https://settings.example/v1",
  "enable_command_tools": true
}
""",
            )
            with patch.dict(os.environ, {"FORESTCODE_ENABLE_COMMAND_TOOLS": "0"}, clear=True):
                config = _load_config_raw(env_file=None, settings_file=settings_file)
                cli_config = _load_config_raw(
                    {"enable_command_tools": True}, env_file=None, settings_file=settings_file
                )
        self.assertFalse(config.agent.features.enable_command_tools)
        self.assertTrue(cli_config.agent.features.enable_command_tools)


class BLayerBansDotenvTest(unittest.TestCase):
    def test_dotenv_cannot_enable_command_tools(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = _write_env(temp_dir, ["FORESTCODE_ENABLE_COMMAND_TOOLS=1"])
            with patch.dict(os.environ, _MODEL_ENV, clear=True):
                config = load_config(env_file=env_file)
        self.assertFalse(config.agent.features.enable_command_tools)

    def test_process_env_can_enable(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_ENABLE_COMMAND_TOOLS": "1"}, clear=True):
            config = load_config(env_file=None)
        self.assertTrue(config.agent.features.enable_command_tools)

    def test_cli_can_override_process_env_off(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_ENABLE_COMMAND_TOOLS": "1"}, clear=True):
            config = load_config({"enable_command_tools": False}, env_file=None)
        self.assertFalse(config.agent.features.enable_command_tools)


class StrictBoolTest(unittest.TestCase):
    def test_typo_raises(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_ENABLE_COMMAND_TOOLS": "treu"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "FORESTCODE_ENABLE_COMMAND_TOOLS"):
                load_config(env_file=None)

    def test_off_parses_false(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_ENABLE_COMMAND_TOOLS": "off"}, clear=True):
            self.assertFalse(load_config(env_file=None).agent.features.enable_command_tools)


class ValidationTest(unittest.TestCase):
    def test_max_turns_zero_rejected(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_MAX_TURNS": "0"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "FORESTCODE_MAX_TURNS"):
                load_config(env_file=None)

    def test_context_chars_zero_rejected(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_MAX_CONTEXT_CHARS": "0"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "FORESTCODE_MAX_CONTEXT_CHARS"):
                load_config(env_file=None)

    def test_compact_trigger_entries_zero_rejected(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_COMPACT_TRIGGER_ENTRIES": "0"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "FORESTCODE_COMPACT_TRIGGER_ENTRIES"):
                load_config(env_file=None)

    def test_negative_rejected(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_MAX_MEMORY_CHARS": "-1"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "FORESTCODE_MAX_MEMORY_CHARS"):
                load_config(env_file=None)

    def test_non_integer_rejected(self):
        with patch.dict(os.environ, {**_MODEL_ENV, "FORESTCODE_MAX_TURNS": "lots"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "FORESTCODE_MAX_TURNS"):
                load_config(env_file=None)

    def test_missing_model_raises_config_error(self):
        with patch.dict(os.environ, {"FORESTCODE_API_KEY": "secret"}, clear=True):
            with self.assertRaisesRegex(ConfigError, "FORESTCODE_MODEL"):
                load_config(env_file=None)


class SecretNotLeakedTest(unittest.TestCase):
    def test_api_key_absent_from_repr(self):
        with patch.dict(os.environ, _MODEL_ENV, clear=True):
            config = load_config(env_file=None)
        self.assertNotIn("secret", repr(config))
        self.assertNotIn("secret", repr(config.model))


if __name__ == "__main__":
    unittest.main()
