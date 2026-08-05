import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forestcode.models import ModelAdapterError, load_model_config_from_env


class ModelConfigTest(unittest.TestCase):
    def test_loads_config_from_environment(self):
        env = {
            "FORESTCODE_API_TYPE": "openai-compatible",
            "FORESTCODE_MODEL": "deepseek-chat",
            "FORESTCODE_BASE_URL": "https://api.deepseek.com/v1",
            "FORESTCODE_API_KEY": "secret",
            "FORESTCODE_TIMEOUT": "30.5",
            "FORESTCODE_REASONING_MODE": "auto",
            "FORESTCODE_REASONING_EFFORT": "high",
        }

        with patch.dict(os.environ, env, clear=True):
            config = load_model_config_from_env(env_file=None)

        self.assertEqual(config.api_type, "openai-compatible")
        self.assertEqual(config.model, "deepseek-chat")
        self.assertEqual(config.base_url, "https://api.deepseek.com/v1")
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.timeout, 30.5)
        self.assertEqual(config.reasoning_mode, "auto")
        self.assertEqual(config.reasoning_effort, "high")

    def test_uses_defaults_for_api_type_and_timeout(self):
        env = {
            "FORESTCODE_MODEL": "qwen-plus",
            "FORESTCODE_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "FORESTCODE_API_KEY": "secret",
            "FORESTCODE_API_TYPE": " ",
        }

        with patch.dict(os.environ, env, clear=True):
            config = load_model_config_from_env(env_file=None)

        self.assertEqual(config.api_type, "openai-compatible")
        self.assertEqual(config.timeout, 60.0)
        self.assertIsNone(config.reasoning_mode)
        self.assertIsNone(config.reasoning_effort)

    def test_requires_model(self):
        env = {
            "FORESTCODE_BASE_URL": "https://api.deepseek.com/v1",
            "FORESTCODE_API_KEY": "secret",
        }

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ModelAdapterError, "FORESTCODE_MODEL"):
                load_model_config_from_env(env_file=None)

    def test_requires_base_url(self):
        env = {
            "FORESTCODE_MODEL": "deepseek-chat",
            "FORESTCODE_API_KEY": "secret",
        }

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ModelAdapterError, "FORESTCODE_BASE_URL"):
                load_model_config_from_env(env_file=None)

    def test_requires_api_key(self):
        env = {
            "FORESTCODE_MODEL": "deepseek-chat",
            "FORESTCODE_BASE_URL": "https://api.deepseek.com/v1",
        }

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ModelAdapterError, "FORESTCODE_API_KEY"):
                load_model_config_from_env(env_file=None)

    def test_rejects_invalid_timeout(self):
        env = {
            "FORESTCODE_MODEL": "deepseek-chat",
            "FORESTCODE_BASE_URL": "https://api.deepseek.com/v1",
            "FORESTCODE_API_KEY": "secret",
            "FORESTCODE_TIMEOUT": "slow",
        }

        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(ModelAdapterError, "FORESTCODE_TIMEOUT"):
                load_model_config_from_env(env_file=None)

    def test_loads_config_from_dotenv_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "FORESTCODE_API_TYPE=openai-compatible",
                        "FORESTCODE_MODEL=deepseek-v4-pro",
                        "FORESTCODE_BASE_URL=https://api.deepseek.com",
                        "FORESTCODE_API_KEY=secret",
                        "FORESTCODE_TIMEOUT=45",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_model_config_from_env(env_file=env_file)

        self.assertEqual(config.model, "deepseek-v4-pro")
        self.assertEqual(config.base_url, "https://api.deepseek.com")
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.timeout, 45.0)

    def test_process_environment_overrides_dotenv_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\n".join(
                    [
                        "FORESTCODE_MODEL=deepseek-v4-pro",
                        "FORESTCODE_BASE_URL=https://api.deepseek.com",
                        "FORESTCODE_API_KEY=file-secret",
                    ]
                ),
                encoding="utf-8",
            )
            env = {
                "FORESTCODE_MODEL": "deepseek-v4-flash",
                "FORESTCODE_API_KEY": "process-secret",
            }

            with patch.dict(os.environ, env, clear=True):
                config = load_model_config_from_env(env_file=env_file)

        self.assertEqual(config.model, "deepseek-v4-flash")
        self.assertEqual(config.api_key, "process-secret")

    def test_loads_utf8_bom_dotenv_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env_file = Path(temp_dir) / ".env"
            env_file.write_text(
                "\ufeffFORESTCODE_API_TYPE=deepseek\n"
                "FORESTCODE_MODEL=deepseek-v4-pro\n"
                "FORESTCODE_BASE_URL=https://api.deepseek.com\n"
                "FORESTCODE_API_KEY=secret\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=True):
                config = load_model_config_from_env(env_file=env_file)

        self.assertEqual(config.api_type, "deepseek")


if __name__ == "__main__":
    unittest.main()
