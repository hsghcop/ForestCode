import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from forestcode.envfile import lookup_env, read_env_file


class ReadEnvFileTest(unittest.TestCase):
    def _write(self, temp_dir: str, content: str) -> Path:
        path = Path(temp_dir) / ".env"
        path.write_text(content, encoding="utf-8")
        return path

    def test_returns_empty_for_none_or_missing(self):
        self.assertEqual(read_env_file(None), {})
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(read_env_file(Path(temp_dir) / "nope.env"), {})

    def test_parses_basic_pairs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "FORESTCODE_MODEL=deepseek-chat\nFORESTCODE_TIMEOUT=45\n")
            values = read_env_file(path)
        self.assertEqual(values["FORESTCODE_MODEL"], "deepseek-chat")
        self.assertEqual(values["FORESTCODE_TIMEOUT"], "45")

    def test_skips_comments_blanks_and_keyless_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "# a comment\n\nFORESTCODE_MODEL=x\nnoequalshere\n=value\n")
            values = read_env_file(path)
        self.assertEqual(values, {"FORESTCODE_MODEL": "x"})

    def test_strips_one_layer_of_matching_quotes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, 'A="quoted"\nB=\'single\'\nC="mismatch\'\n')
            values = read_env_file(path)
        self.assertEqual(values["A"], "quoted")
        self.assertEqual(values["B"], "single")
        self.assertEqual(values["C"], '"mismatch\'')

    def test_tolerates_utf8_bom(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write(temp_dir, "﻿FORESTCODE_API_TYPE=deepseek\n")
            values = read_env_file(path)
        self.assertEqual(values["FORESTCODE_API_TYPE"], "deepseek")


class LookupEnvTest(unittest.TestCase):
    def test_process_env_overrides_file(self):
        with patch.dict(os.environ, {"K": "from-process"}, clear=True):
            self.assertEqual(lookup_env("K", {"K": "from-file"}), "from-process")

    def test_falls_back_to_file(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(lookup_env("K", {"K": "from-file"}), "from-file")

    def test_missing_returns_none(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(lookup_env("K", {}))


if __name__ == "__main__":
    unittest.main()
