"""Leaf module for `.env` parsing and layered env lookup.

This is a dependency leaf: it imports nothing from inside ``forestcode`` so both
``models.config`` and ``config.loader`` can use it without forming an import
cycle through the ``config`` package.
"""

from __future__ import annotations

import os
from pathlib import Path


def read_env_file(env_file: str | Path | None) -> dict[str, str]:
    """Parse a flat ``KEY=VALUE`` ``.env`` file into a dict.

    Skips blank lines, comments (``#``) and lines without ``=``. Tolerates a
    UTF-8 BOM and strips a single layer of matching surrounding quotes.
    """
    if env_file is None:
        return {}

    path = Path(env_file)
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue

        values[key] = _strip_optional_quotes(value.strip())
    return values


def lookup_env(name: str, file_values: dict[str, str]) -> str | None:
    """Look up ``name`` with process env taking precedence over the file."""
    return os.getenv(name) or file_values.get(name)


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
