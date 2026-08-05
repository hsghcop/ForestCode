"""Canonical path normalization shared across the tools layer."""

from __future__ import annotations

import os
from pathlib import Path


def canonical(path: str | Path) -> Path:
    """Return a normalized absolute path for cross-platform comparisons."""
    return Path(os.path.normcase(Path(path).resolve(strict=False)))
