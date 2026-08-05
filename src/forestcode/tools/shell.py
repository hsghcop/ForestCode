"""Shell resolution helpers for command execution tools."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def resolve_shell() -> tuple[str, str, list[str]]:
    raw = os.getenv("FORESTCODE_SHELL", "").strip()
    if raw:
        return _resolve_override(raw)
    if sys.platform == "win32":
        return ("PowerShell", "powershell.exe", ["-NoProfile", "-NonInteractive", "-Command"])
    shell_path = os.getenv("SHELL") or "/bin/sh"
    return (Path(shell_path).name, shell_path, ["-c"])


def build_argv(command: str) -> list[str]:
    _, executable, prefix_args = resolve_shell()
    return [executable, *prefix_args, command]


def describe_shell() -> str:
    return resolve_shell()[0]


def _resolve_override(raw: str) -> tuple[str, str, list[str]]:
    lower = raw.lower()
    if lower == "powershell":
        return ("PowerShell", "powershell.exe", ["-NoProfile", "-NonInteractive", "-Command"])
    if lower == "pwsh":
        return ("pwsh", "pwsh", ["-NoProfile", "-NonInteractive", "-Command"])
    if lower == "cmd":
        return ("cmd", "cmd.exe", ["/c"])
    if lower in ("bash", "sh", "zsh", "fish"):
        return (lower, lower, ["-c"])
    # Preserve the original casing of an explicit executable path, but resolve
    # the basename across platforms: POSIX Path.name does not split on "\", so a
    # Windows-style override would yield the whole string as the label.
    normalized = raw.replace("\\", "/")
    return (normalized.rsplit("/", 1)[-1], raw, ["-c"])
