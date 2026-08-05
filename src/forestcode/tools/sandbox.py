"""Workspace-level path sandbox for file tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ._path import canonical
from .types import PathAccess


DEFAULT_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}

SENSITIVE_NAME_PARTS = (
    ".env",
    "credential",
    "credentials",
    "id_rsa",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "private_key",
    "secret",
    "token",
)


@dataclass(frozen=True, slots=True)
class PathCheck:
    access: PathAccess
    resolved_path: Path
    inside_workspace: bool
    sensitive: bool
    runtime_internal: bool


class WorkspaceSandbox:
    def __init__(
        self,
        workspace_root: str | Path,
        ignored_dirs: set[str] | None = None,
        sensitive_name_parts: tuple[str, ...] = SENSITIVE_NAME_PARTS,
        runtime_internal_dirs: frozenset[Path] = frozenset(),
        runtime_exception_dirs: frozenset[Path] = frozenset(),
    ) -> None:
        self.workspace_root = canonical(workspace_root)
        self.runtime_internal_dirs = frozenset(canonical(path) for path in runtime_internal_dirs)
        self.runtime_exception_dirs = frozenset(canonical(path) for path in runtime_exception_dirs)
        self.ignored_dirs = set(ignored_dirs or DEFAULT_IGNORED_DIRS)
        self.sensitive_name_parts = sensitive_name_parts

    def resolve_path(self, path: str | Path) -> Path:
        raw_path = Path(path).expanduser()
        if raw_path.is_absolute():
            return canonical(raw_path)
        return canonical(self.workspace_root / raw_path)

    def check_access(self, access: PathAccess) -> PathCheck:
        resolved_path = self.resolve_path(access.path)
        inside_workspace = self.is_inside_workspace(resolved_path)
        return PathCheck(
            access=access,
            resolved_path=resolved_path,
            inside_workspace=inside_workspace,
            sensitive=self.is_sensitive_path(resolved_path),
            runtime_internal=inside_workspace and self.is_runtime_internal_path(resolved_path),
        )

    def is_inside_workspace(self, path: Path) -> bool:
        try:
            canonical(path).relative_to(self.workspace_root)
        except ValueError:
            return False
        return True

    def is_sensitive_path(self, path: Path) -> bool:
        resolved = canonical(path)
        try:
            parts = resolved.relative_to(self.workspace_root).parts
        except ValueError:
            parts = resolved.parts
        lowered_parts = [part.lower() for part in parts]
        lowered_name = path.name.lower()
        if lowered_name == ".env" or lowered_name.startswith(".env."):
            return True
        return any(marker in part for part in lowered_parts for marker in self.sensitive_name_parts)

    def should_skip_dir(self, path: Path) -> bool:
        return path.name in self.ignored_dirs

    def is_runtime_internal_path(self, path: Path) -> bool:
        resolved = canonical(path)
        # Exception dirs take priority: paths inside them are not runtime internal.
        for exc in self.runtime_exception_dirs:
            try:
                resolved.relative_to(exc)
                return False
            except ValueError:
                continue
        for root in self.runtime_internal_dirs:
            try:
                resolved.relative_to(root)
            except ValueError:
                continue
            return True
        return False
