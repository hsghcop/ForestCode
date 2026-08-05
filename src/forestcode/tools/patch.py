"""Patch proposal and application services."""

from __future__ import annotations

import difflib
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from ._path import canonical
from .types import PatchProposal, ReadStateStoreProtocol


def compute_content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def compute_diff(old: str, new: str, path: str) -> str:
    if old == new:
        return "(no changes)"
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join(diff)


class PatchService:
    def __init__(self, read_state_store: ReadStateStoreProtocol | None = None) -> None:
        self._read_state_store = read_state_store

    def propose(
        self,
        *,
        path: str,
        resolved_path: Path,
        operation: Literal["edit", "write", "create"],
        base_hash: str | None,
        diff: str,
        updated_content: str,
        tool_call_id: str,
    ) -> PatchProposal:
        return PatchProposal(
            id=f"patch_{uuid4().hex[:12]}",
            path=path,
            resolved_path=canonical(resolved_path),
            operation=operation,
            base_hash=base_hash,
            diff=diff,
            updated_content=updated_content,
            status="proposed",
            tool_call_id=tool_call_id,
            created_at=_now(),
        )

    def apply(self, proposal: PatchProposal) -> None:
        try:
            target = canonical(proposal.resolved_path)
            if proposal.operation == "create":
                if target.exists():
                    raise FileExistsError(f"File already exists: {proposal.path}")
            else:
                if proposal.base_hash is None:
                    raise ValueError("base_hash is required for edit/write patches")
                current = target.read_text(encoding="utf-8", errors="replace")
                current_hash = compute_content_hash(current)
                if current_hash != proposal.base_hash:
                    raise ValueError(f"File changed before patch apply: {proposal.path}")

            parent = target.parent
            parent.mkdir(parents=True, exist_ok=True)
            target.write_text(proposal.updated_content, encoding="utf-8")
            stat = target.stat()
            if self._read_state_store is not None:
                self._read_state_store.record(
                    target,
                    proposal.updated_content,
                    stat.st_mtime,
                    is_partial=False,
                    offset=0,
                    limit=len(proposal.updated_content),
                )
            proposal.status = "applied"
            proposal.applied_at = _now()
            proposal.error = None
        except Exception as exc:
            proposal.status = "failed"
            proposal.error = str(exc)
            raise

    def reject(self, proposal: PatchProposal) -> None:
        proposal.status = "rejected"


def _now() -> str:
    return datetime.now(UTC).isoformat()
