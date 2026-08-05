"""Child transcript paths for isolated subagent sessions (design §Persistence).

- ``child_transcript_dir`` resolves the directory each child transcript lives
  in: ``.forestcode/subagents/<parent-session>/sessions/``. Step 4 gives every
  child its own ``SessionStore`` rooted there so concurrent children never
  append to the parent JSONL. Note that child *tools* must still treat the
  whole workspace ``.forestcode/`` as a runtime-internal directory — this
  helper only computes the canonical path.
"""

from __future__ import annotations

from pathlib import Path

# Mirrors SessionStore's safe-file-name rule so a parent session id can never
# smuggle path separators into the child transcript root (R10).
_UNSAFE_SESSION_ID_CHARS = '\\/:*?"<>|'


def _validate_parent_session_id(parent_session_id: str) -> None:
    if (
        not parent_session_id
        or parent_session_id in {".", ".."}
        or ".." in parent_session_id
        or any(char in parent_session_id for char in _UNSAFE_SESSION_ID_CHARS)
    ):
        raise ValueError("parent_session_id must be a safe file name")


def child_transcript_dir(workspace_root: Path, parent_session_id: str) -> Path:
    """Canonical child transcript root for one parent session.

    Returns ``<workspace>/.forestcode/subagents/<parent-session>/sessions/``.
    Raises ``ValueError`` for an unsafe session id so untrusted text can never
    become a filesystem path.
    """
    _validate_parent_session_id(parent_session_id)
    return (
        Path(workspace_root)
        / ".forestcode"
        / "subagents"
        / parent_session_id
        / "sessions"
    )
