"""JSONL-backed append-only session memory store."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .types import MemoryEntry, SessionMemory


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SessionMeta:
    session_id: str
    title: str
    created_at: str
    updated_at: str
    entry_count: int | None = None
    is_legacy: bool = False


class SessionStore:
    CURRENT_VERSION = 2
    ENTRY_ID_PREFIX = "entry_"
    ENTRY_ID_WIDTH = 6

    def __init__(self, workspace_root: str | Path, session_dir: str | Path | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.session_dir = (
            Path(session_dir).resolve()
            if session_dir is not None
            else self.workspace_root / ".forestcode" / "sessions"
        )
        self._runtime_root = self.session_dir.parent
        self._entry_counters: dict[str, int] = {}

    @property
    def runtime_root(self) -> Path:
        """Private runtime root used by session storage."""
        return self._runtime_root

    def load(self, session_id: str = "default") -> SessionMemory:
        self._ensure_migrated(session_id)
        path = self._jsonl_path(session_id)
        if not path.exists() or path.stat().st_size == 0:
            return self._new_memory(session_id)

        header: dict[str, Any] = {}
        entries: list[MemoryEntry] = []
        runs: list[dict[str, Any]] = []
        last_plan: list[dict[str, Any]] = []
        last_meta: dict[str, Any] = {}

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("session %s: skipping malformed line", session_id)
                continue
            if not isinstance(record, dict):
                logger.warning("session %s: skipping non-object line", session_id)
                continue

            record_type = record.get("_t")
            if record_type == "header":
                header = record
            elif record_type == "entry":
                entries.append(self._record_to_entry(record))
            elif record_type == "run":
                runs.append({key: value for key, value in record.items() if key != "_t"})
            elif record_type == "plan":
                last_plan = list(record.get("items", []))
            elif record_type == "meta":
                last_meta.update({key: value for key, value in record.items() if key != "_t"})

        created_at = str(header.get("created_at", "") or "")
        updated_at = str(last_meta.get("updated_at", "") or created_at)
        return SessionMemory(
            session_id=str(header.get("session_id", session_id) or session_id),
            version=int(header.get("version", self.CURRENT_VERSION) or self.CURRENT_VERSION),
            title=str(last_meta.get("title", "") or ""),
            created_at=created_at,
            updated_at=updated_at,
            workspace_root=str(header.get("workspace_root", str(self.workspace_root)) or str(self.workspace_root)),
            entries=entries,
            runs=runs,
            plan=last_plan,
        )

    def save(self, memory: SessionMemory) -> None:
        self._write_full_jsonl(memory)

    def append_entry(self, session_id: str, entry: MemoryEntry) -> None:
        self._ensure_migrated(session_id)
        self._ensure_header(session_id)
        if not entry.id:
            entry.id = self._next_entry_id(session_id)
        record = {"_t": "entry", **self._entry_to_record(entry)}
        self._append_record(session_id, record)

    def append_run(self, session_id: str, run: dict[str, Any]) -> None:
        self._ensure_migrated(session_id)
        self._ensure_header(session_id)
        now = self._now()
        record = dict(run)
        record.setdefault("timestamp", now)
        self._append_record(session_id, {"_t": "run", **record})
        self._append_record(session_id, {"_t": "meta", "updated_at": now})

    def save_plan(self, session_id: str, plan: list[dict[str, Any]]) -> None:
        self._ensure_migrated(session_id)
        self._ensure_header(session_id)
        self._append_record(session_id, {"_t": "plan", "items": list(plan)})

    def update_meta(self, session_id: str, *, title: str | None = None) -> None:
        self._ensure_migrated(session_id)
        self._ensure_header(session_id)
        record: dict[str, Any] = {"_t": "meta", "updated_at": self._now()}
        if title is not None:
            record["title"] = title
        self._append_record(session_id, record)

    def read_meta(self, session_id: str) -> SessionMeta | None:
        self._ensure_migrated(session_id)
        path = self._jsonl_path(session_id)
        if not path.exists() or path.stat().st_size == 0:
            return None

        header: dict[str, Any] = {}
        last_meta: dict[str, Any] = {}
        entry_count = 0
        with path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("session %s: skipping malformed line", session_id)
                    continue
                if not isinstance(record, dict):
                    logger.warning("session %s: skipping non-object line", session_id)
                    continue

                record_type = record.get("_t")
                if record_type == "header":
                    header = record
                elif record_type == "entry":
                    entry_count += 1
                elif record_type == "meta":
                    last_meta.update({key: value for key, value in record.items() if key != "_t"})

        created_at = str(header.get("created_at", "") or "")
        updated_at = str(last_meta.get("updated_at", "") or created_at)
        return SessionMeta(
            session_id=str(header.get("session_id", session_id) or session_id),
            title=str(last_meta.get("title", "") or ""),
            created_at=created_at,
            updated_at=updated_at,
            entry_count=entry_count,
        )

    def validate_session_id(self, session_id: str) -> None:
        """Public safe-file-name check. Raises ValueError when invalid."""
        self._validate_session_id(session_id)

    def list_sessions(self) -> list[SessionMeta]:
        """All sessions as metadata, sorted by id, including unmigrated legacy .json.

        Legacy entries carry ``is_legacy=True``, ``entry_count=None`` and a
        ``"(legacy)"`` title; their ``updated_at`` uses the file mtime. This is
        the public replacement for callers reaching into ``session_dir`` and the
        ``_jsonl_path`` / ``_legacy_json_path`` helpers.
        """
        if not self.session_dir.exists():
            return []

        jsonl_ids = sorted(path.stem for path in self.session_dir.glob("*.jsonl"))
        jsonl_id_set = set(jsonl_ids)
        metas = [meta for session_id in jsonl_ids if (meta := self.read_meta(session_id)) is not None]

        for path in sorted(self.session_dir.glob("*.json")):
            if path.stem in jsonl_id_set:
                continue
            updated_at = datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat()
            metas.append(
                SessionMeta(
                    session_id=path.stem,
                    title="(legacy)",
                    created_at="",
                    updated_at=updated_at,
                    entry_count=None,
                    is_legacy=True,
                )
            )
        return metas

    def delete_session(self, session_id: str) -> int:
        """Delete a session and all derived files, returning the count removed.

        Removes the JSONL store, its ``.jsonl.tmp`` sibling, the legacy ``.json``
        file, and any ``.json.bak*`` backups. Returns 0 when nothing exists.
        Caller is responsible for refusing to delete the current session.
        """
        self._validate_session_id(session_id)
        jsonl = self._jsonl_path(session_id)
        legacy = self._legacy_json_path(session_id)
        paths = [
            jsonl,
            jsonl.with_suffix(".jsonl.tmp"),
            legacy,
            *legacy.parent.glob(f"{legacy.name}.bak*"),
        ]
        deleted = 0
        for path in paths:
            if path.exists():
                path.unlink()
                deleted += 1
        self._entry_counters.pop(session_id, None)
        return deleted

    def _new_memory(self, session_id: str) -> SessionMemory:
        now = self._now()
        return SessionMemory(
            session_id=session_id,
            version=self.CURRENT_VERSION,
            created_at=now,
            updated_at=now,
            workspace_root=str(self.workspace_root),
        )

    def _append_record(self, session_id: str, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._jsonl_path(session_id).open("a", encoding="utf-8") as file:
            file.write(line)
            file.flush()

    def _write_full_jsonl(self, memory: SessionMemory) -> None:
        path = self._jsonl_path(memory.session_id)
        tmp = path.with_suffix(".jsonl.tmp")
        self._render_jsonl(memory, tmp, refresh_timestamps=True)
        tmp.replace(path)
        self._entry_counters.pop(memory.session_id, None)

    def _render_jsonl(self, memory: SessionMemory, path: Path, *, refresh_timestamps: bool) -> None:
        self._ensure_entry_ids(memory)
        memory.version = self.CURRENT_VERSION

        now = self._now()
        if refresh_timestamps:
            memory.updated_at = now
        if not memory.updated_at:
            memory.updated_at = memory.created_at or now
        if not memory.created_at:
            memory.created_at = memory.updated_at
        if not memory.workspace_root:
            memory.workspace_root = str(self.workspace_root)
        if memory.title is None:
            memory.title = ""

        header = {
            "_t": "header",
            "session_id": memory.session_id,
            "version": self.CURRENT_VERSION,
            "created_at": memory.created_at,
            "workspace_root": memory.workspace_root,
        }
        lines = [json.dumps(header, ensure_ascii=False)]
        for entry in memory.entries:
            lines.append(json.dumps({"_t": "entry", **self._entry_to_record(entry)}, ensure_ascii=False))
        for run in memory.runs:
            lines.append(json.dumps({"_t": "run", **run}, ensure_ascii=False))
        if memory.plan:
            lines.append(json.dumps({"_t": "plan", "items": memory.plan}, ensure_ascii=False))
        lines.append(
            json.dumps(
                {"_t": "meta", "title": memory.title, "updated_at": memory.updated_at},
                ensure_ascii=False,
            )
        )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _ensure_header(self, session_id: str) -> None:
        path = self._jsonl_path(session_id)
        if path.exists() and path.stat().st_size > 0:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        now = self._now()
        header = {
            "_t": "header",
            "session_id": session_id,
            "version": self.CURRENT_VERSION,
            "created_at": now,
            "workspace_root": str(self.workspace_root),
        }
        path.write_text(json.dumps(header, ensure_ascii=False) + "\n", encoding="utf-8")

    def _ensure_migrated(self, session_id: str) -> None:
        jsonl = self._jsonl_path(session_id)
        legacy = self._legacy_json_path(session_id)
        jsonl_empty = (not jsonl.exists()) or jsonl.stat().st_size == 0
        if jsonl_empty and legacy.exists():
            self._migrate_json_to_jsonl(session_id, legacy, jsonl)

    def _migrate_json_to_jsonl(self, session_id: str, src: Path, dst: Path) -> None:
        memory = self._load_json_legacy(session_id, src)
        tmp = dst.with_suffix(".jsonl.tmp")
        self._render_jsonl(memory, tmp, refresh_timestamps=False)
        tmp.replace(dst)
        src.rename(self._unique_bak_path(src))
        self._entry_counters.pop(session_id, None)

    def _load_json_legacy(self, session_id: str, path: Path) -> SessionMemory:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("legacy session file must contain an object")

        raw_entries = list(raw.get("entries", []))
        entries = [
            MemoryEntry(**self._migrate_entry(entry, index))
            for index, entry in enumerate(raw_entries, start=1)
        ]
        created_at = str(raw.get("created_at", "") or "")
        updated_at = str(raw.get("updated_at", "") or created_at)
        now = self._now()
        if not created_at:
            created_at = updated_at or now
        if not updated_at:
            updated_at = created_at

        return SessionMemory(
            session_id=str(raw.get("session_id", session_id) or session_id),
            version=self.CURRENT_VERSION,
            title=str(raw.get("title", "") or ""),
            created_at=created_at,
            updated_at=updated_at,
            workspace_root=str(raw.get("workspace_root", str(self.workspace_root)) or str(self.workspace_root)),
            entries=entries,
            runs=list(raw.get("runs", [])),
            plan=list(raw.get("plan", [])),
        )

    @staticmethod
    def _unique_bak_path(path: Path) -> Path:
        candidate = Path(str(path) + ".bak")
        suffix = 1
        while candidate.exists():
            candidate = Path(str(path) + f".bak.{suffix}")
            suffix += 1
        return candidate

    def _path_for(self, session_id: str) -> Path:
        return self._jsonl_path(session_id)

    def _jsonl_path(self, session_id: str) -> Path:
        self._validate_session_id(session_id)
        return self.session_dir / f"{session_id}.jsonl"

    def _legacy_json_path(self, session_id: str) -> Path:
        self._validate_session_id(session_id)
        return self.session_dir / f"{session_id}.json"

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if (
            not session_id
            or session_id in {".", ".."}
            or ".." in session_id
            or any(char in session_id for char in "\\/:*?\"<>|")
        ):
            raise ValueError("session_id must be a safe file name")

    @classmethod
    def _entry_id(cls, number: int) -> str:
        return f"{cls.ENTRY_ID_PREFIX}{number:0{cls.ENTRY_ID_WIDTH}d}"

    @classmethod
    def _entry_number(cls, entry_id: str) -> int | None:
        if not entry_id.startswith(cls.ENTRY_ID_PREFIX):
            return None
        suffix = entry_id.removeprefix(cls.ENTRY_ID_PREFIX)
        if not suffix.isdigit():
            return None
        return int(suffix)

    def _migrate_entry(self, raw_entry: Any, index: int | None = None) -> dict[str, Any]:
        if not isinstance(raw_entry, dict):
            raise ValueError("session entry must be an object")
        migrated = dict(raw_entry)
        if not migrated.get("id") and index is not None:
            migrated["id"] = self._entry_id(index)
        migrated.setdefault("metadata", {})
        if migrated["metadata"] is None:
            migrated["metadata"] = {}
        return migrated

    def _entry_to_record(self, entry: MemoryEntry) -> dict[str, Any]:
        return asdict(entry)

    def _record_to_entry(self, record: dict[str, Any]) -> MemoryEntry:
        data = {key: value for key, value in record.items() if key != "_t"}
        return MemoryEntry(**self._migrate_entry(data))

    def _ensure_entry_ids(self, memory: SessionMemory) -> None:
        used: set[str] = set()
        next_number = 1
        for entry in memory.entries:
            if entry.id and entry.id not in used:
                used.add(entry.id)
                number = self._entry_number(entry.id)
                if number is not None:
                    next_number = max(next_number, number + 1)
                continue
            while self._entry_id(next_number) in used:
                next_number += 1
            entry.id = self._entry_id(next_number)
            used.add(entry.id)
            next_number += 1

    def _next_entry_id(self, session_id: str) -> str:
        if session_id not in self._entry_counters:
            self._entry_counters[session_id] = self._scan_max_entry_number(session_id) + 1
        number = self._entry_counters[session_id]
        self._entry_counters[session_id] = number + 1
        return self._entry_id(number)

    def _scan_max_entry_number(self, session_id: str) -> int:
        path = self._jsonl_path(session_id)
        if not path.exists():
            return 0
        max_number = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict) or record.get("_t") != "entry":
                continue
            entry_id = record.get("id")
            if not isinstance(entry_id, str):
                continue
            number = self._entry_number(entry_id)
            if number is not None:
                max_number = max(max_number, number)
        return max_number

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()
