"""Confirm-gated shell command tool."""

from __future__ import annotations

import contextlib
import locale
import os
import re
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .shell import build_argv, describe_shell
from .types import CommandProposal, ToolContext, ToolDefinition

if TYPE_CHECKING:
    from forestcode.core.abort import AbortSignal


_DEFAULT_TIMEOUT_SECONDS = 30
_MAX_TIMEOUT_SECONDS = 300
_STDOUT_MAX_LINES = 200
_STDOUT_MAX_BYTES = 50 * 1024
_STDERR_MAX_LINES = 50
_STDERR_MAX_BYTES = 10 * 1024

_DANGEROUS_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"rm\s+.*-[^\s]*r",
        r"\brm\b.*--recursive",
        r"\bsudo\b",
        r"\bformat\b",
        r"\bmkfs\b",
        r"\bdd\s+if=",
        r"\bdel\s+.*/[sfq]",
        r"Remove-Item.*-Recurse",
        r"\bpoweroff\b|\breboot\b|\bshutdown\b",
    )
]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_dangerous(command: str) -> bool:
    return any(pattern.search(command) for pattern in _DANGEROUS_PATTERNS)


def _truncate_tail(text: str, max_lines: int, max_bytes: int) -> str:
    if not text:
        return ""

    omitted_lines = 0
    lines = text.splitlines()
    if len(lines) > max_lines:
        omitted_lines = len(lines) - max_lines
        text = "\n".join(lines[-max_lines:])

    omitted_bytes = 0
    encoded = text.encode("utf-8")
    if len(encoded) > max_bytes:
        omitted_bytes = len(encoded) - max_bytes
        tail = encoded[-max_bytes:]
        while tail:
            try:
                text = tail.decode("utf-8")
                break
            except UnicodeDecodeError:
                tail = tail[1:]
        else:
            text = ""

    prefixes: list[str] = []
    if omitted_lines:
        prefixes.append(f"[... {omitted_lines} lines omitted ...]")
    if omitted_bytes:
        prefixes.append(f"[... {omitted_bytes} bytes omitted ...]")
    if prefixes:
        return "\n".join([*prefixes, text])
    return text


def _candidate_encodings() -> list[str]:
    encodings = ["utf-8", locale.getpreferredencoding(False)]
    result: list[str] = []
    for encoding in encodings:
        if encoding and encoding not in result:
            result.append(encoding)
    return result


def decode_output(raw: bytes) -> str:
    if not raw:
        return ""
    for encoding in _candidate_encodings():
        try:
            return raw.decode(encoding)
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


def _build_display(command: str, cwd: Path, shell_label: str, timeout: int, is_dangerous: bool) -> str:
    lines: list[str] = []
    if is_dangerous:
        lines.append("⚠️ Dangerous command pattern detected. Review carefully before approving.")
    lines.extend(
        [
            f"Command: {command}",
            f"Cwd: {cwd}",
            f"Shell: {shell_label}",
            f"Timeout: {timeout}s",
        ]
    )
    return "\n".join(lines)


def build_command_content(proposal: CommandProposal) -> str:
    stdout = proposal.stdout or ""
    stderr = proposal.stderr or ""
    parts = [stdout if stdout else "(no output)"]
    if stderr:
        parts.extend(["[stderr]", stderr])
    if proposal.exit_code is not None and proposal.exit_code != 0:
        parts.append(f"[exit code: {proposal.exit_code}]")
    if proposal.status == "timeout":
        parts.append(f"[timed out after {proposal.timeout}s]")
    return "\n".join(parts)


def _validate_run_command(arguments: dict[str, Any]) -> dict[str, Any]:
    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError("command must be a non-empty string")

    timeout = arguments.get("timeout", _DEFAULT_TIMEOUT_SECONDS)
    if timeout is None:
        timeout = _DEFAULT_TIMEOUT_SECONDS
    if isinstance(timeout, bool) or not isinstance(timeout, int):
        raise ValueError("timeout must be an integer")
    if timeout <= 0:
        raise ValueError("timeout must be > 0")
    if timeout > _MAX_TIMEOUT_SECONDS:
        timeout = _MAX_TIMEOUT_SECONDS
    return {"command": command, "timeout": timeout}


class CommandService:
    def execute(self, proposal: CommandProposal, *, abort: "AbortSignal | None" = None) -> None:
        # abort=None keeps the original blocking path byte-for-byte (zero
        # regression, plan §11-B3). With a token, use Popen + process-tree kill
        # so a Ctrl+C cancellation terminates the child (best-effort, §7.3/§17).
        if abort is None:
            self._run_blocking(proposal)
        else:
            self._run_cancellable(proposal, abort)

    def _run_blocking(self, proposal: CommandProposal) -> None:
        argv = build_argv(proposal.command)
        try:
            result = subprocess.run(
                argv,
                cwd=proposal.cwd,
                timeout=proposal.timeout,
                capture_output=True,
            )
            proposal.stdout = _truncate_tail(decode_output(result.stdout), _STDOUT_MAX_LINES, _STDOUT_MAX_BYTES)
            proposal.stderr = _truncate_tail(decode_output(result.stderr), _STDERR_MAX_LINES, _STDERR_MAX_BYTES)
            proposal.exit_code = result.returncode
            proposal.status = "executed"
            proposal.executed_at = _now()
        except subprocess.TimeoutExpired as exc:
            proposal.stdout = _truncate_tail(decode_output(exc.stdout or b""), _STDOUT_MAX_LINES, _STDOUT_MAX_BYTES)
            proposal.stderr = _truncate_tail(decode_output(exc.stderr or b""), _STDERR_MAX_LINES, _STDERR_MAX_BYTES)
            proposal.status = "timeout"
            proposal.error = f"Command timed out after {proposal.timeout}s"
            raise
        except Exception as exc:
            proposal.status = "failed"
            proposal.error = str(exc)
            raise

    def _run_cancellable(self, proposal: CommandProposal, abort: "AbortSignal") -> None:
        from forestcode.core.abort import Aborted

        argv = build_argv(proposal.command)
        creationflags = 0
        start_new_session = False
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            start_new_session = True  # own process group for tree kill (§17)
        try:
            proc = subprocess.Popen(
                argv,
                cwd=proposal.cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
        except Exception as exc:
            proposal.status = "failed"
            proposal.error = str(exc)
            raise
        # Set on the main thread; this callback kills the tree so the worker's
        # communicate() unblocks (plan §7.2/§7.3).
        abort.on_abort(lambda: self._kill_tree(proc))
        try:
            out, err = proc.communicate(timeout=proposal.timeout)
        except subprocess.TimeoutExpired:
            self._kill_tree(proc)
            out, err = proc.communicate()
            proposal.stdout = _truncate_tail(decode_output(out or b""), _STDOUT_MAX_LINES, _STDOUT_MAX_BYTES)
            proposal.stderr = _truncate_tail(decode_output(err or b""), _STDERR_MAX_LINES, _STDERR_MAX_BYTES)
            proposal.status = "timeout"
            proposal.error = f"Command timed out after {proposal.timeout}s"
            raise subprocess.TimeoutExpired(cmd=proposal.command, timeout=proposal.timeout)
        if abort.is_set():
            # Killed by cancellation: raise Aborted rather than reporting a
            # normal command result (plan §11-B3).
            raise Aborted()
        proposal.stdout = _truncate_tail(decode_output(out), _STDOUT_MAX_LINES, _STDOUT_MAX_BYTES)
        proposal.stderr = _truncate_tail(decode_output(err), _STDERR_MAX_LINES, _STDERR_MAX_BYTES)
        proposal.exit_code = proc.returncode
        proposal.status = "executed"
        proposal.executed_at = _now()

    @staticmethod
    def _kill_tree(proc) -> None:
        """Best-effort process-tree termination (§17 Windows boundary)."""
        if proc.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=5,
                )
            else:
                # The child is launched with start_new_session=True, so it is
                # the leader of its own process group (pgid == pid). Kill the
                # whole group: terminate() alone would only kill the shell, and
                # a shell like fish (which does not exec the last command) would
                # leave the real child alive holding our pipes, so the worker's
                # communicate() would hang until that child exits.
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:  # noqa: BLE001 - fall back to process-group kill
            with contextlib.suppress(Exception):
                os.killpg(proc.pid, signal.SIGKILL)

    def reject(self, proposal: CommandProposal) -> None:
        proposal.status = "rejected"


def _command_proposer(context: ToolContext, arguments: dict[str, Any], tool_call_id: str) -> CommandProposal:
    command = arguments["command"]
    timeout = arguments["timeout"]
    shell_label = describe_shell()
    is_dangerous = _is_dangerous(command)
    return CommandProposal(
        id=uuid4().hex[:8],
        command=command,
        cwd=context.workspace_root,
        timeout=timeout,
        shell_label=shell_label,
        is_dangerous=is_dangerous,
        display=_build_display(command, context.workspace_root, shell_label, timeout, is_dangerous),
        status="proposed",
        tool_call_id=tool_call_id,
        created_at=_now(),
    )


def _defensive_runner(_context: ToolContext, **_arguments: Any) -> str:
    raise RuntimeError("run_command must go through command_proposer")


def create_run_command_tool() -> ToolDefinition:
    return ToolDefinition(
        name="run_command",
        description="Run a shell command in the workspace root directory.",
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 30, "minimum": 1, "maximum": 300},
            },
            "required": ["command"],
        },
        runner=_defensive_runner,
        risk_level="command",
        is_read_only=False,
        validator=_validate_run_command,
        command_proposer=_command_proposer,
    )
