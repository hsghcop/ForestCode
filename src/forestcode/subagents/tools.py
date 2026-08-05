"""High-level subagent tools exposed only to the parent AgentLoop (design §Model Tool Contract).

Four tools wrap the per-run coordinator:

- ``delegate_task``   — validate agent/description/prompt and the 24,000-char
  combined context budget, enqueue, return the coordinator-generated task id.
- ``wait_subagents``  — bounded wait; the *first* wait that delivers a terminal
  completed/failed body returns it (this result is a normal persistent tool
  result and enters the parent JSONL); repeated waits report
  ``result_omitted`` without repeating the body.
- ``list_subagents``  — current snapshots of this run, no prompts/instructions.
- ``cancel_subagent`` — idempotent cancel; terminal tasks return current state.

``delegate/list/cancel`` set ``persist_result=False`` so the ToolExecutor marks
their results ``state_only``/``transient`` and the SessionRecorder skips them;
``wait_subagents`` keeps the default persistent contract (R10).
"""

from __future__ import annotations

import json
from typing import Any

from forestcode.context.types import ContextFragment
from forestcode.skills.types import SkillSnapshot
from forestcode.tools.types import ToolContext, ToolDefinition

from .child import combined_context_chars, resolve_child_skill_fragments
from .config_loader import format_agent_catalog
from .coordinator import SubagentCoordinator, WaitOutcome
from .types import (
    MAX_COMBINED_CONTEXT_CHARS,
    MAX_DESCRIPTION_CHARS,
    MAX_PROMPT_CHARS,
    AgentConfigSet,
    SubagentRequest,
    SubagentTaskSnapshot,
)

DELEGATE_TASK_DESCRIPTION = (
    "Delegate an independent sub-task to a configured subagent. Returns a task "
    "id and the current status (queued or running). Use wait_subagents to "
    "collect the result before finalizing."
)
WAIT_SUBAGENTS_DESCRIPTION = (
    "Wait (bounded) for one or more delegated subagents to reach a terminal "
    "state. Returns snapshots plus, for tasks completing/failing during this "
    "wait, their final text. A task already delivered in an earlier wait is "
    "reported with result_omitted=true and no body."
)
LIST_SUBAGENTS_DESCRIPTION = (
    "List all subagent tasks of the current run with their current status. "
    "Never returns prompts or instructions."
)
CANCEL_SUBAGENT_DESCRIPTION = (
    "Cancel one delegated subagent. Queued tasks stop immediately; running "
    "tasks enter cancelling and their late results are discarded. Idempotent."
)


def _snapshot_to_dict(snapshot: SubagentTaskSnapshot) -> dict[str, Any]:
    """Neutral structured state; no prompts, instructions, keys or paths."""
    return {
        "task_id": snapshot.task_id,
        "agent_name": snapshot.agent_name,
        "status": snapshot.status,
        "queue_position": snapshot.queue_position,
        "summary": snapshot.summary,
        "error": snapshot.error,
        "cancel_reason": snapshot.cancel_reason,
        "delivered": snapshot.delivered,
    }


def _wait_outcome_to_text(outcome: WaitOutcome) -> str:
    tasks = [_snapshot_to_dict(snapshot) for snapshot in outcome.snapshots]
    results: dict[str, dict[str, Any]] = {}
    omitted: list[str] = []
    for task_id, result in outcome.results.items():
        results[task_id] = {
            "final_text": result.final_text,
            "turn_count": result.turn_count,
            "tool_count": result.tool_count,
        }
    # Snapshots whose delivered flag was already set in a previous wait (or
    # that have no body) carry no result entry: report them explicitly so the
    # model never mistakes absence for an error.
    for snapshot in outcome.snapshots:
        if snapshot.delivered and snapshot.task_id not in results:
            omitted.append(snapshot.task_id)
    return json.dumps(
        {
            "timed_out": outcome.timed_out,
            "tasks": tasks,
            "results": results,
            "result_omitted": omitted,
        },
        ensure_ascii=False,
    )


def _validate_agent(agent_set: AgentConfigSet | None, name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("agent must be a non-empty string")
    name = name.strip()
    if agent_set is None or agent_set.get(name) is None:
        raise ValueError(f"Unknown subagent: {name}")
    return name


def _validate_description(description: Any) -> str:
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    description = description.strip()
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValueError(
            f"description must be at most {MAX_DESCRIPTION_CHARS} characters"
        )
    return description


def _validate_prompt(prompt: Any) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    prompt = prompt.strip()
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt must be at most {MAX_PROMPT_CHARS} characters")
    return prompt


def _run_delegate_task(
    _context: ToolContext,
    coordinator: SubagentCoordinator,
    *,
    agent_set: AgentConfigSet | None,
    skills_snapshot: SkillSnapshot | None,
    activated_skill_names: tuple[str, ...],
    inherited_fragments: tuple[ContextFragment, ...],
    agent: str,
    description: str,
    prompt: str,
) -> str:
    name = _validate_agent(agent_set, agent)
    description = _validate_description(description)
    prompt = _validate_prompt(prompt)
    if agent_set is None:
        raise ValueError("no subagents are configured for this run")
    config = agent_set.get(name)
    if config is None:  # unreachable after _validate_agent; defensive
        raise ValueError(f"Unknown subagent: {name}")
    # Design §Child Construction and Context: the combined budget is checked
    # *before* enqueue and never truncates. Missing/invalid default skills make
    # this delegation fail with a diagnostic instead of silently dropping.
    fragments = resolve_child_skill_fragments(
        config,
        skills_snapshot,
        activated_skill_names,
        inherited_fragments,
    )
    total = combined_context_chars(config.instructions, prompt, fragments)
    if total > MAX_COMBINED_CONTEXT_CHARS:
        raise ValueError(
            "delegation context too large: agent instructions + prompt + "
            f"pre-injected skills total {total} characters, exceeding the "
            f"{MAX_COMBINED_CONTEXT_CHARS}-character limit"
        )
    snapshot = coordinator.delegate(
        SubagentRequest(
            task_id="", agent_name=name, description=description, prompt=prompt
        ),
        timeout_seconds=config.task_timeout_seconds,
    )
    return json.dumps(
        {"task_id": snapshot.task_id, "status": snapshot.status},
        ensure_ascii=False,
    )


def _validate_task_ids(task_ids: Any) -> list[str] | None:
    if task_ids is None:
        return None
    if not isinstance(task_ids, list) or not task_ids:
        raise ValueError("task_ids must be a non-empty array")
    if not all(isinstance(tid, str) and tid.strip() for tid in task_ids):
        raise ValueError("task_ids must contain non-empty strings")
    return [tid.strip() for tid in task_ids]


def _validate_timeout_ms(timeout_ms: Any) -> int:
    if timeout_ms is None:
        return 30_000
    if not isinstance(timeout_ms, int) or isinstance(timeout_ms, bool):
        raise TypeError("timeout_ms must be an integer")
    if not 0 <= timeout_ms <= 60_000:
        raise ValueError("timeout_ms must be within [0, 60000]")
    return timeout_ms


def _run_wait_subagents(
    context: ToolContext,
    coordinator: SubagentCoordinator,
    *,
    task_ids: list[str] | None,
    timeout_ms: int,
) -> str:
    task_ids = _validate_task_ids(task_ids)
    timeout_ms = _validate_timeout_ms(timeout_ms)
    # Design §Model Tool Contract: waiting on a run with no tasks is a tool
    # error, not a silent empty outcome — the model must know it delegated
    # nothing (distinguishable from a bounded wait that simply timed out).
    if task_ids is None and not coordinator.list():
        raise ValueError("no subagent tasks in the current run")
    outcome = coordinator.wait(
        task_ids,
        timeout=timeout_ms / 1000.0,
        abort=getattr(context, "abort", None),
    )
    return _wait_outcome_to_text(outcome)


def _run_list_subagents(_context: ToolContext, coordinator: SubagentCoordinator) -> str:
    snapshots = coordinator.list()
    return json.dumps(
        {"tasks": [_snapshot_to_dict(snapshot) for snapshot in snapshots]},
        ensure_ascii=False,
    )


def _run_cancel_subagent(
    _context: ToolContext, coordinator: SubagentCoordinator, *, task_id: str
) -> str:
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError("task_id must be a non-empty string")
    task_id = task_id.strip()
    snapshot = coordinator.cancel(task_id)
    if snapshot is None:
        raise ValueError(f"Unknown task id: {task_id}")
    return json.dumps(_snapshot_to_dict(snapshot), ensure_ascii=False)


def create_subagent_tools(
    coordinator: SubagentCoordinator,
    *,
    agent_set: AgentConfigSet | None,
    skills_snapshot: SkillSnapshot | None,
    activated_skill_names: tuple[str, ...] = (),
    inherited_fragments: tuple[ContextFragment, ...] = (),
) -> list[ToolDefinition]:
    """Build the four delegation tools bound to one per-run coordinator.

    Only the parent catalog registers these; the child catalog structurally
    removes the names via ``SUBAGENT_TOOLS`` (design §Permission Composition).
    ``agent_set``/``skills_snapshot`` are the run-fixed snapshots; they are
    captured here so every tool call sees the same view.
    """
    catalog = format_agent_catalog(agent_set)
    delegate_description = (
        f"{DELEGATE_TASK_DESCRIPTION}\n\nAvailable subagents for this run:\n{catalog}"
    )
    agent_names = list(agent_set.agents) if agent_set is not None else []
    return [
        ToolDefinition(
            name="delegate_task",
            description=delegate_description,
            input_schema={
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "description": "Configured subagent name",
                        "enum": agent_names,
                    },
                    "description": {
                        "type": "string",
                        "description": "Short scope note for this task",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Self-contained task prompt",
                    },
                },
                "required": ["agent", "description", "prompt"],
            },
            validator=lambda args: {
                "agent": args["agent"],
                "description": args["description"],
                "prompt": args["prompt"],
            },
            runner=lambda context, **kwargs: _run_delegate_task(
                context,
                coordinator,
                agent_set=agent_set,
                skills_snapshot=skills_snapshot,
                activated_skill_names=activated_skill_names,
                inherited_fragments=inherited_fragments,
                agent=kwargs["agent"],
                description=kwargs["description"],
                prompt=kwargs["prompt"],
            ),
            risk_level="read_only",
            is_read_only=True,
            persist_result=False,
        ),
        ToolDefinition(
            name="wait_subagents",
            description=WAIT_SUBAGENTS_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {
                    "task_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task ids to wait for; omit to wait for all undelivered",
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 60000,
                        "default": 30000,
                        "description": "Bounded wait in ms; 0 returns an immediate snapshot",
                    },
                },
            },
            validator=lambda args: {
                "task_ids": args.get("task_ids"),
                "timeout_ms": args.get("timeout_ms", 30_000),
            },
            runner=lambda context, **kwargs: _run_wait_subagents(
                context,
                coordinator,
                task_ids=kwargs.get("task_ids"),
                timeout_ms=kwargs.get("timeout_ms", 30_000),
            ),
            risk_level="read_only",
            is_read_only=True,
            persist_result=True,
        ),
        ToolDefinition(
            name="list_subagents",
            description=LIST_SUBAGENTS_DESCRIPTION,
            input_schema={"type": "object", "properties": {}},
            runner=lambda context, **kwargs: _run_list_subagents(context, coordinator),
            risk_level="read_only",
            is_read_only=True,
            persist_result=False,
        ),
        ToolDefinition(
            name="cancel_subagent",
            description=CANCEL_SUBAGENT_DESCRIPTION,
            input_schema={
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"],
            },
            validator=lambda args: {"task_id": args["task_id"]},
            runner=lambda context, **kwargs: _run_cancel_subagent(
                context, coordinator, task_id=kwargs["task_id"]
            ),
            risk_level="read_only",
            is_read_only=True,
            persist_result=False,
        ),
    ]
