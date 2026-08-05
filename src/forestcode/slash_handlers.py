"""Built-in slash command handlers."""

from __future__ import annotations

from forestcode.memory import MarkdownMemory, ModelSummarizer, SessionCompressor

from .slash_commands import (
    SlashCommand,
    SlashCommandRegistry,
    SlashContext,
    SlashResult,
)


def _handle_exit(_ctx: SlashContext, _args: str) -> SlashResult:
    return SlashResult(action="exit")


def _handle_sessions(ctx: SlashContext, _args: str) -> SlashResult:
    metas = ctx.session_store.list_sessions()
    if not metas:
        print("Sessions> no sessions yet", file=ctx.stdout)
        return SlashResult()

    print("Sessions>", file=ctx.stdout)
    print("id\tentries\tupdated_at\ttitle", file=ctx.stdout)
    for meta in metas:
        entries = "-" if meta.entry_count is None else str(meta.entry_count)
        title = meta.title or "-"
        print(
            f"{meta.session_id}\t{entries}\t{meta.updated_at}\t{title}", file=ctx.stdout
        )
    return SlashResult()


def _handle_switch(ctx: SlashContext, args: str) -> SlashResult:
    new_id = args.strip()
    if not new_id:
        print("Usage: /switch <session-id>", file=ctx.stderr)
        return SlashResult()
    try:
        ctx.session_store.validate_session_id(new_id)
    except ValueError as exc:
        print(f"Invalid session id: {exc}", file=ctx.stderr)
        return SlashResult()
    if ctx.session_id == new_id:
        print(f"Session> already on {new_id}", file=ctx.stdout)
        return SlashResult()
    return SlashResult(action="switch_session", new_session_id=new_id)


def _handle_delete(ctx: SlashContext, args: str) -> SlashResult:
    session_id = args.strip()
    if not session_id:
        print("Usage: /delete <session-id>", file=ctx.stderr)
        return SlashResult()
    try:
        ctx.session_store.validate_session_id(session_id)
    except ValueError as exc:
        print(f"Invalid session id: {exc}", file=ctx.stderr)
        return SlashResult()
    if ctx.session_id == session_id:
        print("Cannot delete current session", file=ctx.stderr)
        return SlashResult()

    answer = ctx.input_func(f"Delete session {session_id}? [y/N] ")
    if answer.strip().lower() not in {"y", "yes"}:
        print("Delete> cancelled", file=ctx.stdout)
        return SlashResult()

    deleted = ctx.session_store.delete_session(session_id)
    print(
        f"Delete> removed {session_id}"
        if deleted
        else f"Delete> no files for {session_id}",
        file=ctx.stdout,
    )
    return SlashResult()


def _handle_name(ctx: SlashContext, args: str) -> SlashResult:
    title = args.strip()
    if ctx.session_id is None:
        print("/name requires --session or /switch first", file=ctx.stderr)
        return SlashResult()
    if not title:
        print("Usage: /name <title>", file=ctx.stderr)
        return SlashResult()
    ctx.session_store.update_meta(ctx.session_id, title=title)
    print(f"Session> named {title}", file=ctx.stdout)
    return SlashResult()


def _handle_compact(ctx: SlashContext, _args: str) -> SlashResult:
    if ctx.session_id is None:
        print("/compact requires --session or /switch first", file=ctx.stderr)
        return SlashResult()
    compressor = SessionCompressor(
        ctx.session_store,
        ctx.session_id,
        ModelSummarizer(
            ctx.model, max_tool_result_chars=ctx.agent.budget.max_tool_result_chars
        ),
        keep_recent_entries=ctx.agent.budget.max_recent_messages,
        compact_trigger_entries=ctx.agent.runtime.compact_trigger_entries,
        max_summary_chars=ctx.agent.budget.max_session_summary_chars,
        max_tool_result_chars=ctx.agent.budget.max_tool_result_chars,
    )
    compacted = compressor.maybe_major_compact()
    print(f"Compact> {'compacted' if compacted else 'no-op'}", file=ctx.stdout)
    return SlashResult()


def _handle_memory(ctx: SlashContext, _args: str) -> SlashResult:
    text = MarkdownMemory(ctx.workspace_root).read()
    if text:
        print(text, file=ctx.stdout)
    else:
        print("Memory> no memory yet", file=ctx.stdout)
    return SlashResult()


def _handle_skills(ctx: SlashContext, _args: str) -> SlashResult:
    """Open the one-shot skill selector; the chosen skill becomes the pending
    selection (PRD R5). Cancelling keeps the current pending unchanged."""
    if (
        ctx.skill_registry is None
        or ctx.skill_selector is None
        or ctx.skill_pending is None
    ):
        print("/skills unavailable: no skill registry wired", file=ctx.stderr)
        return SlashResult()
    # Legacy paths (cli.run_chat) may never have refreshed the catalog: refresh
    # here so the first /skills input sees the current snapshot instead of a
    # None misreported as "no skills found".
    snapshot = ctx.skill_registry.snapshot() or ctx.skill_registry.refresh()
    if not snapshot.descriptors:
        print("Skills> no skills found", file=ctx.stdout)
        return SlashResult()
    chosen = ctx.skill_selector(snapshot)
    if chosen is None:
        return SlashResult()
    ctx.skill_pending.replace(chosen)
    return SlashResult()


def _handle_subagents(ctx: SlashContext, args: str) -> SlashResult:
    """Select one configured subagent for the next ordinary task."""
    if args.strip():
        print("Usage: /subagents", file=ctx.stderr)
        return SlashResult()
    if (
        ctx.agent_registry is None
        or ctx.subagent_pending is None
        or ctx.subagent_selector is None
    ):
        print("/subagents unavailable: no agent selector wired", file=ctx.stderr)
        return SlashResult()
    snapshot = ctx.agent_registry.snapshot()
    if snapshot is None or not snapshot.agents:
        print("Subagents> no subagents found", file=ctx.stdout)
        return SlashResult()
    chosen = ctx.subagent_selector(snapshot)
    if chosen is not None:
        ctx.subagent_pending.replace(chosen)
    return SlashResult()


def build_builtin_slash_registry() -> SlashCommandRegistry:
    registry = SlashCommandRegistry()
    registry.register(
        SlashCommand("exit", "Exit chat.", _handle_exit, aliases=["quit", "q"])
    )
    registry.register(SlashCommand("sessions", "List sessions.", _handle_sessions))
    registry.register(
        SlashCommand(
            "switch", "Switch session.", _handle_switch, argument_hint="<session-id>"
        )
    )
    registry.register(
        SlashCommand(
            "delete", "Delete a session.", _handle_delete, argument_hint="<session-id>"
        )
    )
    registry.register(
        SlashCommand(
            "name", "Name current session.", _handle_name, argument_hint="<title>"
        )
    )
    registry.register(
        SlashCommand("compact", "Compact current session.", _handle_compact)
    )
    registry.register(SlashCommand("memory", "Show MEMORY.md.", _handle_memory))
    registry.register(
        SlashCommand("skills", "Select a skill for this turn.", _handle_skills)
    )
    registry.register(
        SlashCommand("subagents", "Select a subagent for the next task.", _handle_subagents)
    )
    return registry
