"""Command line entrypoint for ForestCode."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO

from forestcode.config import AgentRuntimeConfig, ConfigError, load_config
from forestcode.context import ModelInput
from forestcode.core.types import ModelClient, RunEvent
from forestcode.memory import SessionStore
from forestcode.models import ModelAdapterError
from forestcode.plan import PlanStore
from forestcode.plan.serialization import todos_from_dicts, todos_to_dicts
from forestcode.runtime.factory import (
    ObservedModelClient,
    build_agent_loop,
    build_model_client,
    build_model_client_from_env,
)
from forestcode.skills import (
    PendingSkillSelection,
    SkillRegistry,
)
from forestcode.slash_commands import SlashContext
from forestcode.slash_handlers import build_builtin_slash_registry
from forestcode.subagents import PendingSubagentSelection
from forestcode.terminal.app import ForestCodeCliApp
from forestcode.terminal.bridge import BackendBridge, BackendBridgeConfig
from forestcode.terminal.confirm import ConfirmationController
from forestcode.terminal.input import InputController, StdinInputController
from forestcode.terminal.renderer import FrontendState, build_renderer
from forestcode.terminal.turn_runner import TurnRunner
from forestcode.tools import (
    ApprovalRequest,
    CommandService,
    PatchService,
    ReadStateStore,
    ToolRuntimeServices,
)

__all__ = [
    "ObservedModelClient",
    "build_agent_loop",
    "build_model_client",
    "build_model_client_from_env",
    "build_parser",
    "main",
    "print_agent_event",
    "print_agent_events",
    "run_chat",
]


def make_terminal_confirm(
    stdout: TextIO,
    input_func: Callable[[str], str],
) -> Callable[[ApprovalRequest], bool]:
    # Legacy `forestcode chat` confirm, upgraded to the structured B4 contract
    # (plan §11-B4). Reads neutral fact fields off the request rather than the
    # tool name / preview string.
    def confirm(request: ApprovalRequest) -> bool:
        if request.kind == "command":
            print("Command> requires approval", file=stdout)
            print(request.preview, file=stdout)
            answer = input_func("Execute? [y/N] ")
        else:
            print(f"Patch> {request.tool_name} requires approval", file=stdout)
            print(request.preview, file=stdout)
            answer = input_func("Apply patch? [y/N] ")
        return answer.strip().lower() in {"y", "yes"}

    return confirm


def _build_plan_store(session_store: SessionStore, session_id: str | None) -> PlanStore:
    if session_id is None:
        return PlanStore()
    plan_store = PlanStore(on_change=lambda items: session_store.save_plan(session_id, todos_to_dicts(items)))
    plan_store.seed(todos_from_dicts(session_store.load(session_id).plan))
    return plan_store


def _switch_session(ctx: SlashContext, new_id: str) -> SlashContext:
    workspace = ctx.workspace_root
    new_store = SessionStore(workspace)
    new_plan_store = _build_plan_store(new_store, new_id)
    new_read_state_store = ReadStateStore()
    new_patch_service = PatchService(read_state_store=new_read_state_store)
    new_runtime = replace(
        ctx.runtime,
        read_state_store=new_read_state_store,
        patch_service=new_patch_service,
        plan_store=new_plan_store,
    )
    print(f"Session> {new_id}", file=ctx.stdout, flush=True)
    return replace(
        ctx,
        session_id=new_id,
        session_store=new_store,
        plan_store=new_plan_store,
        runtime=new_runtime,
    )


def _build_slash_context(
    *,
    model: ModelClient,
    workspace: Path,
    session_id: str | None,
    agent: AgentRuntimeConfig,
    stdout: TextIO,
    stderr: TextIO,
    input_func: Callable[[str], str],
) -> SlashContext:
    read_state_store = ReadStateStore()
    patch_service = PatchService(read_state_store=read_state_store)
    session_store = SessionStore(workspace)
    plan_store = _build_plan_store(session_store, session_id)
    runtime = ToolRuntimeServices(
        read_state_store=read_state_store,
        patch_service=patch_service,
        command_service=CommandService(),
        plan_store=plan_store,
        confirm=make_terminal_confirm(stdout, input_func),
    )
    registry = build_builtin_slash_registry()
    # Legacy chat wires the numbered selector (no prompt_toolkit tier here) and
    # its own pending selection so /skills works like the interactive frontend.
    from forestcode.terminal.skill_select import make_numbered_skill_selector

    skill_registry = SkillRegistry(workspace)
    skill_pending = PendingSkillSelection()
    skill_selector = make_numbered_skill_selector(input_func=input_func, stdout=stdout)
    return SlashContext(
        workspace_root=workspace,
        session_id=session_id,
        session_store=session_store,
        plan_store=plan_store,
        runtime=runtime,
        agent=agent,
        model=model,
        stdout=stdout,
        stderr=stderr,
        input_func=input_func,
        registry=registry,
        skill_registry=skill_registry,
        skill_pending=skill_pending,
        skill_selector=skill_selector,
    )


def print_agent_event(
    event: RunEvent,
    stdout: TextIO,
    verbose: bool = False,
    show_reasoning: bool = False,
    reasoning_display_max_chars: int = 2000,
) -> None:
    if event.type == "tool_call_started":
        print(f"Tool> {event.payload['tool_name']}", file=stdout, flush=True)
    elif event.type == "tool_call_finished":
        status = "ok" if event.payload.get("ok") else "error"
        print(f"Tool> {event.payload['tool_name']} {status}", file=stdout, flush=True)
        if event.payload.get("tool_name") == "write_todos" and event.payload.get("summary"):
            print(f"Plan> {event.payload['summary']}", file=stdout, flush=True)
    elif event.type == "assistant_reasoning_received":
        if show_reasoning:
            print(
                f"Reasoning> {_truncate_reasoning(str(event.payload['text']), reasoning_display_max_chars)}",
                file=stdout,
                flush=True,
            )
    elif event.type == "assistant_text_received":
        print(f"Assistant> {event.payload['text']}", file=stdout, flush=True)
    elif event.type == "memory_record_failed":
        print(f"Memory> record failed: {event.payload['error']}", file=stdout, flush=True)
    elif event.type == "session_compaction_finished":
        print(f"Memory> compacted {event.payload['kind']}", file=stdout, flush=True)
    elif event.type == "session_compaction_failed":
        print(f"Memory> compaction failed ({event.payload['kind']}): {event.payload['error']}", file=stdout, flush=True)
    elif verbose:
        print(f"Event> {event.type} {event.payload}", file=stdout, flush=True)


def print_agent_events(
    events: list[RunEvent],
    stdout: TextIO,
    verbose: bool = False,
    show_reasoning: bool = False,
    reasoning_display_max_chars: int = 2000,
) -> None:
    for event in events:
        print_agent_event(
            event,
            stdout,
            verbose=verbose,
            show_reasoning=show_reasoning,
            reasoning_display_max_chars=reasoning_display_max_chars,
        )


def print_context_debug(model_inputs: list[ModelInput], stdout: TextIO) -> None:
    if not model_inputs:
        print("Context> no model request captured", file=stdout)
        return

    for index, model_input in enumerate(model_inputs, start=1):
        metadata = model_input.metadata
        sources = metadata.get("context_sources") or []
        tools = metadata.get("selected_tools") or []
        print(f"Context> turn {index}", file=stdout)
        print(f"Context> sources: {', '.join(sources) if sources else '(none)'}", file=stdout)
        print(f"Context> messages: {metadata.get('message_count', len(model_input.messages))}", file=stdout)
        print(f"Context> tools: {', '.join(tools) if tools else '(none)'}", file=stdout)
        print(f"Context> chars: {metadata.get('char_count', 'unknown')}", file=stdout)
        print(f"Context> truncated: {metadata.get('truncated', False)}", file=stdout)


def run_chat(
    model: ModelClient,
    input_func: Callable[[str], str] = input,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
    workspace_root: Path | None = None,
    session_id: str | None = None,
    agent: AgentRuntimeConfig | None = None,
    debug_context: bool = False,
    verbose_events: bool = False,
    show_reasoning: bool | None = None,
) -> int:
    agent = agent or AgentRuntimeConfig()
    print("ForestCode chat. Type /exit to quit.", file=stdout)
    workspace = (workspace_root or Path.cwd()).resolve()
    should_show_reasoning = _env_flag("FORESTCODE_SHOW_REASONING") if show_reasoning is None else show_reasoning
    renderer = build_renderer(
        stdout,
        stderr,
        no_color=True,
        verbose_events=verbose_events,
        show_reasoning=should_show_reasoning,
        reasoning_display_max_chars=agent.reasoning_display_max_chars,
    )
    pending_skill = PendingSkillSelection()
    pending_subagent = PendingSubagentSelection()
    input_controller = StdinInputController(
        input_func,
        marker_provider=lambda: (
            pending_subagent.marker_text() or pending_skill.marker_text()
        ),
    )
    confirmation = ConfirmationController(
        renderer, input_controller, allow_always=False
    )
    from forestcode.terminal.skill_select import make_numbered_skill_selector
    from forestcode.terminal.subagent_select import make_numbered_subagent_selector

    bridge = BackendBridge(
        BackendBridgeConfig(
            workspace_root=workspace,
            session_id=session_id,
            agent=agent,
            model=model,
            renderer=renderer,
            input_controller=input_controller,
            confirmation_controller=confirmation,
            debug_context=debug_context,
            verbose_events=verbose_events,
            show_reasoning=should_show_reasoning,
            stdout=stdout,
            stderr=stderr,
            pending_skill_selection=pending_skill,
            skill_selector=make_numbered_skill_selector(
                input_func=input_func, stdout=stdout
            ),
            pending_subagent_selection=pending_subagent,
            subagent_selector=make_numbered_subagent_selector(
                input_func=input_func, stdout=stdout
            ),
        )
    )
    runner = TurnRunner(renderer, confirmation)
    if session_id:
        print(f"Session> {session_id}", file=stdout)
        print(f"Workspace> {workspace}", file=stdout)

    while True:
        try:
            user_text = input_controller.read_user_input("ForestCode> ")
        except EOFError:
            print(file=stdout)
            return 0
        except KeyboardInterrupt:
            print("\nInterrupted.", file=stderr)
            return 130

        decision = bridge.classify(user_text)
        if decision.kind == "noop":
            continue
        if decision.kind == "exit":
            return decision.exit_code
        outcome = runner.run(
            bridge,
            decision.task or "",
            transient_fragments=decision.transient_fragments,
            skills_snapshot=decision.skills_snapshot,
            launch_context=decision.launch_context,
        )
        if outcome.action == "exit":
            return outcome.exit_code
        if outcome.action == "error":
            renderer.render_user_error(outcome.error or "Agent error: unknown error")
            return outcome.exit_code or 1


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Args shared by the default `forestcode` entry and the legacy `chat` (§3.1)."""
    parser.add_argument("--workspace", type=Path, default=Path.cwd(), help="Workspace root to expose to tools.")
    parser.add_argument("--session", default="default", help="Session id for file-backed memory.")
    parser.add_argument("--no-session", action="store_true", help="Start without recording/replaying a session.")
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Maximum model/tool turns per user task (overrides FORESTCODE_MAX_TURNS / default 10).",
    )
    parser.add_argument("--debug-context", action="store_true", help="Print context metadata for each model request.")
    parser.add_argument("--verbose-events", action="store_true", help="Print all agent runtime events.")
    parser.add_argument("--show-reasoning", action="store_true", help="Print visible model reasoning artifacts.")
    parser.add_argument("--no-project-rules", action="store_true", help="Do not include AGENTS.md in context.")
    parser.add_argument("--no-long-term-memory", action="store_true", help="Do not include MEMORY.md in context.")
    parser.add_argument(
        "--command-tools", action=argparse.BooleanOptionalAction, default=None,
        help="Enable/disable confirm-gated shell command tools (--command-tools / --no-command-tools).",
    )
    parser.add_argument("--no-color", action="store_true", help="Force plain-text rendering (same as NO_COLOR).")
    parser.add_argument(
        "--no-history", action="store_true",
        help="Do not persist the input box history to disk (forces in-memory history).",
    )


def build_input_controller(
    *,
    no_color: bool,
    no_history: bool,
    session_id: str | None,
    workspace_root: Path,
    model_name: str,
    stdout: TextIO = sys.stdout,
    skill_marker_provider: Callable[[], str | None] | None = None,
    skill_candidates_provider: Callable[
        [], Sequence[tuple[str, str]]
    ] | None = None,
) -> InputController:
    """Pick the input controller tier (plan §2.2, §8).

    Full tier (``PromptToolkitInputController``: rule-wrapped growing input box,
    slash completion, history, status bar) only when prompt_toolkit is
    importable AND stdout is a TTY AND color is on; otherwise the plain
    ``StdinInputController``. Decided independently of the renderer tier, so rich
    output can pair with plain input (the rich-only tier).

    ``skill_marker_provider`` renders the one-shot skill marker in both tiers
    (inside the pt layout, or as a prompt prefix in the plain tier).
    """
    color_off = (
        no_color
        or bool(os.environ.get("NO_COLOR"))
        or not getattr(stdout, "isatty", lambda: False)()
    )
    if color_off:
        return StdinInputController(marker_provider=skill_marker_provider)
    try:
        from forestcode.terminal.pt_input import (
            PromptToolkitInputController,
            build_history,
        )
    except ImportError:
        return StdinInputController(marker_provider=skill_marker_provider)  # prompt_toolkit absent -> plain input

    history = build_history(
        session_enabled=session_id is not None,
        workspace_root=workspace_root,
        no_history=no_history,
    )
    state = FrontendState(
        workspace_root=workspace_root, session_id=session_id, model_name=model_name
    )
    registry = build_builtin_slash_registry()
    return PromptToolkitInputController(
        history=history,
        state_provider=lambda: state,
        slash_commands=lambda: [(c.name, c.description) for c in registry.list()],
        skill_candidates=skill_candidates_provider,
        skill_marker_provider=skill_marker_provider,
    )


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    _add_shared_args(parent)

    parser = argparse.ArgumentParser(prog="forestcode", parents=[parent])
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("chat", parents=[parent], help="Legacy interactive chat (alias for forestcode).")
    return parser


def _cli_overrides(args: argparse.Namespace) -> dict[str, object]:
    return {
        "max_turns": args.max_turns,
        "enable_command_tools": args.command_tools,
        "include_project_rules": False if args.no_project_rules else None,
        "include_long_term_memory": False if args.no_long_term_memory else None,
    }


def _first_run_check(args: argparse.Namespace) -> int | None:
    from forestcode.config.settings_file import (
        create_template,
        default_settings_path,
        load_settings,
    )

    try:
        load_config(_cli_overrides(args))
        return None
    except ConfigError:
        pass

    path = default_settings_path()
    if not path.exists():
        create_template(path)
        print(f"Config created: {path}")
        print("Please fill in api_key and model, then run forestcode again.")
        return 0

    try:
        settings = load_settings(path)
    except ConfigError as exc:
        print(f"Config error in {path}: {exc}", file=sys.stderr)
        return 2

    if not str(settings.get("api_key", "")).strip():
        print(f"Config: {path}")
        print("Please fill in the required fields: api_key, model, base_url.")
        print("Then run forestcode again.")
        return 0

    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    first_run_code = _first_run_check(args)
    if first_run_code is not None:
        return first_run_code

    if args.command == "chat":
        try:
            config = load_config(_cli_overrides(args))
        except ConfigError as exc:
            print(f"Config error: {exc}", file=sys.stderr)
            return 2
        model = build_model_client(config.model)
        session_id = None if args.no_session else args.session
        return run_chat(
            model,
            workspace_root=args.workspace,
            session_id=session_id,
            agent=config.agent,
            debug_context=args.debug_context,
            verbose_events=args.verbose_events,
            show_reasoning=args.show_reasoning or None,
        )

    return _run_interactive(args)


def _run_interactive(args: argparse.Namespace) -> int:
    """Default `forestcode` entry: the new ForestCodeCliApp frontend (§16)."""
    try:
        config = load_config(_cli_overrides(args))
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2
    try:
        model = build_model_client(config.model)
    except ModelAdapterError as exc:
        print(f"Model error: {exc}", file=sys.stderr)
        return 1

    workspace = Path(args.workspace).resolve()
    session_id = None if args.no_session else args.session
    show_reasoning = args.show_reasoning or _env_flag("FORESTCODE_SHOW_REASONING")

    renderer = build_renderer(
        sys.stdout,
        sys.stderr,
        no_color=args.no_color,
        verbose_events=args.verbose_events,
        show_reasoning=show_reasoning,
        reasoning_display_max_chars=config.agent.reasoning_display_max_chars,
    )
    # Skills runtime (design §Pending selection): the bridge owns the pending
    # one-shot selection; the input controller renders its marker read-only.
    pending_skill = PendingSkillSelection()
    pending_subagent = PendingSubagentSelection()
    skill_registry = SkillRegistry(workspace)
    skill_registry.refresh()
    input_controller = build_input_controller(
        no_color=args.no_color,
        no_history=args.no_history,
        session_id=session_id,
        workspace_root=workspace,
        model_name=config.model.model,
        skill_marker_provider=lambda: (
            pending_subagent.marker_text() or pending_skill.marker_text()
        ),
        skill_candidates_provider=lambda: [
            (descriptor.name, descriptor.description)
            for descriptor in skill_registry.list()
        ],
    )
    chooser = None
    skill_selector = None
    subagent_selector = None
    try:
        from forestcode.terminal.pt_input import (
            PromptToolkitInputController,
            make_pt_chooser,
        )
        from forestcode.terminal.skill_select import make_pt_skill_selector
        from forestcode.terminal.subagent_select import make_pt_subagent_selector
        if isinstance(input_controller, PromptToolkitInputController):
            from contextlib import nullcontext
            pause = getattr(renderer, "pause_live", nullcontext)
            chooser = make_pt_chooser(pause, allow_always=True)
            skill_selector = make_pt_skill_selector()
            subagent_selector = make_pt_subagent_selector()
    except ImportError:
        pass
    if skill_selector is None:
        from forestcode.terminal.skill_select import make_numbered_skill_selector
        from forestcode.terminal.subagent_select import make_numbered_subagent_selector
        skill_selector = make_numbered_skill_selector(
            input_func=input_controller.read_confirmation,
            stdout=sys.stdout,
        )
        subagent_selector = make_numbered_subagent_selector(
            input_func=input_controller.read_confirmation,
            stdout=sys.stdout,
        )
    confirmation_controller = ConfirmationController(renderer, input_controller, chooser=chooser)
    bridge = BackendBridge(
        BackendBridgeConfig(
            workspace_root=workspace,
            session_id=session_id,
            agent=config.agent,
            model=model,
            renderer=renderer,
            input_controller=input_controller,
            confirmation_controller=confirmation_controller,
            debug_context=args.debug_context,
            verbose_events=args.verbose_events,
            show_reasoning=show_reasoning,
            stdout=sys.stdout,
            stderr=sys.stderr,
            pending_skill_selection=pending_skill,
            skill_selector=skill_selector,
            skill_registry=skill_registry,
            pending_subagent_selection=pending_subagent,
            subagent_selector=subagent_selector,
        )
    )
    state = FrontendState(
        workspace_root=workspace,
        session_id=session_id,
        model_name=config.model.model,
        command_tools_enabled=config.agent.features.enable_command_tools,
        show_reasoning=show_reasoning,
        debug_context=args.debug_context,
    )
    return ForestCodeCliApp(bridge, renderer, input_controller, state).run()


def _truncate_reasoning(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    raise SystemExit(main())
