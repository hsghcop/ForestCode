"""BackendBridge: the runtime facade for the interactive CLI.

Owns process-level singletons (model/agent, command service, slash registry)
injected by the composition root, plus session-scoped state it creates and
rebuilds on ``/switch``.

Threading split: the Bridge does NOT own threads, queues, or the
abort token — that is ``TurnRunner``'s job. The Bridge only:
  * ``classify`` — main-thread classification of empty/exit/slash vs. a model
    task (slash handlers run inline; they are millisecond-local in this step);
  * ``run_one_turn`` — runs one backend turn given a sink / confirm / abort,
    so the worker thread can call it without touching UI;
  * ``render_turn_epilogue`` — main-thread post-turn rendering (debug context,
    recorded-message dedup).
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TextIO

from forestcode.config import AgentRuntimeConfig
from forestcode.context import ContextFragment
from forestcode.core.abort import AbortSignal
from forestcode.core.events import CallbackEventSink
from forestcode.core.run_state import RunState
from forestcode.core.types import ModelClient, RunEvent
from forestcode.memory import SessionRecorder, SessionStore
from forestcode.models import ModelAdapterError
from forestcode.plan import PlanStore
from forestcode.plan.serialization import todos_from_dicts, todos_to_dicts
from forestcode.runtime.factory import (
    ObservedModelClient,
    build_agent_loop,
    build_subagent_child_runner,
    parent_visible_tool_names,
)
from forestcode.skills import (
    PendingSkillSelection,
    SkillActivationError,
    SkillRegistry,
    SkillSnapshot,
    build_skill_fragments,
    parse_skill_token,
)
from forestcode.slash_commands import SlashContext, looks_like_command
from forestcode.slash_handlers import build_builtin_slash_registry
from forestcode.subagents.child import (
    combined_context_chars,
    resolve_child_skill_fragments,
)
from forestcode.subagents.config_loader import AgentRegistry, resolve_child_model_config
from forestcode.subagents.coordinator import SubagentCoordinator
from forestcode.subagents.pending import PendingSubagentSelection
from forestcode.subagents.persistence import child_transcript_dir
from forestcode.subagents.tools import create_subagent_tools
from forestcode.subagents.types import (
    MAX_COMBINED_CONTEXT_CHARS,
    AgentConfigSet,
    SubagentRequest,
)
from forestcode.tools import (
    ApprovalRequest,
    CommandService,
    MutationGate,
    PatchService,
    ReadStateStore,
    ToolRuntimeServices,
)

from .confirm import ConfirmationController
from .input import InputController
from .renderer import TerminalRenderer

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class BackendBridgeConfig:
    workspace_root: Path
    session_id: str | None
    agent: AgentRuntimeConfig
    model: ModelClient
    renderer: TerminalRenderer
    input_controller: InputController
    confirmation_controller: ConfirmationController
    debug_context: bool = False
    verbose_events: bool = False
    show_reasoning: bool = False
    # Phase-1 slash compatibility (§9.4a): handlers still print to these raw
    # streams. Removed once SlashCommandView lands in phase 2.
    stdout: TextIO = sys.stdout
    stderr: TextIO = sys.stderr
    # Skills runtime (design §Pending selection). ``pending_skill_selection`` is
    # owned by the bridge; ``skill_selector`` is the UI chosen by the composition
    # root (arrow-key menu in the full tier, numbered fallback otherwise).
    pending_skill_selection: PendingSkillSelection | None = None
    skill_selector: Callable[[SkillSnapshot], str | None] | None = None
    skill_registry: SkillRegistry | None = None
    pending_subagent_selection: PendingSubagentSelection | None = None
    subagent_selector: Callable[[AgentConfigSet], str | None] | None = None


@dataclass(slots=True)
class InputOutcome:
    action: Literal["continue", "exit", "error"]
    exit_code: int = 0
    error: str | None = None
    run_state: RunState | None = None
    session_changed: bool = False
    session_id: str | None = None


@dataclass(slots=True)
class Decision:
    """Main-thread classification of a line of user input (plan §7.2)."""

    kind: Literal["noop", "exit", "run"]
    task: str | None = None
    exit_code: int = 0
    session_changed: bool = False
    session_id: str | None = None
    # Skills: the fixed snapshot and transient fragments for this run, resolved
    # on the main thread and consumed by the worker (immutable, thread-safe).
    transient_fragments: tuple[ContextFragment, ...] = ()
    skills_snapshot: SkillSnapshot | None = None
    # Subagents: immutable per-run launch context (design §Child Construction
    # and Context). None on the single-agent path.
    launch_context: RunLaunchContext | None = None


@dataclass(frozen=True, slots=True)
class ManualDelegation:
    """Selected child and task for the manual two-step ``/subagents`` flow."""

    agent_name: str
    task: str


@dataclass(frozen=True, slots=True)
class RunLaunchContext:
    """Immutable, typed per-run launch context (design §Child Construction).

    Fixed before the run starts: the skill snapshot, typed activated skill
    names, transient fragments, an optional manual child selection and the
    validated AgentConfigSet. Never mutated during the run;
    components must not reverse-parse names from labels or bodies (R7).
    """

    skills_snapshot: SkillSnapshot | None
    activated_skill_names: tuple[str, ...]
    transient_fragments: tuple[ContextFragment, ...]
    manual_delegation: ManualDelegation | None = None
    agent_set: AgentConfigSet | None = None


@dataclass(slots=True)
class TurnExecution:
    """Worker-produced turn result for main-thread epilogue rendering."""

    outcome: InputOutcome
    observed_inputs: list | None = None
    recorded_message: str | None = None


@dataclass(slots=True)
class TurnFlags:
    """Per-turn event tracking owned by the Bridge (§5.4)."""

    memory_record_failed: bool = False


class BackendBridge:
    def __init__(self, config: BackendBridgeConfig) -> None:
        self._config = config
        self.workspace_root = Path(config.workspace_root).resolve()
        self.model = config.model
        self.agent = config.agent
        self._renderer = config.renderer
        self._input = config.input_controller
        self._confirm = config.confirmation_controller
        # process-level singletons
        self._command_service = CommandService()
        self._slash_registry = build_builtin_slash_registry()
        # skills runtime: process-local registry + one-shot pending selection
        # (design §Pending selection). The snapshot is refreshed per classify.
        self._skill_registry = config.skill_registry or SkillRegistry(
            workspace_root=self.workspace_root
        )
        self._pending_skill = config.pending_skill_selection or PendingSkillSelection()
        self._skill_selector = config.skill_selector
        self._skill_snapshot: SkillSnapshot | None = None
        # subagents runtime: process-local agent registry; snapshot fixed per
        # parent run (design §Configuration Contract). Empty by default.
        self._agent_registry = AgentRegistry(workspace_root=self.workspace_root)
        self._agent_snapshot: AgentConfigSet | None = None
        self._pending_subagent = (
            config.pending_subagent_selection or PendingSubagentSelection()
        )
        self._subagent_selector = config.subagent_selector
        # session-scoped state (rebuilt on switch)
        self._session_id: str | None = config.session_id
        self._build_session_scope(config.session_id)

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def confirm_controller(self) -> ConfirmationController:
        """The confirmation controller, so the app can build a TurnRunner."""
        return self._confirm

    # -- session-scoped state ---------------------------------------------
    def _build_session_scope(self, session_id: str | None) -> None:
        self._session_store = SessionStore(self.workspace_root)
        self._read_state_store = ReadStateStore()
        self._patch_service = PatchService(read_state_store=self._read_state_store)
        self._plan_store = self._build_plan_store(self._session_store, session_id)
        self._runtime = ToolRuntimeServices(
            read_state_store=self._read_state_store,
            patch_service=self._patch_service,
            command_service=self._command_service,
            plan_store=self._plan_store,
            confirm=self._confirm.confirm,
        )
        self._session_id = session_id

    def _build_plan_store(
        self, session_store: SessionStore, session_id: str | None
    ) -> PlanStore:
        if session_id is None:
            return PlanStore()
        plan_store = PlanStore(
            on_change=lambda items: session_store.save_plan(
                session_id, todos_to_dicts(items)
            )
        )
        plan_store.seed(todos_from_dicts(session_store.load(session_id).plan))
        return plan_store

    def switch_session(self, new_id: str) -> None:
        """Rebuild all session-scoped state for a new session (§9.5)."""
        self._build_session_scope(new_id)
        # A pending skill selection must not leak across sessions (PRD R6).
        self._pending_skill.clear()
        self._pending_subagent.clear()
        self._renderer.render_session_status(new_id)

    def _build_slash_context(self) -> SlashContext:
        return SlashContext(
            workspace_root=self.workspace_root,
            session_id=self._session_id,
            session_store=self._session_store,
            plan_store=self._plan_store,
            runtime=self._runtime,
            agent=self.agent,
            model=self.model,
            stdout=self._config.stdout,
            stderr=self._config.stderr,
            input_func=self._input.read_confirmation,
            registry=self._slash_registry,
            skill_registry=self._skill_registry,
            skill_pending=self._pending_skill,
            skill_selector=self._skill_selector,
            agent_registry=self._agent_registry,
            subagent_pending=self._pending_subagent,
            subagent_selector=self._subagent_selector,
        )

    def pending_skill_marker(self) -> str | None:
        """Read-only marker text for the next input prompt (PRD R5)."""
        return self._pending_skill.marker_text()

    def pending_subagent_marker(self) -> str | None:
        """Read-only marker for a child selected by ``/subagents``."""
        return self._pending_subagent.marker_text()

    # -- classification (main thread) -------------------------------------
    def classify(self, text: str) -> Decision:
        # All slash commands run inline here on the main thread (v0.10 parity).
        # Plan decision #35 would route runs_long slash (`/compact`, which calls
        # the model) through TurnRunner so Ctrl+C cancels it instead of exiting;
        # that is intentionally deferred — it needs spinner suppression for the
        # handler's raw prints, SlashResult->InputOutcome mapping, and abort
        # threading into ModelSummarizer. Model *turns* are already cancellable
        # via TurnRunner (§7), which is the core of B3.
        stripped = text.strip()
        if not stripped:
            return Decision(kind="noop")
        if stripped.lower() in {"exit", "quit"}:
            self._pending_skill.clear()
            self._pending_subagent.clear()
            return Decision(kind="exit", exit_code=0)

        # Skills: refresh once per top-level classification (PRD R4/R6). Both the
        # /skills slash handler and the $name resolution below read this snapshot;
        # it stays fixed for the run that follows.
        self._skill_snapshot = self._skill_registry.refresh()
        # Subagents: refresh the agent config set once per classification; the
        # snapshot stays fixed for the run that follows (design §Configuration
        # Contract / R1).
        self._agent_snapshot = self._agent_registry.refresh(
            valid_tool_names=parent_visible_tool_names(
                self.workspace_root,
                self.agent,
                session_store=self._session_store,
                session_id=self._session_id,
                skills_snapshot=self._skill_snapshot,
                enable_write_tools=True,
            ),
            skills_snapshot=self._skill_snapshot,
        )

        user_task = text
        session_changed = False
        if stripped.startswith("/"):
            head, _, raw_args = stripped[1:].partition(" ")
            command = self._slash_registry.get(head)
            if command is not None:
                result = command.handler(self._build_slash_context(), raw_args.strip())
                if result.action == "exit":
                    self._pending_skill.clear()
                    self._pending_subagent.clear()
                    return Decision(kind="exit", exit_code=result.exit_code)
                if result.action == "switch_session":
                    if result.new_session_id is None:
                        self._renderer.render_user_error(
                            "Slash command error: missing new session id"
                        )
                        return Decision(kind="noop")
                    self.switch_session(result.new_session_id)
                    session_changed = True
                if result.prompt_text is not None:
                    user_task = result.prompt_text
                else:
                    # Selection/reporting commands complete on the main thread.
                    return Decision(
                        kind="noop",
                        session_changed=session_changed,
                        session_id=self._session_id,
                    )
            elif looks_like_command(head):
                available = ", ".join(
                    f"/{cmd.name}" for cmd in self._slash_registry.list()
                )
                self._renderer.render_user_error(f"Unknown command: /{head}")
                self._renderer.render_user_error(f"Available: {available}")
                return Decision(kind="noop")
            # else: not command-like -> fall through to model with original text

        # Skills activation (PRD R4/R6): resolve an explicit leading
        # ``$skill-name`` token or the pending ``/skills`` selection. The snapshot
        # is fixed for this run; explicit tokens override the pending selection;
        # errors never reach the model.
        parsed = parse_skill_token(user_task)
        if parsed.error is not None:
            self._renderer.render_user_error(parsed.error)
            return Decision(kind="noop")
        task = parsed.task if parsed.name is not None else user_task
        activation_name, activation_error = self._resolve_activation(parsed.name)
        if activation_error is not None:
            self._renderer.render_user_error(activation_error)
            return Decision(kind="noop")
        try:
            fragments = self._build_skill_fragments(activation_name)
        except SkillActivationError as exc:
            if parsed.name is not None:
                self._renderer.render_user_error(f"Skills> {exc}")
                return Decision(kind="noop")
            self._pending_skill.clear()
            self._renderer.render_warning(
                f"Selected skill is no longer available: {exc.name}"
            )
            fragments = self._build_skill_fragments(None)
        snapshot = self._skill_snapshot
        manual_delegation: ManualDelegation | None = None
        pending_agent = self._pending_subagent.name
        if pending_agent is not None:
            agent_config = (
                self._agent_snapshot.get(pending_agent)
                if self._agent_snapshot is not None
                else None
            )
            if agent_config is None:
                self._pending_subagent.clear()
                self._renderer.render_user_error(
                    f"Selected subagent is no longer available: {pending_agent}"
                )
                return Decision(kind="noop")
            else:
                manual_delegation = ManualDelegation(
                    agent_name=pending_agent,
                    task=task,
                )
        # Immutable per-run launch context: typed activated skill names, the
        # fixed agent snapshot and the manual delegation (if any). The worker
        # reads this; nothing here is mutable across the run (design §Child
        # Construction and Context).
        launch_context = RunLaunchContext(
            skills_snapshot=(
                snapshot if snapshot is not None and snapshot.descriptors else None
            ),
            activated_skill_names=(
                (activation_name,) if activation_name is not None else ()
            ),
            transient_fragments=fragments,
            manual_delegation=manual_delegation,
            agent_set=self._agent_snapshot,
        )
        return Decision(
            kind="run",
            task=task,
            session_changed=session_changed,
            session_id=self._session_id,
            transient_fragments=fragments,
            skills_snapshot=(
                snapshot if snapshot is not None and snapshot.descriptors else None
            ),
            launch_context=launch_context,
        )

    # -- skills resolution helpers -----------------------------------------
    def _resolve_activation(
        self, explicit_name: str | None
    ) -> tuple[str | None, str | None]:
        """Return ``(activation_name, error)`` for this run.

        Explicit ``$skill-name`` wins over the pending selection (PRD R4). An
        unknown explicit skill is a user error (``error`` set, no model call). A
        pending skill that vanished is cleared with a warning and the run
        proceeds normally.
        """
        snapshot = self._skill_snapshot
        if explicit_name is not None:
            if snapshot is None or snapshot.get(explicit_name) is None:
                return None, f"Unknown skill: {explicit_name}"
            return explicit_name, None
        pending_name = self._pending_skill.name
        if pending_name is not None:
            if snapshot is None or snapshot.get(pending_name) is None:
                name = pending_name
                self._pending_skill.clear()
                self._renderer.render_warning(
                    f"Selected skill is no longer available: {name}"
                )
                return None, None
            return pending_name, None
        return None, None

    def _build_skill_fragments(
        self, activation_name: str | None
    ) -> tuple[ContextFragment, ...]:
        """Build the catalog fragment plus, for manual activation, the body fragment.

        The snapshot is fixed for this run; fragments are transient and never
        recorded (PRD R7). Delegates to the shared skills helper.
        """
        snapshot = self._skill_snapshot
        if snapshot is None or not snapshot.descriptors:
            return ()
        return build_skill_fragments(snapshot, activation_name)

    # -- one backend turn (worker thread) ---------------------------------
    def run_one_turn(
        self,
        user_task: str,
        *,
        sink: Callable[[RunEvent], None],
        confirm: Callable[[ApprovalRequest], bool],
        abort: AbortSignal | None = None,
        transient_fragments: tuple[ContextFragment, ...] = (),
        skills_snapshot: SkillSnapshot | None = None,
        launch_context: RunLaunchContext | None = None,
        confirm_proxy: Any | None = None,
    ) -> TurnExecution:
        """Run one model turn. Emits events to ``sink`` (no direct render).

        Returns a TurnExecution for the main thread to render. Raises ``Aborted``
        (caller cancels) and lets generic exceptions propagate so the TurnRunner
        reports them; ``ModelAdapterError`` is caught and reported as a model
        error (§5.3).

        ``launch_context`` carries the run-fixed skill/agent snapshots and an
        optional manual child selection. Manual selection runs only that child;
        otherwise configured agents are exposed to the parent through the four
        autonomous delegation tools.
        """
        observed = ObservedModelClient(self.model)
        flags = TurnFlags()

        def on_event(event: RunEvent) -> None:
            try:
                if event.type == "memory_record_failed":
                    flags.memory_record_failed = True
                sink(event)
            except Exception:
                logger.exception("event handler error (%s)", event.type)

        if launch_context is not None:
            for skill_name in launch_context.activated_skill_names:
                on_event(RunEvent("skill_activated", {"name": skill_name}))

        # Per-turn runtime with the thread-bridged confirm + abort overriding the
        # session defaults (plan §7.2). dataclasses.replace keeps the rest.
        runtime = replace(self._runtime, confirm=confirm)
        coordinator: SubagentCoordinator | None = None
        subagent_tools: list[Any] = []
        launch_fragments = transient_fragments
        if launch_context is not None and launch_context.agent_set is not None:
            agent_set = launch_context.agent_set
            # The real CLI injects a ModelRouter whose ``config`` feeds child
            # model inheritance; a model without it (e.g. a test fake) cannot
            # resolve children, so subagents stay inert on that path.
            parent_model = getattr(self._config.model, "config", None)
            manual = launch_context.manual_delegation
            if manual is not None and agent_set.get(manual.agent_name) is None:
                self._pending_skill.clear()
                self._pending_subagent.clear()
                return TurnExecution(
                    outcome=InputOutcome(
                        action="error",
                        exit_code=1,
                        error=f"Unknown subagent: {manual.agent_name}",
                        session_id=self._session_id,
                    )
                )
            if manual is not None and parent_model is None:
                self._pending_skill.clear()
                self._pending_subagent.clear()
                return TurnExecution(
                    outcome=InputOutcome(
                        action="error",
                        exit_code=1,
                        error="Subagent error: parent model configuration is unavailable",
                        session_id=self._session_id,
                    )
                )
            if agent_set.agents and parent_model is not None:
                # Per-parent-run single-writer gate shared by the parent and all
                # children (design §Mutation Gate): patch/save-memory/command
                # mutation sections serialize across every agent of this run.
                mutation_gate = MutationGate()
                runtime = replace(runtime, mutation_gate=mutation_gate)
                session_dir = None
                if self._session_id is not None:
                    try:
                        session_dir = child_transcript_dir(
                            self.workspace_root, self._session_id
                        )
                    except ValueError:
                        # Unsafe session id: children simply skip transcripts.
                        logger.warning("unsafe session id, child transcripts disabled")
                coordinator_ref: list[SubagentCoordinator] = []

                def approval_started(task_id: str) -> None:
                    coordinator_ref[0].approval_started(task_id)

                def approval_finished(task_id: str) -> None:
                    coordinator_ref[0].approval_finished(task_id)

                child_runner = build_subagent_child_runner(
                        workspace_root=self.workspace_root,
                        agent_set=agent_set,
                        parent_model=parent_model,
                        environ=os.environ.get,
                        skills_snapshot=launch_context.skills_snapshot,
                        activated_skill_names=launch_context.activated_skill_names,
                        inherited_fragments=launch_context.transient_fragments,
                        parent_visible_tools=parent_visible_tool_names(
                            self.workspace_root,
                            self.agent,
                            session_store=self._session_store,
                            session_id=self._session_id,
                            skills_snapshot=launch_context.skills_snapshot,
                            enable_write_tools=True,
                        ),
                        mutation_gate=mutation_gate,
                        confirm_bridge=confirm_proxy,
                        approval_started=approval_started,
                        approval_finished=approval_finished,
                        events=CallbackEventSink(on_event),
                        session_root=session_dir,
                        agent=self.agent,
                        command_service=self._command_service,
                    )
                coordinator = SubagentCoordinator(
                    child_runner,
                    events=CallbackEventSink(on_event),
                    session_dir=session_dir,
                    confirm_cancel=(
                        confirm_proxy.cancel_task
                        if confirm_proxy is not None
                        and hasattr(confirm_proxy, "cancel_task")
                        else None
                    ),
                )
                coordinator_ref.append(coordinator)
                subagent_tools = create_subagent_tools(
                    coordinator,
                    agent_set=agent_set,
                    skills_snapshot=launch_context.skills_snapshot,
                    activated_skill_names=launch_context.activated_skill_names,
                    inherited_fragments=launch_context.transient_fragments,
                )
                if manual is not None:
                    try:
                        return self._run_manual_subagent(
                            coordinator=coordinator,
                            delegation=manual,
                            agent_set=agent_set,
                            parent_model=parent_model,
                            launch_context=launch_context,
                            on_event=on_event,
                            abort=abort,
                        )
                    finally:
                        coordinator.cleanup()
                        self._pending_skill.clear()
                        self._pending_subagent.clear()
        try:
            loop = build_agent_loop(
                observed,
                self.workspace_root,
                agent=self.agent,
                session_id=self._session_id,
                events=CallbackEventSink(on_event),
                enable_write_tools=True,
                runtime=runtime,
                session_store=self._session_store,
                abort=abort,
                skills_snapshot=skills_snapshot,
                transient_fragments=launch_fragments,
                subagents=coordinator,
                subagent_tools=subagent_tools,
            )
            state = loop.run(user_task)
        except ModelAdapterError as exc:
            return TurnExecution(
                outcome=InputOutcome(
                    action="error",
                    exit_code=1,
                    error=f"Model error: {exc}",
                    session_id=self._session_id,
                )
            )
        finally:
            # PRD R6: once a run actually starts, the one-shot selection is
            # consumed — clear it on completion, failure, and cancellation alike.
            self._pending_skill.clear()
            self._pending_subagent.clear()

        if state.final_text is None:
            return TurnExecution(
                outcome=InputOutcome(
                    action="error",
                    exit_code=1,
                    error=f"Agent error: {state.error or 'unknown error'}",
                    run_state=state,
                    session_id=self._session_id,
                ),
                observed_inputs=observed.inputs,
            )

        recorded = None
        if state.final_text and self._session_id and not flags.memory_record_failed:
            recorded = f"recorded .forestcode/sessions/{self._session_id}.jsonl"
        return TurnExecution(
            outcome=InputOutcome(
                action="continue",
                run_state=state,
                session_id=self._session_id,
            ),
            observed_inputs=observed.inputs,
            recorded_message=recorded,
        )

    def _run_manual_subagent(
        self,
        *,
        coordinator: SubagentCoordinator,
        delegation: ManualDelegation,
        agent_set: AgentConfigSet,
        parent_model: Any,
        launch_context: RunLaunchContext,
        on_event: Callable[[RunEvent], None],
        abort: AbortSignal | None,
    ) -> TurnExecution:
        """Run one selected child and return its answer without a parent model call."""
        config = agent_set.get(delegation.agent_name)
        if config is None:
            return self._manual_subagent_error(
                f"Unknown subagent: {delegation.agent_name}"
            )
        try:
            resolve_child_model_config(parent_model, config.model, os.environ.get)
            fragments = resolve_child_skill_fragments(
                config,
                launch_context.skills_snapshot,
                launch_context.activated_skill_names,
                launch_context.transient_fragments,
            )
            total = combined_context_chars(
                config.instructions, delegation.task, fragments
            )
            if total > MAX_COMBINED_CONTEXT_CHARS:
                raise ValueError(
                    "delegation context too large: agent instructions + prompt + "
                    f"pre-injected skills total {total} characters, exceeding the "
                    f"{MAX_COMBINED_CONTEXT_CHARS}-character limit"
                )
        except (ModelAdapterError, ValueError) as exc:
            return self._manual_subagent_error(f"Subagent error: {exc}")

        queued = coordinator.delegate(
            SubagentRequest(
                task_id="",
                agent_name=delegation.agent_name,
                description="selected via /subagents",
                prompt=delegation.task,
            ),
            timeout_seconds=config.task_timeout_seconds,
        )
        result = None
        final_snapshot = queued
        while True:
            outcome = coordinator.wait(
                [queued.task_id], timeout=60, abort=abort
            )
            final_snapshot = outcome.snapshots[0]
            result = outcome.results.get(queued.task_id) or result
            if final_snapshot.status in {"completed", "failed", "cancelled"}:
                break
            if (
                final_snapshot.status == "cancelling"
                and final_snapshot.cancel_reason == "timeout"
            ):
                return self._manual_subagent_error(
                    f"Subagent error: {delegation.agent_name} timed out"
                )

        if final_snapshot.status != "completed":
            detail = (
                final_snapshot.error
                or final_snapshot.cancel_reason
                or final_snapshot.status
            )
            return self._manual_subagent_error(
                f"Subagent error: {delegation.agent_name} {detail}"
            )
        final_text = result.final_text if result is not None else None
        if not final_text:
            return self._manual_subagent_error(
                f"Subagent error: {delegation.agent_name} returned no final response"
            )

        state = RunState.start(delegation.task)
        if result is not None:
            state.turns = result.turn_count
        state.add_assistant_text(final_text)
        state.finish(final_text)
        on_event(RunEvent("assistant_text_received", {"text": final_text}))

        recorded = None
        if self._session_id is not None:
            try:
                SessionRecorder(
                    self._session_store,
                    session_id=self._session_id,
                    max_tool_result_store_chars=self.agent.tool_output_max_chars,
                ).record_run(state)
            except Exception as exc:  # noqa: BLE001 - recording must not hide child answer
                on_event(RunEvent("memory_record_failed", {"error": str(exc)}))
            else:
                recorded = f"recorded .forestcode/sessions/{self._session_id}.jsonl"
        return TurnExecution(
            outcome=InputOutcome(
                action="continue",
                run_state=state,
                session_id=self._session_id,
            ),
            observed_inputs=[],
            recorded_message=recorded,
        )

    def _manual_subagent_error(self, message: str) -> TurnExecution:
        return TurnExecution(
            outcome=InputOutcome(
                action="error",
                exit_code=1,
                error=message,
                session_id=self._session_id,
            ),
            observed_inputs=[],
        )

    # -- post-turn rendering (main thread) --------------------------------
    def render_turn_epilogue(self, execution: TurnExecution) -> None:
        if self._config.debug_context and execution.observed_inputs is not None:
            self._renderer.render_context_debug(execution.observed_inputs)
        if execution.recorded_message:
            self._renderer.render_memory_status(execution.recorded_message)
