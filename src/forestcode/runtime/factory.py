"""Backend assembly functions for ForestCode.

These build the model client and a per-turn ``AgentLoop`` from already-resolved
config. They are the reusable装配 layer: ``cli.py`` (composition root) and the
terminal ``BackendBridge`` both call into here, but this module imports no
frontend code.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from forestcode.config import AgentRuntimeConfig
from forestcode.context import ContextFragment, ContextRequest, ModelInput, ToolCatalog
from forestcode.context.builder import ContextBuilder
from forestcode.context.providers import SessionContextProvider
from forestcode.context.tool_catalog import ToolVisibilityPolicy
from forestcode.core.abort import AbortSignal
from forestcode.core.agent_loop import AgentLoop
from forestcode.core.events import EventSink, InMemoryEventSink
from forestcode.core.stop_policy import MaxTurnsStopPolicy
from forestcode.core.tool_executor import ToolExecutor
from forestcode.core.turn_processor import TurnProcessor
from forestcode.core.types import ModelClient, ModelOutput
from forestcode.memory import (
    MemoryManager,
    ModelSummarizer,
    SessionCompactionController,
    SessionCompressor,
    SessionRecorder,
    SessionStore,
)
from forestcode.models import (
    DeepSeekAdapter,
    ModelConfig,
    ModelRouter,
    OpenAICompatibleAdapter,
    ProviderRegistry,
    load_model_config_from_env,
)
from forestcode.skills import SkillSnapshot
from forestcode.subagents.child import (
    ConfirmBridgeProtocol,
    combined_context_chars,
    resolve_child_skill_fragments,
)
from forestcode.subagents.config_loader import resolve_child_model_config
from forestcode.subagents.coordinator import ChildRunnerProtocol
from forestcode.subagents.events import SubagentEventSink
from forestcode.subagents.types import (
    MAX_COMBINED_CONTEXT_CHARS,
    AgentConfigSet,
    SubagentRequest,
    SubagentResult,
)
from forestcode.tools import (
    ApprovalRequest,
    ToolRuntimeServices,
    create_builtin_tool_registry,
)
from forestcode.tools.mutation_gate import MutationGate
from forestcode.tools.patch import PatchService
from forestcode.tools.read_state import ReadStateStore
from forestcode.tools.skills import create_load_skill_tool


class ObservedModelClient:
    """CLI-only wrapper that records model inputs without changing model behavior."""

    def __init__(self, inner: ModelClient) -> None:
        self.inner = inner
        self.inputs: list[ModelInput] = []

    def complete(
        self, model_input: ModelInput, *, abort: AbortSignal | None = None
    ) -> ModelOutput:
        self.inputs.append(model_input)
        return self.inner.complete(model_input, abort=abort)


def build_model_client(model_config: ModelConfig) -> ModelRouter:
    registry = ProviderRegistry()
    registry.register("openai-compatible", OpenAICompatibleAdapter())
    registry.register("deepseek", DeepSeekAdapter())
    return ModelRouter(config=model_config, registry=registry)


def build_model_client_from_env() -> ModelRouter:
    """Compatibility/test entry. Main path uses build_model_client(config.model)."""
    return build_model_client(load_model_config_from_env())


def build_agent_loop(
    model: ModelClient,
    workspace_root: Path,
    agent: AgentRuntimeConfig | None = None,
    session_id: str | None = None,
    events: EventSink | None = None,
    enable_write_tools: bool = False,
    runtime: ToolRuntimeServices | None = None,
    session_store: SessionStore | None = None,
    abort: AbortSignal | None = None,
    skills_snapshot: SkillSnapshot | None = None,
    transient_fragments: tuple[ContextFragment, ...] = (),
    subagents: Any | None = None,
    subagent_tools: Iterable[Any] = (),
) -> AgentLoop:
    agent = agent or AgentRuntimeConfig()
    event_sink = events or InMemoryEventSink()
    session_store = session_store or (
        SessionStore(workspace_root) if session_id else None
    )
    tool_registry = create_builtin_tool_registry(
        session_store=session_store,
        session_id=session_id,
        enable_memory_write=agent.features.include_long_term_memory,
    )
    if skills_snapshot is not None and skills_snapshot.descriptors:
        # Only expose load_skill when the catalog is non-empty; otherwise the
        # tool set stays identical to the no-skills behavior.
        tool_registry.register(create_load_skill_tool(skills_snapshot))
    for subagent_tool in subagent_tools:
        tool_registry.register(subagent_tool)
    plan_store = runtime.plan_store if runtime is not None else None
    runtime_internal_dirs: frozenset[Path] = frozenset()
    tool_results_dir: Path | None = None
    runtime_exception_dirs: frozenset[Path] = frozenset()
    if session_store is not None:
        runtime_root = session_store.runtime_root
        workspace = Path(workspace_root).resolve()
        try:
            runtime_root.relative_to(workspace)
            runtime_root_is_private = runtime_root != workspace
        except ValueError:
            runtime_root_is_private = False
        if runtime_root_is_private:
            runtime_internal_dirs = frozenset({runtime_root})
            tool_results_dir = runtime_root / "tool-results"
            runtime_exception_dirs = frozenset({tool_results_dir})
        else:
            warnings.warn(
                f"SessionStore.runtime_root ({runtime_root}) is not inside workspace "
                "or equals workspace root. Skipping runtime isolation injection.",
                stacklevel=2,
            )
    session_provider = SessionContextProvider(session_store) if session_store else None
    memory_manager = (
        MemoryManager(
            SessionRecorder(
                session_store,
                session_id=session_id,
                max_tool_result_store_chars=agent.tool_output_max_chars,
            )
        )
        if session_store and session_id
        else None
    )
    compaction_controller = (
        SessionCompactionController(
            SessionCompressor(
                session_store,
                session_id,
                ModelSummarizer(
                    model, max_tool_result_chars=agent.budget.max_tool_result_chars
                ),
                keep_recent_entries=agent.budget.max_recent_messages,
                compact_trigger_entries=agent.runtime.compact_trigger_entries,
                max_summary_chars=agent.budget.max_session_summary_chars,
                max_tool_result_chars=agent.budget.max_tool_result_chars,
            ),
            auto_compact=agent.runtime.auto_compact,
        )
        if session_store and session_id
        else None
    )
    loop = AgentLoop(
        model=model,
        context_builder=ContextBuilder(
            workspace_root=workspace_root,
            tool_catalog=ToolCatalog(
                tool_registry,
                read_only_only=not enable_write_tools,
                enable_command_tools=agent.features.enable_command_tools,
            ),
            budget=agent.budget,
            request=ContextRequest(
                workspace_root=str(workspace_root),
                session_id=session_id,
                include_project_rules=agent.features.include_project_rules,
                include_long_term_memory=agent.features.include_long_term_memory,
                transient_fragments=transient_fragments,
            ),
            session_provider=session_provider,
            plan_store=plan_store,
        ),
        turn_processor=TurnProcessor(),
        tool_executor=ToolExecutor(
            tool_registry,
            workspace_root=workspace_root,
            max_output_chars=agent.tool_output_max_chars,
            runtime_internal_dirs=runtime_internal_dirs,
            runtime_exception_dirs=runtime_exception_dirs,
            runtime=runtime,
            enable_command_tools=agent.features.enable_command_tools,
            abort=abort,
            session_id=session_id,
            tool_results_dir=tool_results_dir,
        ),
        events=event_sink,
        stop_policy=MaxTurnsStopPolicy(max_turns=agent.runtime.max_turns),
        memory_manager=memory_manager,
        compaction_controller=compaction_controller,
        abort=abort,
        subagents=subagents,
    )
    return loop


def parent_visible_tool_names(
    workspace_root: Path,
    agent: AgentRuntimeConfig,
    *,
    session_store: SessionStore | None = None,
    session_id: str | None = None,
    skills_snapshot: SkillSnapshot | None = None,
    enable_write_tools: bool = False,
) -> frozenset[str]:
    """Names of the tools the parent run would expose (design §Permission Composition).

    The parent ToolCatalog (with the parent's capability flags) is the hard
    ceiling for every child: children can only see a subset of these names.
    This must mirror the registry/catalog assembled in ``build_agent_loop``.
    """
    agent = agent or AgentRuntimeConfig()
    session_store = session_store or (
        SessionStore(workspace_root) if session_id else None
    )
    registry = create_builtin_tool_registry(
        session_store=session_store,
        session_id=session_id,
        enable_memory_write=agent.features.include_long_term_memory,
    )
    if skills_snapshot is not None and skills_snapshot.descriptors:
        registry.register(create_load_skill_tool(skills_snapshot))
    catalog = ToolCatalog(
        registry,
        read_only_only=not enable_write_tools,
        enable_command_tools=agent.features.enable_command_tools,
    )
    return frozenset(tool.name for tool in catalog.list_visible_tools())


def build_subagent_child_runner(
    *,
    workspace_root: Path,
    agent_set: AgentConfigSet,
    parent_model: ModelConfig,
    environ: Callable[[str], str | None],
    skills_snapshot: SkillSnapshot | None,
    activated_skill_names: tuple[str, ...],
    inherited_fragments: tuple[ContextFragment, ...],
    parent_visible_tools: frozenset[str],
    mutation_gate: MutationGate | None,
    confirm_bridge: ConfirmBridgeProtocol | None,
    events: EventSink,
    session_root: Path | None,
    agent: AgentRuntimeConfig,
    approval_started: Callable[[str], None] | None = None,
    approval_finished: Callable[[str], None] | None = None,
    command_service: Any | None = None,
    model_factory: Callable[[ModelConfig], ModelClient] = build_model_client,
) -> ChildRunnerProtocol:
    """Build the per-parent-run child runner (design §Child Construction).

    Returns an object implementing ``ChildRunnerProtocol``; each ``run``
    resolves the agent/model config against the run-fixed snapshots, builds a
    fully independent child object graph (own model router, session store,
    read state, patch service, catalog and recorder) and drives one
    ``AgentLoop`` to completion.

    Runtime isolation: the child's ToolExecutor treats the *whole* workspace
    ``.forestcode/`` as runtime-internal (design §Persistence), not just its own
    transcript subtree, so a child can never read the parent session or another
    child's transcript.
    """

    class _SubagentChildRunner:
        def __init__(self) -> None:
            self._workspace = Path(workspace_root).resolve()
            self._agent_set = agent_set
            self._parent_model = parent_model
            self._environ = environ
            self._skills_snapshot = skills_snapshot
            self._activated_skill_names = activated_skill_names
            self._inherited_fragments = inherited_fragments
            self._parent_visible_tools = frozenset(parent_visible_tools)
            self._mutation_gate = mutation_gate
            self._confirm_bridge = confirm_bridge
            self._approval_started = approval_started
            self._approval_finished = approval_finished
            self._events = events
            self._session_root = (
                Path(session_root).resolve() if session_root is not None else None
            )
            self._agent = agent
            self._command_service = command_service
            # Injectable model construction (tests use a fake; the bridge uses
            # the real ModelRouter). Each child gets its own router instance.
            self._model_factory = model_factory

        def run(
            self, request: SubagentRequest, *, abort: AbortSignal
        ) -> SubagentResult:
            # 1) Agent config against the run-fixed snapshot.
            config = self._agent_set.get(request.agent_name)
            if config is None:
                raise ValueError(f"Unknown subagent: {request.agent_name}")
            # 2) Model config: field-wise inheritance + api_key_env rules (R8).
            #    ModelAdapterError propagates and becomes failed(child_error).
            model_config = resolve_child_model_config(
                self._parent_model, config.model, self._environ
            )
            # 3) Skill fragments: activated ∪ default_skills, dedup by typed
            #    name; a missing/invalid default skill makes the agent invalid.
            fragments = resolve_child_skill_fragments(
                config,
                self._skills_snapshot,
                self._activated_skill_names,
                self._inherited_fragments,
            )
            # 4) Final combined-budget check (never truncates).
            total = combined_context_chars(
                config.instructions, request.prompt, fragments
            )
            if total > MAX_COMBINED_CONTEXT_CHARS:
                raise ValueError(
                    "delegation context too large: agent instructions + prompt + "
                    f"pre-injected skills total {total} characters, exceeding the "
                    f"{MAX_COMBINED_CONTEXT_CHARS}-character limit"
                )
            # 5) Independent child object graph (design §Child Construction).
            child_session = (
                SessionStore(self._workspace, session_dir=self._session_root)
                if self._session_root is not None
                else None
            )
            child_read_state = ReadStateStore()
            child_patch = PatchService(read_state_store=child_read_state)
            child_registry = create_builtin_tool_registry(
                session_store=child_session,
                session_id=request.task_id,
                enable_memory_write=self._agent.features.include_long_term_memory,
            )
            if self._skills_snapshot is not None and self._skills_snapshot.descriptors:
                child_registry.register(create_load_skill_tool(self._skills_snapshot))
            # Visibility: parent catalog is the hard ceiling; profile + allow/
            # deny only filter what the child sees (design §Permission
            # Composition). Subagent tools are structurally absent.
            catalog = ToolCatalog(
                child_registry,
                read_only_only=False,
                enable_command_tools=self._agent.features.enable_command_tools,
                visibility=ToolVisibilityPolicy(
                    self._parent_visible_tools,
                    config.permission_profile,
                    config.tools.allow,
                    config.tools.deny,
                ),
            )
            child_confirm: Callable[[ApprovalRequest], bool] | None = None
            bridge = self._confirm_bridge
            if bridge is not None:
                task_id = request.task_id

                def _child_confirm(req: ApprovalRequest) -> bool:
                    if self._approval_started is not None:
                        self._approval_started(task_id)
                    try:
                        return bridge(req, task_id=task_id, abort=abort)
                    finally:
                        if self._approval_finished is not None:
                            self._approval_finished(task_id)

                child_confirm = _child_confirm
            child_runtime = ToolRuntimeServices(
                read_state_store=child_read_state,
                patch_service=child_patch,
                command_service=self._command_service,
                plan_store=None,
                confirm=child_confirm,
                mutation_gate=self._mutation_gate,
            )
            # Runtime isolation: the whole workspace .forestcode/ is internal;
            # only tool-results stays readable (existing controlled exception).
            forestcode_root = self._workspace / ".forestcode"
            runtime_internal: frozenset[Path] = frozenset()
            tool_results_dir: Path | None = None
            runtime_exceptions: frozenset[Path] = frozenset()
            try:
                forestcode_root.relative_to(self._workspace)
                # Protect the namespace even before it exists.  Otherwise a
                # sessionless edit/full child could create .forestcode itself
                # and only later runs would treat it as runtime-internal.
                runtime_internal = frozenset({forestcode_root})
                tool_results_dir = forestcode_root / "tool-results"
                runtime_exceptions = frozenset({tool_results_dir})
            except ValueError:
                pass
            child_executor = ToolExecutor(
                child_registry,
                workspace_root=self._workspace,
                max_output_chars=self._agent.tool_output_max_chars,
                runtime_internal_dirs=runtime_internal,
                runtime_exception_dirs=runtime_exceptions,
                runtime=child_runtime,
                enable_command_tools=self._agent.features.enable_command_tools,
                abort=abort,
                session_id=request.task_id,
                tool_results_dir=tool_results_dir,
            )
            child_session_provider = (
                SessionContextProvider(child_session)
                if child_session is not None
                else None
            )
            child_recorder = (
                MemoryManager(
                    SessionRecorder(
                        child_session,
                        session_id=request.task_id,
                        max_tool_result_store_chars=self._agent.tool_output_max_chars,
                    )
                )
                if child_session is not None
                else None
            )
            child_loop = AgentLoop(
                model=self._model_factory(model_config),
                context_builder=ContextBuilder(
                    workspace_root=self._workspace,
                    tool_catalog=catalog,
                    budget=self._agent.budget,
                    request=ContextRequest(
                        workspace_root=str(self._workspace),
                        session_id=request.task_id,
                        include_project_rules=self._agent.features.include_project_rules,
                        include_long_term_memory=False,
                        transient_fragments=fragments,
                    ),
                    session_provider=child_session_provider,
                    plan_store=None,
                    system_prefix=config.instructions,
                ),
                turn_processor=TurnProcessor(),
                tool_executor=child_executor,
                events=SubagentEventSink(
                    self._events,
                    task_id=request.task_id,
                    agent_name=request.agent_name,
                ),
                stop_policy=MaxTurnsStopPolicy(max_turns=self._agent.runtime.max_turns),
                memory_manager=child_recorder,
                compaction_controller=None,
                abort=abort,
            )
            # 6) Drive the child loop; the coordinator maps the returned state to
            #    completed/failed and handles late/aborted results.
            state = child_loop.run(request.prompt)
            if state.final_text is None:
                raise ValueError(state.error or "child run ended without a result")
            return SubagentResult(
                task_id=request.task_id,
                agent_name=request.agent_name,
                final_text=state.final_text,
                turn_count=state.turns,
                tool_count=len(state.tool_results),
            )

    return _SubagentChildRunner()
