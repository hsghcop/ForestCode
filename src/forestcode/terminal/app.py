"""ForestCodeCliApp: the interactive main loop (§16).

Reads input on the main thread, classifies it via ``BackendBridge.classify``,
and runs model turns through ``TurnRunner`` (worker thread for backend, UI on the
main thread — plan §7.2). Idle Ctrl+C / EOF exit here; a Ctrl+C *during a turn*
is handled inside ``TurnRunner`` (cancel the turn, stay in the loop).
"""

from __future__ import annotations

from dataclasses import replace

from .bridge import BackendBridge
from .input import InputController
from .renderer import FrontendState, TerminalRenderer
from .turn_runner import TurnRunner


class ForestCodeCliApp:
    def __init__(
        self,
        bridge: BackendBridge,
        renderer: TerminalRenderer,
        input_controller: InputController,
        frontend_state: FrontendState,
    ) -> None:
        self._bridge = bridge
        self._renderer = renderer
        self._input = input_controller
        self._state = frontend_state
        self._turn_runner = TurnRunner(renderer, bridge.confirm_controller)

    def run(self) -> int:
        self._renderer.render_welcome(self._state)
        while True:
            try:
                prompt = self._renderer.render_prompt(self._state)
                text = self._input.read_user_input(prompt)
                decision = self._bridge.classify(text)
            except EOFError:
                self._renderer.newline()
                return 0
            except KeyboardInterrupt:
                # Idle Ctrl+C (at the prompt) or during an inline slash command.
                self._renderer.render_system_error("Interrupted.")
                return 130

            if decision.session_changed:
                self._state = replace(self._state, session_id=decision.session_id)
            if decision.kind == "exit":
                return decision.exit_code
            if decision.kind == "noop":
                continue

            # decision.kind == "run": hand the model task to the TurnRunner.
            outcome = self._turn_runner.run(
                self._bridge,
                decision.task or "",
                transient_fragments=decision.transient_fragments,
                skills_snapshot=decision.skills_snapshot,
                launch_context=decision.launch_context,
            )
            if outcome.session_changed:
                self._state = replace(self._state, session_id=outcome.session_id)
            if outcome.action == "exit":
                return outcome.exit_code
            if outcome.action == "error":
                self._renderer.render_system_error(outcome.error or "unknown error")
                return outcome.exit_code or 1
