import io
import tempfile
import unittest
from pathlib import Path

from forestcode.config import AgentRuntimeConfig
from forestcode.core import FakeModelClient
from forestcode.memory import SessionStore
from forestcode.plan import PlanStore
from forestcode.slash_commands import SlashCommand, SlashContext, SlashResult, looks_like_command
from forestcode.tools import ToolRuntimeServices


def _handler(_ctx, _args):
    return SlashResult()


class SlashCommandRegistryTest(unittest.TestCase):
    def test_register_rejects_duplicate_name(self):
        from forestcode.slash_commands import SlashCommandRegistry

        registry = SlashCommandRegistry()
        registry.register(SlashCommand("quit", "quit", _handler))

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(SlashCommand("quit", "quit again", _handler))

    def test_register_rejects_alias_conflicts(self):
        from forestcode.slash_commands import SlashCommandRegistry

        registry = SlashCommandRegistry()
        registry.register(SlashCommand("quit", "quit", _handler, aliases=["q"]))

        with self.assertRaisesRegex(ValueError, "alias conflicts"):
            registry.register(SlashCommand("other", "other", _handler, aliases=["q"]))
        with self.assertRaisesRegex(ValueError, "alias conflicts"):
            registry.register(SlashCommand("other2", "other", _handler, aliases=["quit"]))
        with self.assertRaisesRegex(ValueError, "conflicts with alias"):
            registry.register(SlashCommand("q", "q", _handler))

    def test_register_requires_lowercase_names_and_aliases(self):
        from forestcode.slash_commands import SlashCommandRegistry

        registry = SlashCommandRegistry()
        with self.assertRaisesRegex(ValueError, "lowercase"):
            registry.register(SlashCommand("Quit", "quit", _handler))
        with self.assertRaisesRegex(ValueError, "lowercase"):
            registry.register(SlashCommand("quit", "quit", _handler, aliases=["Q"]))

    def test_get_by_name_alias_and_list_hidden(self):
        from forestcode.slash_commands import SlashCommandRegistry

        registry = SlashCommandRegistry()
        quit_command = SlashCommand("quit", "quit", _handler, aliases=["q"])
        hidden = SlashCommand("hidden", "hidden", _handler, is_hidden=True)
        registry.register(quit_command)
        registry.register(hidden)

        self.assertIs(registry.get("quit"), quit_command)
        self.assertIs(registry.get("Quit"), quit_command)
        self.assertIs(registry.get("q"), quit_command)
        self.assertIsNone(registry.get("missing"))
        self.assertEqual([command.name for command in registry.list()], ["quit"])
        self.assertEqual([command.name for command in registry.list(include_hidden=True)], ["hidden", "quit"])

    def test_looks_like_command(self):
        for value in ["compact", "a", "a-b", "a_b", "a1", "Exit", "Quit"]:
            self.assertTrue(looks_like_command(value), value)
        for value in ["", "a/b", "a.b", "a b", "a?", "1abc", "x" * 40]:
            self.assertFalse(looks_like_command(value), value)

    def test_slash_result_defaults(self):
        result = SlashResult()

        self.assertEqual(result.action, "continue")
        self.assertEqual(result.exit_code, 0)
        self.assertIsNone(result.prompt_text)

    def test_context_keeps_resources_isolated(self):
        from forestcode.slash_handlers import build_builtin_slash_registry

        with tempfile.TemporaryDirectory() as tmp:
            ctx = SlashContext(
                workspace_root=Path(tmp),
                session_id=None,
                session_store=SessionStore(tmp),
                plan_store=PlanStore(),
                runtime=ToolRuntimeServices(),
                agent=AgentRuntimeConfig(),
                model=FakeModelClient([]),
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                input_func=lambda prompt: "",
                registry=build_builtin_slash_registry(),
            )

            self.assertIsNone(ctx.session_id)
            self.assertIsNotNone(ctx.session_store)
            self.assertIsNotNone(ctx.registry.get("exit"))


if __name__ == "__main__":
    unittest.main()
