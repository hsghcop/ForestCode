import unittest

from forestcode.context import ToolCatalog
from forestcode.tools import ToolDefinition, ToolRegistry


class ToolCatalogTest(unittest.TestCase):
    def test_exports_only_read_only_tool_schemas_by_default(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="read_file",
                description="Read a file.",
                input_schema={"type": "object"},
                runner=lambda _context: "ok",
            )
        )
        registry.register(
            ToolDefinition(
                name="write_file",
                description="Write a file.",
                input_schema={"type": "object"},
                runner=lambda _context: "ok",
                risk_level="write",
                is_read_only=False,
            )
        )

        catalog = ToolCatalog(registry)

        schemas = catalog.list_model_schemas()
        self.assertEqual([schema["function"]["name"] for schema in schemas], ["read_file"])
        self.assertEqual(catalog.selected_tool_names(), ["read_file"])

    def test_write_mode_does_not_export_command_tools_by_default(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="read_file",
                description="Read a file.",
                input_schema={"type": "object"},
                runner=lambda _context: "ok",
            )
        )
        registry.register(
            ToolDefinition(
                name="write_file",
                description="Write a file.",
                input_schema={"type": "object"},
                runner=lambda _context: "ok",
                risk_level="write",
                is_read_only=False,
            )
        )
        registry.register(
            ToolDefinition(
                name="run_command",
                description="Run a command.",
                input_schema={"type": "object"},
                runner=lambda _context: "ok",
                risk_level="command",
                is_read_only=False,
            )
        )

        catalog = ToolCatalog(registry, read_only_only=False)

        self.assertEqual(catalog.selected_tool_names(), ["read_file", "write_file"])

    def test_command_tools_require_explicit_enable_flag(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="run_command",
                description="Run a command.",
                input_schema={"type": "object"},
                runner=lambda _context: "ok",
                risk_level="command",
                is_read_only=False,
            )
        )

        catalog = ToolCatalog(registry, enable_command_tools=True)

        self.assertEqual(catalog.selected_tool_names(), ["run_command"])


if __name__ == "__main__":
    unittest.main()
