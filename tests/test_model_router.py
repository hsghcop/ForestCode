import unittest

from forestcode.core.types import Message, ModelInput, ModelOutput
from forestcode.models import ModelAdapterError, ModelConfig, ModelRouter, ProviderRegistry


class RecordingAdapter:
    def __init__(self, output: ModelOutput):
        self.output = output
        self.calls = []

    def complete(self, config: ModelConfig, model_input: ModelInput) -> ModelOutput:
        self.calls.append((config, model_input))
        return self.output


class ModelRouterTest(unittest.TestCase):
    def test_registry_registers_and_gets_adapter(self):
        registry = ProviderRegistry()
        adapter = RecordingAdapter(ModelOutput(text="ok"))

        registry.register("openai-compatible", adapter)

        self.assertIs(registry.get("openai-compatible"), adapter)

    def test_registry_rejects_empty_api_type(self):
        registry = ProviderRegistry()

        with self.assertRaisesRegex(ModelAdapterError, "api_type"):
            registry.register(" ", RecordingAdapter(ModelOutput(text="ok")))

    def test_registry_reports_missing_adapter(self):
        registry = ProviderRegistry()

        with self.assertRaisesRegex(ModelAdapterError, "not-registered"):
            registry.get("not-registered")

    def test_router_calls_registered_adapter(self):
        output = ModelOutput(text="done")
        adapter = RecordingAdapter(output)
        registry = ProviderRegistry()
        registry.register("openai-compatible", adapter)
        config = ModelConfig(
            api_type="openai-compatible",
            model="deepseek-chat",
            base_url="https://api.deepseek.com/v1",
            api_key="secret",
        )
        router = ModelRouter(config=config, registry=registry)
        model_input = ModelInput(messages=[Message(role="user", content="hello")])

        result = router.complete(model_input)

        self.assertIs(result, output)
        self.assertEqual(adapter.calls, [(config, model_input)])


if __name__ == "__main__":
    unittest.main()
