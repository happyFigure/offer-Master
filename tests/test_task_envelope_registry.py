import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))


class TaskEnvelopeRegistryTest(unittest.TestCase):
    def test_default_registry_resolves_browser_application_contract(self) -> None:
        from app.agent_runtime.contracts.registry import default_task_envelope_registry
        from app.agent_runtime.contracts.tasks.browser_application import BrowserExecutionResult, BrowserTaskEnvelope
        from app.agent_runtime.tool_registry import APPLICATION_FIND_APPLY_ENTRY_TOOL

        entry = default_task_envelope_registry().get_by_capability(APPLICATION_FIND_APPLY_ENTRY_TOOL)

        self.assertEqual("browser.prepare_application", entry.task_type)
        self.assertIs(BrowserTaskEnvelope, entry.envelope_cls)
        self.assertIs(BrowserExecutionResult, entry.result_cls)
        self.assertEqual("codex_or_multica", entry.default_executor)

    def test_registry_rejects_duplicate_capability(self) -> None:
        from app.agent_runtime.contracts.base import ExecutionResultBase, TaskEnvelopeBase
        from app.agent_runtime.contracts.registry import TaskEnvelopeRegistry

        registry = TaskEnvelopeRegistry()
        registry.register(
            capability="test.capability",
            task_type="test.task",
            envelope_cls=TaskEnvelopeBase,
            result_cls=ExecutionResultBase,
            default_executor="test-executor",
        )

        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(
                capability="test.capability",
                task_type="test.other",
                envelope_cls=TaskEnvelopeBase,
                result_cls=ExecutionResultBase,
                default_executor="test-executor",
            )


if __name__ == "__main__":
    unittest.main()
