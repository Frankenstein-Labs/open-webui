"""Tests for the CORTEX bridge feature flags and diagnostics.

The whole integration is opt-in, so these tests are primarily about *not*
activating it accidentally, and about the diagnostics endpoint shape.
"""

import sys
from pathlib import Path

import pytest

# Locate the backend root by walking up to the directory holding the package,
# so the suite passes whether pytest is run from backend/ or the repo root.
BACKEND = next(parent for parent in Path(__file__).resolve().parents if (parent / 'open_webui').is_dir())
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from open_webui.inference.cortex import bridge  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv('CORTEX_AGENT_ORCHESTRATOR', raising=False)
    monkeypatch.delenv('CORTEX_AGENT_CHAINING', raising=False)
    bridge.reset_registry()
    yield
    bridge.reset_registry()


def test_orchestrator_is_disabled_by_default():
    assert bridge.orchestrator_enabled() is False
    assert bridge.should_orchestrate({'conversation_mode': 'agent', 'messages': []}) is False


def test_agent_mode_is_taken_over_when_enabled(monkeypatch):
    monkeypatch.setenv('CORTEX_AGENT_ORCHESTRATOR', 'true')
    assert bridge.should_orchestrate({'conversation_mode': 'agent', 'messages': []}) is True


def test_discussion_mode_is_left_to_the_legacy_path(monkeypatch):
    monkeypatch.setenv('CORTEX_AGENT_ORCHESTRATOR', 'true')
    form_data = {'conversation_mode': 'discussion', 'messages': [{'role': 'user', 'content': 'hello there'}]}
    assert bridge.should_orchestrate(form_data) is False


def test_mode_read_from_metadata(monkeypatch):
    monkeypatch.setenv('CORTEX_AGENT_ORCHESTRATOR', 'true')
    form_data = {'metadata': {'conversation_mode': 'agent'}, 'messages': []}
    assert bridge.should_orchestrate(form_data) is True


def test_explicit_task_capability_is_taken_over_outside_agent_mode(monkeypatch):
    monkeypatch.setenv('CORTEX_AGENT_ORCHESTRATOR', 'true')
    form_data = {'messages': [{'role': 'user', 'content': 'take a screenshot of the website'}]}
    assert bridge.should_orchestrate(form_data) is True


def test_plain_conversation_is_not_taken_over(monkeypatch):
    monkeypatch.setenv('CORTEX_AGENT_ORCHESTRATOR', 'true')
    form_data = {'messages': [{'role': 'user', 'content': 'tell me a joke'}]}
    assert bridge.should_orchestrate(form_data) is False


def test_chaining_defaults_to_enabled(monkeypatch):
    assert bridge.chaining_enabled() is True
    monkeypatch.setenv('CORTEX_AGENT_CHAINING', 'false')
    assert bridge.chaining_enabled() is False


def test_registry_exposes_both_engines():
    registry = bridge.get_registry()
    assert registry.list() == ['ai-manus', 'openhands']


def test_registry_is_cached_between_calls():
    assert bridge.get_registry() is bridge.get_registry()


def test_plan_for_is_a_dry_run():
    plan = bridge.plan_for(
        {'conversation_mode': 'agent', 'messages': [{'role': 'user', 'content': 'open the browser'}]}
    )
    assert plan['enabled'] is False  # flag off by default
    assert plan['plan']['primary'] == 'ai-manus'
    assert plan['requirements']['mode'] == 'agent'


def test_describe_engines_reports_capabilities():
    described = bridge.describe_engines()
    engines = {entry['engine']: entry for entry in described['engines']}
    assert 'browser' in engines['ai-manus']['tools']
    assert 'screen' in engines['ai-manus']['tools']
    assert 'mcp' in engines['openhands']['tools']
    assert described['capabilities']


def test_describe_engines_reports_provider_without_the_key(monkeypatch):
    monkeypatch.setenv('CORTEX_LLM_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-v1-should-not-leak')
    described = bridge.describe_engines()
    assert described['llmProvider']['provider'] == 'openrouter'
    assert described['llmProvider']['hasApiKey'] is True
    assert 'should-not-leak' not in str(described)
    assert 'apiKey' not in described['llmProvider']


def test_describe_engines_reports_provider_error_instead_of_raising(monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.delenv('OPENHANDS_API_KEY', raising=False)
    described = bridge.describe_engines()
    assert 'error' in described['llmProvider']


def test_route_agent_completion_returns_openai_shaped_result(monkeypatch):
    """Full gateway path: form_data -> orchestrator -> OpenAI-style response."""
    import asyncio

    from open_webui.inference.cortex.adapters.base import BaseEngineAdapter
    from open_webui.inference.cortex.capabilities import AgentCapability, Capability, EngineRegistry
    from open_webui.inference.cortex.orchestrator import CortexOrchestrator

    class ScriptedAdapter(BaseEngineAdapter):
        name = 'openhands'

        def __init__(self) -> None:
            super().__init__()
            self._capabilities = AgentCapability(
                engine=self.name,
                tools=frozenset({Capability.CHAT, Capability.TERMINAL, Capability.FILES}),
            )

        async def _stream(self, session, instruction, context):  # type: ignore[no-untyped-def]
            yield self.message(session, 'task finished')

    registry = EngineRegistry()
    registry.register(ScriptedAdapter())
    monkeypatch.setattr(bridge, 'get_orchestrator', lambda: CortexOrchestrator(registry))

    result = asyncio.run(
        bridge.route_agent_completion(
            {'conversation_mode': 'agent', 'model': 'cortex', 'messages': [{'role': 'user', 'content': 'go'}]},
            None,
        )
    )
    assert result['object'] == 'chat.completion'
    assert result['choices'][0]['message']['content'] == 'task finished'
    assert result['engine'] == 'openhands'
    assert result['routing']['primary'] == 'openhands'
