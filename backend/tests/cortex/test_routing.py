"""Tests for the CORTEX capability router and engine registry contracts.

These exercise the Python mirror of `packages/engine-adapters` and
`packages/core` from CORTEX-Software-Agent-SDK against the real adapter
capability declarations used by the web app.
"""

import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from open_webui.inference.cortex.adapters.ai_manus import AiManusAdapter  # noqa: E402
from open_webui.inference.cortex.adapters.openhands import OpenHandsAdapter  # noqa: E402
from open_webui.inference.cortex.capabilities import (  # noqa: E402
    Capability,
    EngineRegistry,
)
from open_webui.inference.cortex.routing import derive_requirements, plan_route  # noqa: E402


def build_registry(*, openhands_browser: bool = True) -> EngineRegistry:
    registry = EngineRegistry()
    registry.register(OpenHandsAdapter(enable_browser=openhands_browser))
    registry.register(AiManusAdapter())
    return registry


def test_registry_rejects_duplicate_engines():
    registry = build_registry()
    try:
        registry.register(OpenHandsAdapter())
    except ValueError as exc:
        assert 'already registered' in str(exc)
    else:  # pragma: no cover
        raise AssertionError('duplicate registration must fail')


def test_registry_lists_declared_capabilities():
    registry = build_registry()
    assert registry.list() == ['ai-manus', 'openhands']
    capabilities = registry.capabilities()
    assert Capability.BROWSER in capabilities['ai-manus'].tools
    assert Capability.SCREEN in capabilities['ai-manus'].tools
    assert Capability.COMPUTER in capabilities['ai-manus'].tools
    assert Capability.TERMINAL in capabilities['openhands'].tools
    assert Capability.MCP in capabilities['openhands'].tools


def test_ai_manus_alone_covers_screen_and_computer():
    registry = build_registry()
    assert registry.engines_supporting({Capability.SCREEN, Capability.COMPUTER}) == ['ai-manus']


def test_code_task_routes_to_openhands():
    registry = build_registry()
    plan = plan_route(
        registry,
        {'messages': [{'role': 'user', 'content': 'refactor the repository and open a pull request'}]},
    )
    assert plan.primary is not None
    assert plan.primary.engine == 'openhands'


def test_browser_task_routes_to_ai_manus():
    registry = build_registry()
    plan = plan_route(
        registry,
        {'messages': [{'role': 'user', 'content': 'open a browser and take a screenshot of the page'}]},
    )
    assert plan.primary is not None
    assert plan.primary.engine == 'ai-manus'
    assert Capability.SCREEN in plan.primary.capabilities


def test_agent_mode_requires_execution_capabilities_without_keywords():
    requirements = derive_requirements({'conversation_mode': 'agent', 'messages': [{'role': 'user', 'content': 'go'}]})
    assert Capability.TERMINAL in requirements.capabilities
    assert Capability.CODE not in requirements.capabilities


def test_agent_mode_task_chains_engines_when_no_single_engine_covers_all():
    # No engine covers this set alone: ai-manus owns the virtual screen, while
    # MCP and sub-agent delegation are OpenHands-only. The router must chain.
    registry = build_registry()
    plan = plan_route(
        registry,
        {'conversation_mode': 'agent', 'messages': [{'role': 'user', 'content': 'do the work'}]},
        required={Capability.TERMINAL, Capability.SCREEN, Capability.MCP, Capability.DELEGATION},
        allow_chaining=True,
    )
    assert plan.is_chained
    assert plan.engine_names[0] == 'openhands'
    assert plan.engine_names[1] == 'ai-manus'
    assert Capability.SCREEN in plan.selections[1].capabilities
    assert not plan.unsupported


def test_chaining_can_be_disabled():
    registry = build_registry()
    plan = plan_route(
        registry,
        {'conversation_mode': 'agent', 'messages': [{'role': 'user', 'content': 'do the work'}]},
        required={Capability.TERMINAL, Capability.SCREEN, Capability.MCP, Capability.DELEGATION},
        allow_chaining=False,
    )
    assert not plan.is_chained
    assert Capability.SCREEN in plan.unsupported


def test_router_uses_capabilities_not_engine_names():
    """A registry of a single engine must still route by capability coverage."""

    class OnlyBrowser(OpenHandsAdapter):
        name = 'only-browser'

        def __init__(self) -> None:
            super().__init__()
            from open_webui.inference.cortex.capabilities import AgentCapability

            self._capabilities = AgentCapability(
                engine='only-browser',
                tools=frozenset({Capability.CHAT, Capability.BROWSER}),
            )

    registry = EngineRegistry()
    registry.register(OnlyBrowser())
    plan = plan_route(registry, {'messages': [{'role': 'user', 'content': 'browse the website'}]})
    assert plan.primary is not None
    assert plan.primary.engine == 'only-browser'


def test_openhands_declares_browser_only_when_enabled():
    with_browser = OpenHandsAdapter(enable_browser=True).get_capabilities()
    without_browser = OpenHandsAdapter(enable_browser=False).get_capabilities()
    assert Capability.BROWSER in with_browser.tools
    assert Capability.BROWSER not in without_browser.tools


def test_unknown_capability_is_reported_not_dropped():
    class ChatOnly(OpenHandsAdapter):
        name = 'chat-only'

        def __init__(self) -> None:
            super().__init__()
            from open_webui.inference.cortex.capabilities import AgentCapability

            self._capabilities = AgentCapability(engine='chat-only', tools=frozenset({Capability.CHAT}))

    registry = EngineRegistry()
    registry.register(ChatOnly())
    plan = plan_route(
        registry,
        {'messages': [{'role': 'user', 'content': 'take a screenshot in the browser'}]},
        allow_chaining=False,
    )
    assert Capability.BROWSER in plan.unsupported


def test_env_flag_defaults_are_off():
    # Guard rail: the bridge must never orchestrate unless explicitly enabled.
    os.environ.pop('CORTEX_AGENT_ORCHESTRATOR', None)
    from open_webui.inference.cortex.bridge import orchestrator_enabled

    assert orchestrator_enabled() is False
