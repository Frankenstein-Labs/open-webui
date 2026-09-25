"""Bridge between CORTEX Web's inference gateway and the CORTEX orchestrator.

Feature-flagged and opt-in: when `CORTEX_AGENT_ORCHESTRATOR` is off (default),
nothing here runs and the existing single-engine behaviour is untouched. When
on, Agent-mode requests are routed by capability through the registry and, if
needed, chained across engines.

This is the only place the orchestrator is wired into the web app, so the rest
of the integration can evolve without touching chat middleware.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from open_webui.inference.cortex.adapters.ai_manus import AiManusAdapter
from open_webui.inference.cortex.adapters.openhands import OpenHandsAdapter
from open_webui.inference.cortex.capabilities import Capability, EngineContext, EngineRegistry
from open_webui.inference.cortex.computers import get_computer_registry
from open_webui.inference.cortex.orchestrator import CortexOrchestrator, describe_registry
from open_webui.inference.cortex.policy import CortexPolicy
from open_webui.inference.cortex.routing import derive_requirements, plan_route

log = logging.getLogger(__name__)

_registry_lock = threading.Lock()
_registry: EngineRegistry | None = None


def orchestrator_enabled() -> bool:
    return os.getenv('CORTEX_AGENT_ORCHESTRATOR', 'false').lower() == 'true'


def chaining_enabled() -> bool:
    return os.getenv('CORTEX_AGENT_CHAINING', 'true').lower() == 'true'


def build_registry() -> EngineRegistry:
    """Create the default registry: OpenHands for software work, ai-manus for
    computer/browser/screen work."""
    registry = EngineRegistry()
    registry.register(OpenHandsAdapter(enable_browser=os.getenv('CORTEX_OPENHANDS_BROWSER', 'true').lower() == 'true'))
    registry.register(AiManusAdapter())
    return registry


def get_registry() -> EngineRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = build_registry()
            log.info('CORTEX engine registry initialised with engines: %s', _registry.list())
        return _registry


def reset_registry() -> None:
    """Test hook: drop the cached registry."""
    global _registry
    with _registry_lock:
        _registry = None


def get_orchestrator() -> CortexOrchestrator:
    return CortexOrchestrator(
        get_registry(),
        CortexPolicy.from_env(),
        computers=get_computer_registry() if _computers_enabled() else None,
    )


def _computers_enabled() -> bool:
    """The shared computer plane is opt-in, like the orchestrator itself."""
    return os.getenv('CORTEX_COMPUTER_ENABLED', 'false').lower() == 'true'


def describe_engines() -> dict[str, Any]:
    from open_webui.inference.cortex.providers import ProviderConfigError, resolve_provider_config

    described = describe_registry(get_registry())
    try:
        # to_metadata() deliberately excludes the credential.
        described['llmProvider'] = resolve_provider_config().to_metadata()
    except ProviderConfigError as exc:
        described['llmProvider'] = {'error': str(exc)}
    return described


def plan_for(form_data: dict[str, Any]) -> dict[str, Any]:
    """Dry-run routing for diagnostics: no engine is executed."""
    plan = plan_route(
        get_registry(),
        form_data,
        allow_chaining=chaining_enabled(),
    )
    return {
        'enabled': orchestrator_enabled(),
        'chaining': chaining_enabled(),
        'requirements': derive_requirements(form_data).to_dict(),
        'plan': plan.to_dict(),
    }


def _build_context(form_data: dict[str, Any], user: Any = None) -> EngineContext:
    metadata = form_data.get('metadata') or {}
    return EngineContext(
        workspace_id=str(metadata.get('workspace_id') or ''),
        user_id=str(getattr(user, 'id', '') or ''),
        chat_id=str(metadata.get('chat_id') or form_data.get('chat_id') or ''),
        policy_id=str(metadata.get('policy_id') or os.getenv('CORTEX_POLICY_ID', '')),
        metadata={'workspace': str(metadata.get('workspace') or '')} if metadata.get('workspace') else {},
    )


async def route_agent_completion(
    form_data: dict[str, Any],
    user: Any = None,
    *,
    stream: bool = False,
) -> dict[str, Any] | Any:
    """Route an Agent-mode request through the orchestrator."""
    from open_webui.inference.cortex.adapters.base import openai_response

    orchestrator = get_orchestrator()
    text, _events, routing = await orchestrator.collect(
        form_data,
        context=_build_context(form_data, user),
        allow_chaining=chaining_enabled(),
    )
    engine = str(routing.get('primary') or 'cortex')
    result = openai_response(str(form_data.get('model') or engine), text, engine, routing=routing)
    if not stream:
        return result

    import json
    from collections.abc import AsyncIterator

    async def chunks() -> AsyncIterator[str]:
        yield f'data: {json.dumps(result)}\n\n'
        yield 'data: [DONE]\n\n'

    return chunks()


def should_orchestrate(form_data: dict[str, Any]) -> bool:
    """True when the orchestrator should handle this request instead of the
    legacy single-engine path."""
    if not orchestrator_enabled():
        return False
    metadata = form_data.get('metadata') or {}
    mode = str(metadata.get('conversation_mode') or form_data.get('conversation_mode') or '').lower()
    if mode == 'agent':
        return True
    # Outside Agent mode, only take over when the task genuinely needs a
    # capability no conversational engine provides.
    requirements = derive_requirements(form_data).capabilities
    task_capabilities = {
        Capability.BROWSER,
        Capability.SCREEN,
        Capability.COMPUTER,
        Capability.TERMINAL,
        Capability.CODE,
        Capability.WEB_SEARCH,
    }
    return bool(requirements & task_capabilities)
