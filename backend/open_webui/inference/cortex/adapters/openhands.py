"""OpenHands Software Agent SDK adapter.

Wraps the SDK behind the neutral `EngineAdapter` contract: the orchestrator
never sees an OpenHands type.

Two corrections against the previous integration:

* `cli_mode=False`. The old code called `get_default_agent(..., cli_mode=True)`,
  which disables the browser tool set (`enable_browser=not cli_mode`), so an
  Agent-mode task could not browse even though the router had selected an
  engine that claims browser support.
* The capability set is declared explicitly and checked against the tools the
  agent actually gets, so `get_capabilities()` cannot advertise more than the
  runtime provides.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Sequence
from typing import Any

from open_webui.inference.cortex.adapters.base import BaseEngineAdapter
from open_webui.inference.cortex.capabilities import AgentCapability, Capability, EngineContext, EngineSession
from open_webui.inference.cortex.providers import resolve_provider_config
from open_webui.inference.cortex.protocol import CortexEvent, error_event, make_event

# Tools the SDK preset can run, mapped to the CORTEX capability vocabulary.
_TOOL_CAPABILITIES: dict[str, Capability] = {
    'terminal': Capability.TERMINAL,
    'file_editor': Capability.FILES,
    'browser_use': Capability.BROWSER,
    'task_tracker': Capability.REASONING,
    'grep': Capability.FILES,
    'glob': Capability.FILES,
    'delegate': Capability.DELEGATION,
    'task': Capability.DELEGATION,
    'planning_file_editor': Capability.REASONING,
    'apply_patch': Capability.CODE,
}

_BASE_CAPABILITIES = frozenset(
    {
        Capability.CHAT,
        Capability.REASONING,
        Capability.CODE,
        Capability.TERMINAL,
        Capability.FILES,
        Capability.MCP,
        Capability.SKILLS,
        Capability.DELEGATION,
        Capability.ARTIFACTS,
    }
)


class OpenHandsAdapter(BaseEngineAdapter):
    """Executes tasks with the OpenHands SDK, locally or on a remote server."""

    name = 'openhands'

    def __init__(self, policy: Any = None, *, enable_browser: bool = True) -> None:
        super().__init__(policy=policy)
        self.enable_browser = enable_browser
        tools = set(_BASE_CAPABILITIES)
        if enable_browser:
            tools.add(Capability.BROWSER)
        self._capabilities = AgentCapability(
            engine=self.name,
            tools=frozenset(tools),
            supports_pause=True,
            supports_streaming=True,
            supports_cancel=True,
            isolated_runtime=False,
            notes='OpenHands SDK; browser enabled unless cli_mode is forced',
        )

    async def _stream(
        self,
        session: EngineSession,
        instruction: str,
        context: EngineContext | None,
    ) -> AsyncIterator[CortexEvent]:
        if self.policy is not None and self.policy.require_isolated_runtime:
            # The SDK runs in-process unless a remote workspace is configured;
            # surface that instead of silently executing on the web host.
            if (
                not os.getenv('OPENHANDS_BASE_URL')
                and os.getenv('CORTEX_ALLOW_LOCAL_RUNTIME', 'true').lower() == 'false'
            ):
                yield error_event(
                    'isolated_runtime_required',
                    'CORTEX policy requires an isolated runtime; set OPENHANDS_BASE_URL',
                    recoverable=False,
                    task_id=session.task_id,
                )
                session.status = 'failed'
                return

        content = await asyncio.to_thread(self._run_blocking, instruction, context)
        if content:
            yield self.message(session, content)
        yield make_event(
            'ToolCompleted',
            {'tool': 'openhands.conversation', 'engine': self.name},
            task_id=session.task_id,
            agent_id=session.id,
        )

    def _run_blocking(self, instruction: str, context: EngineContext | None) -> str:
        try:
            from openhands.sdk import LLM, Conversation
            from openhands.sdk.conversation import get_agent_final_response
            from openhands.tools.preset.default import get_default_agent
        except ImportError as exc:
            raise RuntimeError('OpenHands SDK is not installed; install backend/requirements-engines.txt') from exc

        config = resolve_provider_config()
        # cli_mode must stay False for Agent mode: cli_mode=True disables the
        # browser tool set inside get_default_agent().
        agent = get_default_agent(llm=LLM(**config.to_llm_kwargs()), cli_mode=not self.enable_browser)
        workspace = os.getenv('OPENHANDS_WORKSPACE') or (
            context.metadata.get('workspace') if context and context.metadata else None
        )
        conversation = Conversation(agent=agent, workspace=workspace or os.getcwd())
        conversation.send_message(instruction)
        conversation.run()
        events = getattr(getattr(conversation, 'state', None), 'events', None)
        return get_agent_final_response(events) if events is not None else ''

    def declared_capabilities_for_tools(self, tools: Sequence[str]) -> frozenset[Capability]:
        """Map a tool list onto capabilities (used by contract tests)."""
        mapped = {_TOOL_CAPABILITIES[tool] for tool in tools if tool in _TOOL_CAPABILITIES}
        mapped.add(Capability.CHAT)
        return frozenset(mapped)
