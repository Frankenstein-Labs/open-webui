"""Shared adapter machinery: session bookkeeping and engine-event normalization.

Adapters translate engine-native streams into `CortexEvent`s. The helpers here
cover the parts every adapter needs so each one stays focused on its engine's
protocol, and so unknown engine payloads degrade to a safe, lossless default
instead of crashing the task.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from open_webui.inference.cortex.capabilities import EngineContext, EngineSession
from open_webui.inference.cortex.protocol import CortexEvent, make_event


def text_from_event(value: Any) -> str:
    """Extract assistant text from heterogeneous engine payloads.

    Walks dicts/lists depth-first so a nested `data.content`, `message.text` or
    a list of deltas all resolve without per-engine special cases.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('content', 'message', 'text', 'output', 'delta', 'result'):
            if key not in value:
                continue
            candidate = value[key]
            if isinstance(candidate, str) and candidate:
                return candidate
            if isinstance(candidate, (dict, list)):
                nested = text_from_event(candidate)
                if nested:
                    return nested
        for candidate in value.values():
            nested = text_from_event(candidate)
            if nested:
                return nested
    if isinstance(value, list):
        return ''.join(text_from_event(item) for item in value)
    return ''


def openai_response(model: str, content: str, engine: str, routing: dict[str, Any] | None = None) -> dict[str, Any]:
    """Shape an engine result as an OpenAI-compatible chat completion."""
    result: dict[str, Any] = {
        'id': f'{engine}-completion',
        'object': 'chat.completion',
        'created': 0,
        'model': model,
        'choices': [
            {
                'index': 0,
                'message': {'role': 'assistant', 'content': content},
                'finish_reason': 'stop',
            }
        ],
        'engine': engine,
    }
    if routing is not None:
        result['routing'] = routing
    return result


class BaseEngineAdapter:
    """Common lifecycle and session storage for engine adapters.

    Subclasses implement `_stream` and declare `name`/`capabilities`.
    """

    name: str = ''
    _capabilities: Any = None

    def __init__(self, policy: Any = None) -> None:
        self._sessions: dict[str, EngineSession] = {}
        self._initialized = False
        self.policy = policy

    # ── lifecycle ────────────────────────────────────────────────────────
    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._sessions.clear()
        self._initialized = False

    def get_capabilities(self) -> Any:
        return self._capabilities

    async def create_session(self, task: dict[str, Any], context: EngineContext) -> EngineSession:
        session = EngineSession(
            id=f'{self.name}_session_{len(self._sessions) + 1}',
            task_id=str(task.get('id') or ''),
            engine=self.name,
        )
        self._sessions[session.id] = session
        return session

    async def execute(
        self,
        session_id: str,
        instruction: str,
        context: EngineContext | None = None,
    ) -> AsyncIterator[CortexEvent]:
        session = self._require(session_id)
        session.status = 'running'
        try:
            async for event in self._stream(session, instruction, context):
                yield event
        finally:
            if not session.is_terminal:
                session.status = 'completed'

    async def cancel(self, session_id: str, reason: str) -> None:
        session = self._require(session_id)
        session.status = 'cancelled'
        session.detail = reason
        await self._cancel(session_id, reason)

    async def pause(self, session_id: str) -> None:
        session = self._require(session_id)
        session.status = 'paused'

    async def resume(self, session_id: str) -> None:
        session = self._require(session_id)
        session.status = 'running'

    async def get_status(self, session_id: str) -> EngineSession:
        return self._require(session_id)

    # ── hooks for subclasses ─────────────────────────────────────────────
    async def _stream(
        self,
        session: EngineSession,
        instruction: str,
        context: EngineContext | None,
    ) -> AsyncIterator[CortexEvent]:
        raise NotImplementedError

    async def _cancel(self, session_id: str, reason: str) -> None:
        """Best-effort engine-side cancellation; adapters override when supported."""

    # ── helpers ──────────────────────────────────────────────────────────
    def _require(self, session_id: str) -> EngineSession:
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f'Unknown {self.name} session: {session_id}')
        return session

    def message(self, session: EngineSession, content: str, **extra: Any) -> CortexEvent:
        payload = {'content': content, **extra}
        return make_event('AgentMessage', payload, task_id=session.task_id, agent_id=session.id)
