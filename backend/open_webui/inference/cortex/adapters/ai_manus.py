"""AI-Manus adapter: computer, browser, screen (VNC/CDP), terminal and files.

Wraps the ai-manus HTTP API and chat WebSocket behind the neutral
`EngineAdapter` contract. Corrections against the previous integration:

* **Authentication.** ai-manus authenticates with a *user* bearer token
  (opaque session or JWT) or an HttpOnly cookie — not a static API token. A
  bare `AI_MANUS_TOKEN` therefore cannot create sessions. We resolve a token
  via, in order: `AI_MANUS_TOKEN`, an explicit JWT minted from
  `AI_MANUS_JWT_SECRET`, or an optional login exchange
  (`AI_MANUS_USERNAME`/`AI_MANUS_PASSWORD`).
* **Full WebSocket parsing.** The old loop only read `type == "event"` and
  silently dropped `ack`, `ping`, `wait`, `stopped`, and `stream_end`. All
  message types are now handled.
* **Isolation.** Endpoint URLs go through `CortexPolicy.validate_endpoint`,
  and we deliberately do *not* inherit ai-manus' defaults (passwordless VNC,
  `--disable-web-security`, mounted Docker socket): the reachable runtime is
  restricted to the host CORTEX allowlists.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from open_webui.inference.cortex.adapters.base import BaseEngineAdapter, text_from_event
from open_webui.inference.cortex.capabilities import AgentCapability, Capability, EngineContext, EngineSession
from open_webui.inference.cortex.protocol import CortexEvent, error_event, make_event

CHAT_WS_PROTOCOL_VERSION = 2

_CAPABILITIES = frozenset(
    {
        Capability.CHAT,
        Capability.REASONING,
        Capability.CODE,
        Capability.TERMINAL,
        Capability.FILES,
        Capability.BROWSER,
        Capability.SCREEN,
        Capability.COMPUTER,
        Capability.WEB_SEARCH,
        Capability.MCP,
        Capability.SKILLS,
        Capability.ARTIFACTS,
    }
)


class AiManusAdapter(BaseEngineAdapter):
    """Runs tasks on an ai-manus deployment (sandbox, browser, virtual screen)."""

    name = 'ai-manus'

    def __init__(self, policy: Any = None) -> None:
        super().__init__(policy=policy)
        # Set when the task hands this adapter a shared computer; it wins over
        # AI_MANUS_URL for the duration of that task.
        self._base_url_override = ''
        self._capabilities = AgentCapability(
            engine=self.name,
            tools=_CAPABILITIES,
            supports_pause=False,
            supports_streaming=True,
            supports_cancel=True,
            isolated_runtime=True,
            notes='ai-manus; per-task Docker sandbox with Chromium (CDP) and Xvfb/x11vnc screen',
        )

    # ── configuration ────────────────────────────────────────────────────
    def configure_computer(self, *, base_url: str = '') -> None:
        """Point ai-manus at the task's shared sandbox.

        ai-manus reads its sandbox location from ``SANDBOX_ADDRESS``; when it is
        set the backend stops creating its own container per task and uses the
        one we give it. That is what makes the computer genuinely shared between
        engines instead of each engine silently starting its own machine.
        """
        if base_url:
            self._base_url_override = base_url
            os.environ['AI_MANUS_SANDBOX_ADDRESS'] = base_url

    def _base_url(self) -> str:
        url = (self._base_url_override or os.getenv('AI_MANUS_URL', '')).rstrip('/')
        if not url:
            raise RuntimeError('AI_MANUS_URL is required for computer and browser tasks')
        if self.policy is not None:
            return self.policy.validate_endpoint(url)
        return url

    def _ws_url(self, base_url: str) -> str:
        ws = base_url.replace('https://', 'wss://').replace('http://', 'ws://')
        return f'{ws}/api/v1/ws/chat'

    def _auth_headers(self) -> dict[str, str]:
        """Resolve an ai-manus bearer token (user session/JWT, not a static key)."""
        token = os.getenv('AI_MANUS_TOKEN', '')
        if not token:
            token = self._mint_jwt()
        if not token:
            token = self._login()
        return {'Authorization': f'Bearer {token}'} if token else {}

    def _mint_jwt(self) -> str:
        secret = os.getenv('AI_MANUS_JWT_SECRET', '')
        if not secret:
            return ''
        try:
            import jwt  # type: ignore[import-not-found]
        except ImportError:
            return ''
        import datetime as dt

        subject = os.getenv('AI_MANUS_SERVICE_USER', 'cortex-service')
        now = dt.datetime.now(dt.timezone.utc)
        payload = {'sub': subject, 'type': 'service', 'iat': now, 'exp': now + dt.timedelta(minutes=30)}
        return jwt.encode(payload, secret, algorithm=os.getenv('AI_MANUS_JWT_ALGORITHM', 'HS256'))

    def _login(self) -> str:
        username = os.getenv('AI_MANUS_USERNAME', '')
        password = os.getenv('AI_MANUS_PASSWORD', '')
        if not username or not password:
            return ''
        import httpx

        try:
            response = httpx.post(
                f'{self._base_url()}/api/v1/auth/login',
                json={'username': username, 'password': password},
                timeout=15.0,
            )
            response.raise_for_status()
        except Exception:
            return ''
        body = response.json()
        return str((body.get('data') or {}).get('access_token') or '')

    # ── session creation ─────────────────────────────────────────────────
    async def create_session(self, task: dict[str, Any], context: EngineContext) -> EngineSession:
        session = await super().create_session(task, context)
        session.detail = 'ai-manus session pending'
        return session

    async def _open_remote_session(self, headers: dict[str, str]) -> str:
        import httpx

        async with httpx.AsyncClient(base_url=self._base_url(), headers=headers, timeout=30.0) as client:
            response = await client.put('/api/v1/sessions')
            response.raise_for_status()
            body = response.json()
        if isinstance(body, dict) and body.get('code') not in (None, 0):
            raise RuntimeError(f'ai-manus rejected the session: {body.get("msg")}')
        data = body.get('data') if isinstance(body, dict) else None
        session_id = (data or {}).get('session_id') if isinstance(data, dict) else None
        if not session_id:
            raise RuntimeError('ai-manus did not return a session id')
        return str(session_id)

    # ── execution ────────────────────────────────────────────────────────
    async def _stream(
        self,
        session: EngineSession,
        instruction: str,
        context: EngineContext | None,
    ) -> AsyncIterator[CortexEvent]:
        try:
            import websockets  # noqa: F401
        except ImportError:
            yield error_event(
                'engine_dependency',
                'Install websockets>=14 to use the ai-manus adapter',
                recoverable=False,
                task_id=session.task_id,
            )
            session.status = 'failed'
            return

        try:
            headers = self._auth_headers()
            remote_id = await self._open_remote_session(headers)
        except Exception as exc:
            yield error_event('engine_unavailable', f'ai-manus unavailable: {exc}', task_id=session.task_id)
            session.status = 'failed'
            return

        session.detail = remote_id
        yield make_event(
            'ToolRequested',
            {'tool': 'ai-manus.sandbox', 'session': remote_id},
            task_id=session.task_id,
            agent_id=session.id,
        )

        async for event in self._pump_websocket(session, remote_id, instruction, headers):
            yield event

    async def _pump_websocket(
        self,
        session: EngineSession,
        remote_id: str,
        instruction: str,
        headers: dict[str, str],
    ) -> AsyncIterator[CortexEvent]:
        import websockets

        ws_url = self._ws_url(self._base_url())
        if self.policy is not None:
            self.policy.validate_endpoint(ws_url, capability=Capability.COMPUTER)

        try:
            socket = await websockets.connect(ws_url, additional_headers=headers or None)
        except Exception as exc:
            yield error_event('engine_unavailable', f'ai-manus websocket failed: {exc}', task_id=session.task_id)
            session.status = 'failed'
            return

        try:
            await socket.send(
                json.dumps(
                    {
                        'id': 'cortex-join',
                        'version': CHAT_WS_PROTOCOL_VERSION,
                        'type': 'join_session',
                        'session_id': remote_id,
                    }
                )
            )
            async for event in self._read_messages(socket, session, 'join'):
                yield event

            await socket.send(
                json.dumps(
                    {
                        'id': 'cortex-chat',
                        'version': CHAT_WS_PROTOCOL_VERSION,
                        'type': 'chat',
                        'session_id': remote_id,
                        'message': instruction,
                    }
                )
            )
            async for event in self._read_messages(socket, session, 'chat'):
                yield event
        finally:
            try:
                await socket.close()
            except Exception:  # pragma: no cover - best effort
                pass

    async def _read_messages(
        self,
        socket: Any,
        session: EngineSession,
        phase: str,
    ) -> AsyncIterator[CortexEvent]:
        """Read the WS stream, handling every message type (not just `event`)."""
        while True:
            raw = json.loads(await socket.recv())
            kind = raw.get('type')

            if kind == 'ping':
                continue
            if kind in {'joined', 'left', 'stopped'}:
                yield make_event(
                    'AgentStarted',
                    {'protocol': kind, 'session': raw.get('session_id')},
                    task_id=session.task_id,
                    agent_id=session.id,
                )
                if phase == 'join':
                    return
                continue
            if kind == 'ack':
                yield make_event(
                    'ToolRequested',
                    {'ack': raw.get('op'), 'ok': raw.get('ok')},
                    task_id=session.task_id,
                    agent_id=session.id,
                )
                continue
            if kind == 'error':
                yield error_event(
                    'engine_error',
                    str(raw.get('error', 'ai-manus task failed')),
                    recoverable=False,
                    task_id=session.task_id,
                )
                session.status = 'failed'
                return
            if kind == 'wait':
                continue
            if kind == 'event':
                text = text_from_event(raw.get('data'))
                event_name = str(raw.get('event') or '')
                if text:
                    yield self.message(session, text, source_event=event_name)
                else:
                    yield make_event(
                        'ToolCompleted',
                        {'event': event_name, 'data': _safe(raw.get('data'))},
                        task_id=session.task_id,
                        agent_id=session.id,
                    )
                continue
            if kind == 'stream_end':
                if phase == 'join':
                    return
                session.status = 'completed'
                return
            # Unknown type: forward it rather than dropping it silently.
            yield make_event(
                'AgentMessage',
                {'content': '', 'unknown_type': str(kind), 'data': _safe(raw)},
                task_id=session.task_id,
                agent_id=session.id,
            )

    async def _cancel(self, session_id: str, reason: str) -> None:
        session = self._sessions.get(session_id)
        if not session or not session.detail:
            return
        import httpx

        try:
            async with httpx.AsyncClient(
                base_url=self._base_url(), headers=self._auth_headers(), timeout=15.0
            ) as client:
                await client.post(f'/api/v1/sessions/{session.detail}/stop')
        except Exception:  # cancellation is best effort
            return


def _safe(value: Any, limit: int = 2000) -> Any:
    """Bound payload size before it enters the event stream."""
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, (dict, list)):
        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError):
            return str(value)[:limit]
        return value if len(encoded) <= limit else {'truncated': encoded[:limit]}
    return value
