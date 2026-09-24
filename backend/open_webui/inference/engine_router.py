"""Task-aware routing between the ai-manus and OpenHands engines.

The engines remain external/optional: Open WebUI can run without either one, while
configuration makes the integration explicit and keeps the deployment lightweight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


class EngineConfigurationError(RuntimeError):
    """Raised when the selected engine is not configured or installed."""


@dataclass(frozen=True)
class EngineDecision:
    name: str
    reason: str


_COMPUTER_TERMS = re.compile(
    r'\b(browser|browse|website|webpage|computer|desktop|vnc|click|navigate|visual|screen|screenshot|terminal|shell)\b',
    re.IGNORECASE,
)
_CODE_TERMS = re.compile(
    r'\b(code|coding|debug|debugging|refactor|repository|repo|git|commit|pull request|implement|program)\b',
    re.IGNORECASE,
)


def choose_engine(form_data: dict[str, Any]) -> EngineDecision:
    """Choose an engine from an explicit override, model prefix, or task intent."""
    metadata = form_data.get('metadata') or {}
    explicit = (metadata.get('engine') or form_data.get('engine') or os.getenv('OPEN_WEBUI_ENGINE') or 'auto').lower()
    if explicit in {'ai-manus', 'aimanus', 'open-divine', 'openhands-computer'}:
        return EngineDecision('ai-manus', 'explicit configuration')
    if explicit in {'openhands', 'openhands-sdk', 'software-agent-sdk'}:
        return EngineDecision('openhands', 'explicit configuration')

    model = str(form_data.get('model') or '')
    if model.startswith(('ai-manus/', 'open-divine/')):
        return EngineDecision('ai-manus', 'model namespace')
    if model.startswith(('openhands/', 'openhands-sdk/')):
        return EngineDecision('openhands', 'model namespace')

    text = ' '.join(str(item.get('content', '')) for item in form_data.get('messages', []) if isinstance(item, dict))
    if _COMPUTER_TERMS.search(text):
        return EngineDecision('ai-manus', 'computer or browser task')
    if _CODE_TERMS.search(text):
        return EngineDecision('openhands', 'software-engineering task')

    default = os.getenv('OPEN_WEBUI_DEFAULT_ENGINE', 'openhands').lower()
    return EngineDecision(
        'ai-manus' if default in {'ai-manus', 'aimanus', 'open-divine'} else 'openhands',
        'configured default',
    )


def _text_from_event(value: Any) -> str:
    """Extract useful assistant text from heterogeneous agent event payloads."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('content', 'message', 'text', 'output', 'delta'):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate:
                return candidate
            if isinstance(candidate, (dict, list)):
                result = _text_from_event(candidate)
                if result:
                    return result
        for candidate in value.values():
            result = _text_from_event(candidate)
            if result:
                return result
    if isinstance(value, list):
        return ''.join(_text_from_event(item) for item in value)
    return ''


def _openai_response(model: str, content: str, engine: str) -> dict[str, Any]:
    return {
        'id': f'{engine}-completion',
        'object': 'chat.completion',
        'created': 0,
        'model': model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
        'engine': engine,
    }


async def _run_ai_manus(form_data: dict[str, Any]) -> str:
    """Run a task through the ai-manus API and its computer-enabled WebSocket."""
    try:
        import httpx
    except ImportError as exc:
        raise EngineConfigurationError('Install httpx to use the ai-manus adapter') from exc

    base_url = os.getenv('AI_MANUS_URL', '').rstrip('/')
    token = os.getenv('AI_MANUS_TOKEN', '')
    if not base_url:
        raise EngineConfigurationError('AI_MANUS_URL is required for computer/browser tasks')

    headers = {'Authorization': f'Bearer {token}'} if token else {}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30.0) as client:
        response = await client.put('/api/v1/sessions')
        response.raise_for_status()
        payload = response.json()
        session = payload.get('data', payload)
        session_id = session.get('session_id') or session.get('id')
        if not session_id:
            raise RuntimeError('ai-manus did not return a session id')

    ws_url = base_url.replace('https://', 'wss://').replace('http://', 'ws://') + '/api/v1/ws/chat'
    try:
        import websockets
    except ImportError as exc:
        raise EngineConfigurationError('Install websockets to use the ai-manus adapter') from exc

    prompt = '\n\n'.join(
        str(message.get('content', ''))
        for message in form_data.get('messages', [])
        if isinstance(message, dict) and message.get('role') != 'system'
    )
    texts: list[str] = []
    async with websockets.connect(ws_url, additional_headers=headers or None) as socket:
        await socket.send(
            json.dumps({'id': 'open-webui-join', 'version': 2, 'type': 'join_session', 'session_id': session_id})
        )
        await socket.recv()
        await socket.send(
            json.dumps(
                {
                    'id': 'open-webui-chat',
                    'version': 2,
                    'type': 'chat',
                    'session_id': session_id,
                    'message': prompt,
                }
            )
        )
        while True:
            raw = json.loads(await socket.recv())
            if raw.get('type') == 'stream_end':
                break
            if raw.get('type') == 'error':
                raise RuntimeError(raw.get('error', 'ai-manus task failed'))
            if raw.get('type') == 'event':
                text = _text_from_event(raw.get('data'))
                if text:
                    texts.append(text)
    return ''.join(texts) or 'ai-manus completed the task without a textual summary.'


async def _run_openhands(form_data: dict[str, Any]) -> str:
    """Run a task with the optional OpenHands Software Agent SDK."""
    try:
        from openhands.sdk import Conversation, LLM
        from openhands.tools.preset.default import get_default_agent
    except ImportError as exc:
        raise EngineConfigurationError(
            'OpenHands SDK is not installed; install backend/requirements-engines.txt'
        ) from exc

    prompt = '\n\n'.join(
        str(message.get('content', ''))
        for message in form_data.get('messages', [])
        if isinstance(message, dict) and message.get('role') != 'system'
    )
    llm_config: dict[str, Any] = {
        'model': os.getenv('OPENHANDS_MODEL', os.getenv('OPENAI_MODEL', 'gpt-4o')),
        'api_key': os.getenv('OPENHANDS_API_KEY', os.getenv('OPENAI_API_KEY', '')),
        'usage_id': 'open-webui',
        'drop_params': True,
    }
    base_url = os.getenv('OPENHANDS_BASE_URL', os.getenv('OPENAI_API_BASE', ''))
    if base_url:
        llm_config['base_url'] = base_url
    if not llm_config['api_key']:
        raise EngineConfigurationError('OPENHANDS_API_KEY or OPENAI_API_KEY is required')

    def execute() -> str:
        llm = LLM(**llm_config)
        agent = get_default_agent(llm=llm, cli_mode=True)
        conversation = Conversation(agent=agent, workspace=os.getenv('OPENHANDS_WORKSPACE', os.getcwd()))
        conversation.send_message(prompt)
        conversation.run()
        state = getattr(conversation, 'state', None)
        return _text_from_event(getattr(state, 'events', None)) or 'OpenHands completed the task.'

    return await asyncio.to_thread(execute)


async def generate_chat_completion(
    form_data: dict[str, Any],
    *,
    stream: bool = False,
) -> dict[str, Any] | AsyncIterator[str]:
    """Dispatch a chat task to the selected engine using an OpenAI-shaped result."""
    decision = choose_engine(form_data)
    if decision.name == 'ai-manus':
        content = await _run_ai_manus(form_data)
    else:
        content = await _run_openhands(form_data)

    result = _openai_response(str(form_data.get('model', decision.name)), content, decision.name)
    result['routing'] = {'engine': decision.name, 'reason': decision.reason}
    if not stream:
        return result

    async def chunks() -> AsyncIterator[str]:
        yield f'data: {json.dumps(result)}\n\n'
        yield 'data: [DONE]\n\n'

    return chunks()
