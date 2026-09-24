"""Tests for the CORTEX orchestrator and the ai-manus WebSocket parser.

Uses explicit test doubles that implement the real `EngineAdapter` protocol so
the orchestration logic (chaining, event normalization, failure isolation) is
exercised end to end without any engine running.
"""

import asyncio
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from open_webui.inference.cortex.adapters.ai_manus import AiManusAdapter  # noqa: E402
from open_webui.inference.cortex.adapters.base import BaseEngineAdapter  # noqa: E402
from open_webui.inference.cortex.capabilities import (  # noqa: E402
    AgentCapability,
    Capability,
    EngineContext,
    EngineRegistry,
)
from open_webui.inference.cortex.orchestrator import CortexOrchestrator  # noqa: E402


class ScriptedAdapter(BaseEngineAdapter):
    """Test double returning a fixed script of events per instruction."""

    def __init__(self, name: str, capabilities: frozenset[Capability], reply: str = 'ok', fail: bool = False) -> None:
        super().__init__()
        self.name = name
        self._capabilities = AgentCapability(engine=name, tools=capabilities, supports_streaming=True)
        self.reply = reply
        self.fail = fail
        self.instructions: list[str] = []

    async def _stream(self, session, instruction, context):  # type: ignore[no-untyped-def]
        self.instructions.append(instruction)
        if self.fail:
            raise RuntimeError('engine exploded')
        yield self.message(session, f'{self.reply}')


def _context() -> EngineContext:
    return EngineContext(workspace_id='ws_test')


def _run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_single_engine_task_emits_normalized_event_stream():
    registry = EngineRegistry()
    registry.register(ScriptedAdapter('solo', frozenset({Capability.CHAT, Capability.TERMINAL}), reply='done'))
    orchestrator = CortexOrchestrator(registry)

    text, events, routing = _run(
        orchestrator.collect(
            {'messages': [{'role': 'user', 'content': 'run the build'}]},
            context=_context(),
        )
    )
    assert text == 'done'
    types = [event.type for event in events]
    assert types[0] == 'TaskCreated'
    assert 'TaskAssigned' in types
    assert 'TaskStarted' in types
    assert 'TaskCompleted' in types
    assert routing['primary'] == 'solo'


def test_chain_passes_previous_result_into_the_next_engine():
    registry = EngineRegistry()
    primary = ScriptedAdapter(
        'openhands',
        frozenset({Capability.CHAT, Capability.TERMINAL, Capability.FILES, Capability.CODE}),
        reply='code written',
    )
    follow_up = ScriptedAdapter('ai-manus', frozenset({Capability.CHAT, Capability.SCREEN, Capability.BROWSER}))
    registry.register(primary)
    registry.register(follow_up)

    orchestrator = CortexOrchestrator(registry)
    _, events, routing = _run(
        orchestrator.collect(
            {
                'conversation_mode': 'agent',
                'messages': [{'role': 'user', 'content': 'do the work'}],
            },
            context=_context(),
            required={Capability.TERMINAL, Capability.CODE, Capability.SCREEN, Capability.BROWSER},
        )
    )
    assert routing['primary'] == 'openhands'
    assert 'ai-manus' in [entry['engine'] for entry in routing['chain']]
    # The follow-up engine must receive the primary engine's output as context.
    assert len(follow_up.instructions) == 1
    assert 'code written' in follow_up.instructions[0]
    assert 'do the work' in follow_up.instructions[0]
    assert 'TaskCompleted' in [event.type for event in events]


def test_engine_failure_isolated_as_error_event_and_task_failed():
    registry = EngineRegistry()
    registry.register(ScriptedAdapter('broken', frozenset({Capability.CHAT, Capability.TERMINAL}), fail=True))
    orchestrator = CortexOrchestrator(registry)
    _, events, _ = _run(orchestrator.collect({'messages': [{'role': 'user', 'content': 'run'}]}, context=_context()))
    types = [event.type for event in events]
    assert 'ErrorRaised' in types
    assert 'TaskFailed' in types
    error = next(event for event in events if event.type == 'ErrorRaised')
    assert error.payload['recoverable'] is True


def test_no_registered_engine_raises():
    from open_webui.inference.cortex.orchestrator import EngineUnavailableError

    orchestrator = CortexOrchestrator(EngineRegistry())
    try:
        _run(orchestrator.collect({'messages': []}, context=_context()))
    except EngineUnavailableError:
        pass
    else:  # pragma: no cover
        raise AssertionError('an empty registry must raise EngineUnavailableError')


def test_unsupported_capabilities_are_reported_on_the_stream():
    registry = EngineRegistry()
    registry.register(ScriptedAdapter('chat-only', frozenset({Capability.CHAT})))
    orchestrator = CortexOrchestrator(registry)
    _, events, _ = _run(
        orchestrator.collect(
            {'messages': [{'role': 'user', 'content': 'take a screenshot in the browser'}]},
            context=_context(),
        )
    )
    errors = [event for event in events if event.type == 'ErrorRaised']
    assert any('capability_unsupported' == event.payload['code'] for event in errors)


def test_events_carry_protocol_version_and_ids():
    registry = EngineRegistry()
    registry.register(ScriptedAdapter('solo', frozenset({Capability.CHAT, Capability.TERMINAL})))
    orchestrator = CortexOrchestrator(registry)
    _, events, _ = _run(orchestrator.collect({'messages': [{'role': 'user', 'content': 'run'}]}, context=_context()))
    for event in events:
        payload = event.to_dict()
        assert payload['version'] == 1
        assert payload['id']
        assert payload['timestamp']
    assigned = next(event for event in events if event.type == 'TaskAssigned')
    assert assigned.task_id is not None


# ── ai-manus WebSocket parsing ──────────────────────────────────────────────


class FakeSocket:
    """Minimal websocket double: replays queued frames and records sends."""

    def __init__(self, frames: list[dict]) -> None:
        self.frames = list(frames)
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, raw: str) -> None:
        import json

        self.sent.append(json.loads(raw))

    async def recv(self) -> str:
        import json

        if not self.frames:
            raise AssertionError('fake socket ran out of frames')
        return json.dumps(self.frames.pop(0))

    async def close(self) -> None:
        self.closed = True


def _drain(adapter: AiManusAdapter, socket: FakeSocket, phase: str):  # type: ignore[no-untyped-def]
    async def run():  # type: ignore[no-untyped-def]
        session = await adapter.create_session({'id': 't1'}, _context())
        return [event async for event in adapter._read_messages(socket, session, phase)]

    return _run(run())


def test_ws_parser_handles_joined_then_returns():
    adapter = AiManusAdapter()
    socket = FakeSocket([{'type': 'joined', 'session_id': 'remote-1'}])
    events = _drain(adapter, socket, 'join')
    assert [event.type for event in events] == ['AgentStarted']


def test_ws_parser_ignores_ping_and_wait_without_dropping_the_stream():
    adapter = AiManusAdapter()
    socket = FakeSocket(
        [
            {'type': 'ping'},
            {'type': 'wait'},
            {'type': 'event', 'event': 'message', 'data': {'content': 'working'}},
            {'type': 'stream_end'},
        ]
    )
    events = _drain(adapter, socket, 'chat')
    assert [event.type for event in events] == ['AgentMessage']
    assert events[0].payload['content'] == 'working'


def test_ws_parser_reads_ack_as_tool_event():
    adapter = AiManusAdapter()
    socket = FakeSocket([{'type': 'ack', 'op': 'chat', 'ok': True}, {'type': 'stream_end'}])
    events = _drain(adapter, socket, 'chat')
    assert [event.type for event in events] == ['ToolRequested']
    assert events[0].payload['ack'] == 'chat'


def test_ws_parser_surfaces_error_and_stops():
    adapter = AiManusAdapter()
    socket = FakeSocket([{'type': 'error', 'error': 'sandbox crashed'}])
    events = _drain(adapter, socket, 'chat')
    assert [event.type for event in events] == ['ErrorRaised']
    assert events[0].payload['message'] == 'sandbox crashed'


def test_ws_parser_forwards_unknown_types_instead_of_dropping_them():
    adapter = AiManusAdapter()
    socket = FakeSocket([{'type': 'brand_new', 'payload': 1}, {'type': 'stream_end'}])
    events = _drain(adapter, socket, 'chat')
    assert len(events) == 1
    assert events[0].payload['unknown_type'] == 'brand_new'


def test_ws_parser_reads_stopped_during_chat():
    adapter = AiManusAdapter()
    socket = FakeSocket([{'type': 'stopped', 'session_id': 's'}, {'type': 'stream_end'}])
    events = _drain(adapter, socket, 'chat')
    assert [event.type for event in events] == ['AgentStarted']


def test_ws_parser_extracts_nested_text():
    adapter = AiManusAdapter()
    socket = FakeSocket(
        [
            {'type': 'event', 'event': 'message', 'data': {'data': {'content': 'nested text'}}},
            {'type': 'stream_end'},
        ]
    )
    events = _drain(adapter, socket, 'chat')
    assert events[0].payload['content'] == 'nested text'
