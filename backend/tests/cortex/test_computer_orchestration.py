"""End-to-end test of the shared computer plane through the orchestrator.

The point of this suite is the claim that makes the feature worth having: when a
task is chained across two engines, both of them are handed the *same* computer,
and the machine is released exactly once when the task ends -- including when an
engine fails.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import _backend  # noqa: F401

from open_webui.inference.cortex.capabilities import (
    AgentCapability,
    Capability,
    EngineContext,
    EngineSession,
    EngineRegistry,
)
from open_webui.inference.cortex.computer import ComputerKind, ComputerState
from open_webui.inference.cortex.computer_local import LocalComputerProvider
from open_webui.inference.cortex.computer_policy import ComputerPolicy
from open_webui.inference.cortex.computers import ComputerRegistry
from open_webui.inference.cortex.orchestrator import CortexOrchestrator
from open_webui.inference.cortex.policy import CortexPolicy
from open_webui.inference.cortex.protocol import make_event


def run(coro):
    return asyncio.run(coro)


class ScriptedEngine:
    """A real EngineAdapter implementation that records the computer it got."""

    def __init__(self, name, capabilities, *, text='done', fail=False):
        self.name = name
        self._capabilities = AgentCapability(engine=name, tools=frozenset(capabilities))
        self.text = text
        self.fail = fail
        self.received_url: list[str] = []
        self.computer = None
        self._sessions: dict[str, EngineSession] = {}

    def configure_computer(self, *, base_url=''):
        self.received_url.append(base_url)
        self.computer = base_url or 'in-host'

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    def get_capabilities(self):
        return self._capabilities

    async def create_session(self, task, context):
        session = EngineSession(id=f'{self.name}_s', task_id=task['id'], engine=self.name)
        self._sessions[session.id] = session
        return session

    async def execute(self, session_id, instruction, context=None):
        if self.fail:
            raise RuntimeError('engine exploded')
        yield self.message(session_id, self.text)

    def message(self, session_id, content):
        session = self._sessions[session_id]
        return make_event('AgentMessage', {'content': content}, task_id=session.task_id, agent_id=session.id)

    async def cancel(self, session_id, reason):
        pass

    async def pause(self, session_id):
        pass

    async def resume(self, session_id):
        pass

    async def get_status(self, session_id):
        return self._sessions[session_id]


def build(tmp_path, engines, *, policy=None):
    registry = EngineRegistry()
    for engine in engines:
        registry.register(engine)
    computers = ComputerRegistry(
        ComputerPolicy(allow_local=True),
        providers={ComputerKind.LOCAL: LocalComputerProvider()},
    )
    orchestrator = CortexOrchestrator(registry, policy or CortexPolicy(), computers=computers)
    return orchestrator, computers


def question(text='write a script and run it in the browser'):
    return {'messages': [{'role': 'user', 'content': text}]}


def test_task_provisions_one_computer_and_releases_it(tmp_path, monkeypatch):
    monkeypatch.setenv('CORTEX_COMPUTER_WORKSPACE_ROOT', str(tmp_path / 'ws'))
    monkeypatch.setenv('CORTEX_COMPUTER_KIND', 'local')

    engine = ScriptedEngine('openhands', {Capability.CODE, Capability.TERMINAL})
    orchestrator, computers = build(tmp_path, [engine])

    events = run(_collect(orchestrator, question()))

    assert computers.size == 0  # released at the end
    kinds = [event.type for event in events]
    assert 'TaskAssigned' in kinds
    assert 'ArtifactCreated' in kinds
    assert 'AgentStopped' in kinds
    assert kinds[-1] == 'AgentStopped'

    # The computer was described, not leaked as a credential.
    artifact = next(event for event in events if event.type == 'ArtifactCreated')
    assert artifact.payload['computerKind'] == 'local'
    assert artifact.payload['computerId']


def test_both_engines_receive_the_same_computer(tmp_path, monkeypatch):
    monkeypatch.setenv('CORTEX_COMPUTER_WORKSPACE_ROOT', str(tmp_path / 'ws'))
    monkeypatch.setenv('CORTEX_COMPUTER_KIND', 'local')

    # Primary covers the most capabilities, so it is selected alone; the second
    # engine uniquely owns the browser/screen, so those are chained to it. This
    # is the real shape of "code with OpenHands, verify in a browser with
    # ai-manus".
    primary = ScriptedEngine(
        'openhands',
        {Capability.CHAT, Capability.REASONING, Capability.CODE, Capability.TERMINAL, Capability.FILES},
        text='code written',
    )
    secondary = ScriptedEngine(
        'ai-manus',
        {Capability.CHAT, Capability.BROWSER, Capability.SCREEN, Capability.COMPUTER},
        text='verified in browser',
    )
    orchestrator, _computers = build(tmp_path, [primary, secondary])
    events = run(_collect(orchestrator, question('build it and verify it in the browser')))

    started = [event.payload['engine'] for event in events if event.type == 'TaskStarted']
    assert started == ['openhands', 'ai-manus']

    artifact = next(event for event in events if event.type == 'ArtifactCreated')
    computer_id = artifact.payload['computerId']

    # Both engines were configured for the very same computer; the local one has
    # no URL, so they each get the in-host marker but the same id.
    assert primary.received_url and secondary.received_url
    stopped = next(event for event in events if event.type == 'AgentStopped')
    assert stopped.payload['computerId'] == computer_id


def test_computer_is_released_even_when_an_engine_fails(tmp_path, monkeypatch):
    monkeypatch.setenv('CORTEX_COMPUTER_WORKSPACE_ROOT', str(tmp_path / 'ws'))
    monkeypatch.setenv('CORTEX_COMPUTER_KIND', 'local')

    failing = ScriptedEngine('openhands', {Capability.CODE, Capability.TERMINAL}, fail=True)
    orchestrator, computers = build(tmp_path, [failing])

    events = run(_collect(orchestrator, question()))

    assert computers.size == 0, 'a failed task must not strand a container'
    assert 'TaskFailed' in [event.type for event in events]


def test_pure_chat_task_provisions_no_computer(tmp_path, monkeypatch):
    monkeypatch.setenv('CORTEX_COMPUTER_WORKSPACE_ROOT', str(tmp_path / 'ws'))
    monkeypatch.setenv('CORTEX_COMPUTER_KIND', 'local')

    chat = ScriptedEngine('openhands', {Capability.CHAT, Capability.REASONING})
    orchestrator, computers = build(tmp_path, [chat])

    events = run(_collect(orchestrator, question('what is a monad?')))

    assert 'ArtifactCreated' not in [event.type for event in events]
    assert computers.size == 0


def test_generator_close_releases_the_computer(tmp_path, monkeypatch):
    """An abandoned stream must still tear the machine down."""
    monkeypatch.setenv('CORTEX_COMPUTER_WORKSPACE_ROOT', str(tmp_path / 'ws'))
    monkeypatch.setenv('CORTEX_COMPUTER_KIND', 'local')

    engine = ScriptedEngine('openhands', {Capability.CODE, Capability.TERMINAL})
    orchestrator, computers = build(tmp_path, [engine])

    async def abandon():
        stream = orchestrator.route(question())
        async for event in stream:
            if event.type == 'ArtifactCreated':
                # Stop consuming and drop the generator mid-flight.
                await stream.aclose()
                break

    run(abandon())
    assert computers.size == 0


def test_no_computer_registry_keeps_legacy_behaviour(tmp_path):
    """Without the plane attached, routing still works and provisions nothing."""
    registry = EngineRegistry()
    registry.register(ScriptedEngine('openhands', {Capability.CODE, Capability.TERMINAL}))
    orchestrator = CortexOrchestrator(registry, CortexPolicy())

    events = run(_collect(orchestrator, question()))
    assert 'TaskCompleted' in [event.type for event in events]
    assert 'ArtifactCreated' not in [event.type for event in events]


async def _collect(orchestrator, form_data):
    events = []
    async for event in orchestrator.route(form_data):
        events.append(event)
    return events
