"""Cortex-owned capability vocabulary and neutral engine contracts.

Python mirror of `packages/core` + `packages/engine-adapters` from
CORTEX-Software-Agent-SDK. The orchestrator reasons about *capabilities*
(`browser`, `screen`, `terminal`, `files`, ...) rather than engine names, so a
new engine can be registered without touching routing logic.

An engine is only ever reached through `EngineAdapter`. Engine-specific types
never cross this boundary: adapters translate both ways.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from open_webui.inference.cortex.protocol import CortexEvent


class Capability(str, Enum):
    """Normalized task capabilities.

    Deliberately closed: routing compares sets, so an unknown capability would
    silently never match. Adapters map their native tools onto these values.
    """

    CHAT = 'chat'
    REASONING = 'reasoning'
    CODE = 'code'
    TERMINAL = 'terminal'
    FILES = 'files'
    BROWSER = 'browser'
    SCREEN = 'screen'
    COMPUTER = 'computer'
    WEB_SEARCH = 'web_search'
    MCP = 'mcp'
    SKILLS = 'skills'
    DELEGATION = 'delegation'
    ARTIFACTS = 'artifacts'


@dataclass(frozen=True, slots=True)
class AgentCapability:
    """What an engine can do. Mirrors the TypeScript `AgentCapability`."""

    engine: str
    tools: frozenset[Capability] = frozenset()
    supports_pause: bool = False
    supports_streaming: bool = False
    supports_cancel: bool = True
    isolated_runtime: bool = False
    notes: str = ''

    def supports(self, capability: Capability) -> bool:
        return capability in self.tools

    def covers(self, required: Iterable[Capability]) -> bool:
        return set(required) <= set(self.tools)

    def to_dict(self) -> dict[str, Any]:
        return {
            'engine': self.engine,
            'tools': sorted(capability.value for capability in self.tools),
            'supportsPause': self.supports_pause,
            'supportsStreaming': self.supports_streaming,
            'supportsCancel': self.supports_cancel,
            'isolatedRuntime': self.isolated_runtime,
            'notes': self.notes,
        }


@dataclass(slots=True)
class EngineContext:
    """Execution context handed to an engine when a session is created."""

    workspace_id: str
    user_id: str | None = None
    chat_id: str | None = None
    policy_id: str | None = None
    budget: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class EngineSession:
    """Lifecycle-tracking handle for one engine session."""

    id: str
    task_id: str
    engine: str
    status: str = 'created'  # created|running|paused|completed|failed|cancelled
    detail: str = ''

    TERMINAL_STATUSES = ('completed', 'failed', 'cancelled')

    def __post_init__(self) -> None:
        if not self.id:
            self.id = f'session_{uuid.uuid4()}'

    @property
    def is_terminal(self) -> bool:
        return self.status in self.TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'taskId': self.task_id,
            'engine': self.engine,
            'status': self.status,
            'detail': self.detail,
        }


@runtime_checkable
class EngineAdapter(Protocol):
    """Neutral contract every engine must implement.

    Mirrors the TypeScript `EngineAdapter` interface
    (`packages/engine-adapters`).
    """

    name: str

    async def initialize(self) -> None: ...

    async def shutdown(self) -> None: ...

    def get_capabilities(self) -> AgentCapability: ...

    async def create_session(self, task: dict[str, Any], context: EngineContext) -> EngineSession: ...

    def execute(
        self, session_id: str, instruction: str, context: EngineContext | None = None
    ) -> AsyncIterator[CortexEvent]: ...

    async def cancel(self, session_id: str, reason: str) -> None: ...

    async def pause(self, session_id: str) -> None: ...

    async def resume(self, session_id: str) -> None: ...

    async def get_status(self, session_id: str) -> EngineSession: ...


class EngineRegistry:
    """Registry of engine adapters, mirroring the TypeScript `EngineRegistry`."""

    def __init__(self) -> None:
        self._adapters: dict[str, EngineAdapter] = {}

    def register(self, adapter: EngineAdapter) -> None:
        if adapter.name in self._adapters:
            raise ValueError(f'Engine adapter already registered: {adapter.name}')
        self._adapters[adapter.name] = adapter

    def unregister(self, name: str) -> None:
        self._adapters.pop(name, None)

    def get(self, name: str) -> EngineAdapter:
        adapter = self._adapters.get(name)
        if adapter is None:
            raise KeyError(f'Engine adapter is not registered: {name}')
        return adapter

    def has(self, name: str) -> bool:
        return name in self._adapters

    def list(self) -> list[str]:
        return sorted(self._adapters)

    def capabilities(self) -> dict[str, AgentCapability]:
        return {name: adapter.get_capabilities() for name, adapter in self._adapters.items()}

    def engines_supporting(self, required: Iterable[Capability]) -> list[str]:
        """Engines that expose every required capability, in registration order."""
        return [name for name, adapter in self._adapters.items() if adapter.get_capabilities().covers(required)]
