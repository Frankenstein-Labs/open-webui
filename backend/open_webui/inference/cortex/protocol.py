"""Versioned event protocol shared by the CORTEX engine bridge.

Python mirror of `packages/protocol` in CORTEX-Software-Agent-SDK: the wire
shape and version tag must stay identical so events emitted here can be
consumed by the TypeScript orchestrator (and vice versa) without translation.

Every event is tagged with `version` so persisted streams stay readable when
the contract evolves.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

PROTOCOL_VERSION = 1

EVENT_TYPES: tuple[str, ...] = (
    'TaskCreated',
    'TaskAssigned',
    'TaskStarted',
    'TaskCompleted',
    'TaskFailed',
    'AgentStarted',
    'AgentStopped',
    'AgentMessage',
    'ToolRequested',
    'ToolCompleted',
    'ArtifactCreated',
    'ReviewRequested',
    'ReviewCompleted',
    'ErrorRaised',
)

_TERMINAL_EVENTS = frozenset({'TaskCompleted', 'TaskFailed'})


@dataclass(slots=True)
class CortexEvent:
    """A single versioned event on the CORTEX event bus."""

    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    id: str = ''
    version: int = PROTOCOL_VERSION
    timestamp: str = ''
    task_id: str | None = None
    agent_id: str | None = None
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f'Unknown CORTEX event type: {self.type}')
        if not self.id:
            self.id = f'{self.type}_{uuid.uuid4()}'
        if not self.timestamp:
            self.timestamp = dt.datetime.now(dt.timezone.utc).isoformat()

    @property
    def is_terminal(self) -> bool:
        return self.type in _TERMINAL_EVENTS

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'version': self.version,
            'type': self.type,
            'timestamp': self.timestamp,
            'taskId': self.task_id,
            'agentId': self.agent_id,
            'correlationId': self.correlation_id,
            'payload': self.payload,
        }

    def to_sse(self) -> str:
        return f'data: {json.dumps(self.to_dict())}\n\n'

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> 'CortexEvent':
        return cls(
            type=str(raw.get('type', '')),
            payload=dict(raw.get('payload') or {}),
            id=str(raw.get('id') or ''),
            version=int(raw.get('version') or PROTOCOL_VERSION),
            timestamp=str(raw.get('timestamp') or ''),
            task_id=raw.get('taskId'),
            agent_id=raw.get('agentId'),
            correlation_id=raw.get('correlationId'),
        )

    def __iter__(self) -> Iterator[str]:
        yield self.to_sse()


def make_event(
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    task_id: str | None = None,
    agent_id: str | None = None,
    correlation_id: str | None = None,
) -> CortexEvent:
    """Build an event with the same ergonomics as the TypeScript `makeEvent`."""
    return CortexEvent(
        type=event_type,
        payload=payload or {},
        task_id=task_id,
        agent_id=agent_id,
        correlation_id=correlation_id,
    )


def error_event(code: str, message: str, *, recoverable: bool = True, task_id: str | None = None) -> CortexEvent:
    return make_event(
        'ErrorRaised',
        {'code': code, 'message': message, 'recoverable': recoverable},
        task_id=task_id,
    )


def tool_event(tool: str, *, phase: str, input_data: Any = None, output: Any = None) -> CortexEvent:
    """Emit `ToolRequested`/`ToolCompleted` with a normalized payload."""
    event_type = 'ToolRequested' if phase == 'requested' else 'ToolCompleted'
    payload: dict[str, Any] = {'tool': tool}
    if input_data is not None:
        payload['input'] = input_data
    if output is not None:
        payload['output'] = output
    return make_event(event_type, payload)
