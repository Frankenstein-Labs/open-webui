"""CORTEX orchestrator: runs a routing plan and normalizes engine output.

The orchestrator owns the chain. It asks the router for an ordered list of
engines, executes each one, and merges their events into a single versioned
CORTEX stream so the chat layer sees one coherent answer even when the work was
split between engines.

Follow-up engines receive the primary engine's result as context, which is how
"write the code with OpenHands, then verify it in a browser with ai-manus"
becomes a single task rather than two unrelated prompts.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Iterable
from typing import Any

from open_webui.inference.cortex.capabilities import (
    AgentCapability,
    Capability,
    EngineContext,
    EngineRegistry,
)
from open_webui.inference.cortex.policy import CortexPolicy, CortexPolicyError
from open_webui.inference.cortex.protocol import CortexEvent, error_event, make_event
from open_webui.inference.cortex.routing import RoutingPlan, plan_route

log = logging.getLogger(__name__)

_FOLLOW_UP_TEMPLATE = (
    'Previous step ({engine}) produced the following result. Use the capabilities '
    'you are responsible for to complete or verify the task.\n\n'
    'Original request:\n{request}\n\nPrevious result:\n{result}'
)


class EngineUnavailableError(RuntimeError):
    """Raised when no registered engine can serve the requested capabilities."""


class CortexOrchestrator:
    """Executes capability-routed tasks across one or more engines."""

    def __init__(self, registry: EngineRegistry, policy: CortexPolicy | None = None) -> None:
        self.registry = registry
        self.policy = policy or CortexPolicy.from_env()

    async def route(
        self,
        form_data: dict[str, Any],
        *,
        context: EngineContext | None = None,
        allow_chaining: bool = True,
        preferred_engine: str | None = None,
        required: Iterable[Capability] | None = None,
    ) -> AsyncIterator[CortexEvent]:
        """Yield the normalized CORTEX event stream for `form_data`.

        `required` overrides capability inference, which lets a caller (or a
        planner) state the capabilities explicitly instead of relying on
        keyword cues.
        """
        request = _request_text(form_data)
        task_id = f'task_{uuid.uuid4()}'
        context = context or EngineContext(workspace_id=f'ws_{uuid.uuid4()}')

        plan: RoutingPlan = plan_route(
            self.registry,
            form_data,
            required=required,
            preferred_engine=preferred_engine,
            allow_chaining=allow_chaining,
        )
        if plan.requirements is not None:
            yield make_event('TaskCreated', plan.requirements.to_dict(), task_id=task_id)
        if plan.unsupported:
            yield error_event(
                'capability_unsupported',
                'No engine covers: ' + ', '.join(sorted(capability.value for capability in plan.unsupported)),
                recoverable=True,
                task_id=task_id,
            )
        if not plan.selections:
            raise EngineUnavailableError('No engine adapter is registered')

        yield make_event(
            'TaskAssigned',
            {'plan': plan.to_dict(), 'engines': plan.engine_names},
            task_id=task_id,
        )

        previous_result = ''
        previous_engine = ''
        for index, selection in enumerate(plan.selections):
            adapter = self.registry.get(selection.engine)
            instruction = request
            if index > 0 and previous_result:
                instruction = _FOLLOW_UP_TEMPLATE.format(
                    engine=previous_engine,
                    request=request,
                    result=previous_result,
                )

            session = await adapter.create_session(
                {
                    'id': task_id,
                    'title': request[:120] or 'agent task',
                    'objective': request,
                    'context': context,
                    'capabilities': sorted(capability.value for capability in selection.capabilities),
                },
                context,
            )
            yield make_event(
                'TaskStarted',
                {'engine': selection.engine, 'session': session.to_dict(), 'reason': selection.reason},
                task_id=task_id,
                agent_id=session.id,
            )

            last_text = ''
            try:
                async for event in adapter.execute(session.id, instruction, context):
                    if event.type == 'AgentMessage':
                        text = str(event.payload.get('content') or '')
                        if text:
                            last_text = text
                    yield event
            except CortexPolicyError as exc:
                yield error_event(exc.code, str(exc), recoverable=False, task_id=task_id)
                yield make_event('TaskFailed', {'engine': selection.engine, 'code': exc.code}, task_id=task_id)
                return
            except Exception as exc:  # engine failures must not lose the task graph
                log.exception('Engine %s failed during task %s', selection.engine, task_id)
                yield error_event('engine_failure', f'{selection.engine}: {exc}', recoverable=True, task_id=task_id)
                yield make_event('TaskFailed', {'engine': selection.engine, 'message': str(exc)}, task_id=task_id)
                return

            previous_result = last_text or previous_result
            previous_engine = selection.engine
            yield make_event(
                'TaskCompleted',
                {'engine': selection.engine, 'capabilities': sorted(c.value for c in selection.capabilities)},
                task_id=task_id,
                agent_id=session.id,
            )

    async def collect(
        self,
        form_data: dict[str, Any],
        *,
        context: EngineContext | None = None,
        allow_chaining: bool = True,
        preferred_engine: str | None = None,
        required: Iterable[Capability] | None = None,
    ) -> tuple[str, list[CortexEvent], dict[str, Any]]:
        """Run a plan to completion and return (text, events, routing metadata)."""
        events: list[CortexEvent] = []
        texts: list[str] = []
        routing: dict[str, Any] = {}
        async for event in self.route(
            form_data,
            context=context,
            allow_chaining=allow_chaining,
            preferred_engine=preferred_engine,
            required=required,
        ):
            events.append(event)
            if event.type == 'AgentMessage':
                content = str(event.payload.get('content') or '')
                if content:
                    texts.append(content)
            elif event.type == 'TaskAssigned':
                plan = event.payload.get('plan') or {}
                routing = plan
        return '\n\n'.join(texts).strip(), events, routing


def describe_registry(registry: EngineRegistry) -> dict[str, Any]:
    """Capability summary used by the admin/diagnostic endpoint."""
    capabilities: dict[str, AgentCapability] = registry.capabilities()
    return {
        'engines': [capability.to_dict() for capability in capabilities.values()],
        'capabilities': sorted(capability.value for capability in Capability),
    }


def _request_text(form_data: dict[str, Any]) -> str:
    messages = form_data.get('messages') or []
    parts = [
        str(message.get('content', ''))
        for message in messages
        if isinstance(message, dict) and message.get('role') != 'system'
    ]
    return '\n\n'.join(part for part in parts if part)
