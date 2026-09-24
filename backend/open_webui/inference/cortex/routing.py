"""Capability-based routing across CORTEX engines.

The router never picks an engine by name. It derives the *required
capabilities* for a task, then asks the `EngineRegistry` which adapters cover
them. Keywords are only used to guess which capabilities a task implies — the
decision itself is always a capability-set comparison, so registering a new
engine immediately makes it selectable.

When no single engine covers the whole requirement, the router builds a
*chain*: a primary engine plus follow-up engines for the leftover
capabilities (for example code with OpenHands, then browser verification with
ai-manus).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from open_webui.inference.cortex.capabilities import (
    Capability,
    EngineRegistry,
)

# Capabilities ranked by how much they constrain engine choice. Higher rank is
# assigned first when a chain is needed, so the most specialized engine keeps
# the capability only it can serve.
_CAPABILITY_PRIORITY: tuple[Capability, ...] = (
    Capability.COMPUTER,
    Capability.SCREEN,
    Capability.BROWSER,
    Capability.TERMINAL,
    Capability.MCP,
    Capability.SKILLS,
    Capability.FILES,
    Capability.CODE,
    Capability.WEB_SEARCH,
    Capability.ARTIFACTS,
    Capability.DELEGATION,
    Capability.REASONING,
    Capability.CHAT,
)

_COMPUTER_PATTERN = re.compile(
    r'\b(browser|browse|website|web page|webpage|computer|desktop|vnc|click|navigate|'
    r'visual|screen|screenshot|scrape|crawl|login to|fill (?:in )?(?:the )?form|takeover)\b',
    re.IGNORECASE,
)
_CODE_PATTERN = re.compile(
    r'\b(code|coding|debug|debugging|refactor|repository|repo|git|commit|pull request|'
    r'implement|program|compile|unit test|test suite|bug|patch|migrate)\b',
    re.IGNORECASE,
)
_SEARCH_PATTERN = re.compile(
    r'\b(search|look up|research|find (?:out|the latest)|news|price|documentation|compare)\b',
    re.IGNORECASE,
)
_TERMINAL_PATTERN = re.compile(
    r'\b(run|execute|install|build|deploy|script|shell|terminal|command|docker|npm|pip)\b',
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TaskRequirements:
    """Capabilities a task needs, plus a short label used for logs and events."""

    capabilities: frozenset[Capability]
    intent: str
    mode: str = ''

    def to_dict(self) -> dict[str, Any]:
        return {
            'intent': self.intent,
            'mode': self.mode,
            'capabilities': sorted(capability.value for capability in self.capabilities),
        }


@dataclass(frozen=True, slots=True)
class EngineSelection:
    """One engine in a routing plan, with the capabilities it is responsible for."""

    engine: str
    capabilities: frozenset[Capability]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'engine': self.engine,
            'capabilities': sorted(capability.value for capability in self.capabilities),
            'reason': self.reason,
        }


@dataclass(slots=True)
class RoutingPlan:
    """Ordered engine chain plus any capability no engine could serve."""

    selections: list[EngineSelection] = field(default_factory=list)
    unsupported: frozenset[Capability] = frozenset()
    requirements: TaskRequirements | None = None

    @property
    def primary(self) -> EngineSelection | None:
        return self.selections[0] if self.selections else None

    @property
    def is_chained(self) -> bool:
        return len(self.selections) > 1

    @property
    def engine_names(self) -> list[str]:
        return [selection.engine for selection in self.selections]

    def to_dict(self) -> dict[str, Any]:
        return {
            'primary': self.primary.engine if self.primary else None,
            'chain': [selection.to_dict() for selection in self.selections],
            'unsupported': sorted(capability.value for capability in self.unsupported),
            'requirements': self.requirements.to_dict() if self.requirements else None,
        }


def _text_of(form_data: dict[str, Any]) -> str:
    messages = form_data.get('messages') or []
    return ' '.join(str(item.get('content', '')) for item in messages if isinstance(item, dict))


def derive_requirements(form_data: dict[str, Any]) -> TaskRequirements:
    """Infer the capabilities a task requires from its content and mode."""
    text = _text_of(form_data)
    metadata = form_data.get('metadata') or {}
    mode = str(metadata.get('conversation_mode') or form_data.get('conversation_mode') or '').lower()

    required: set[Capability] = {Capability.CHAT, Capability.REASONING}
    intent_parts: list[str] = []

    if mode == 'agent':
        # Agent mode is task-oriented: assume tool access even when the prompt
        # is terse, so a bare "fix it" still reaches an executing engine.
        required.update({Capability.TERMINAL, Capability.FILES})
        intent_parts.append('agent')

    if _CODE_PATTERN.search(text):
        required.update({Capability.CODE, Capability.TERMINAL, Capability.FILES})
        intent_parts.append('code')
    if _COMPUTER_PATTERN.search(text):
        required.update({Capability.BROWSER, Capability.SCREEN, Capability.COMPUTER})
        intent_parts.append('computer')
    if _SEARCH_PATTERN.search(text):
        required.update({Capability.WEB_SEARCH, Capability.BROWSER})
        intent_parts.append('research')
    if _TERMINAL_PATTERN.search(text):
        required.update({Capability.TERMINAL, Capability.FILES})
        intent_parts.append('terminal')

    intent = '+'.join(dict.fromkeys(intent_parts)) or 'conversation'
    return TaskRequirements(capabilities=frozenset(required), intent=intent, mode=mode)


def _engine_cover(name: str, registry: EngineRegistry, required: frozenset[Capability]) -> frozenset[Capability]:
    return frozenset(
        capability for capability in required if registry.get(name).get_capabilities().supports(capability)
    )


def plan_route(
    registry: EngineRegistry,
    form_data: dict[str, Any],
    *,
    required: Iterable[Capability] | None = None,
    preferred_engine: str | None = None,
    allow_chaining: bool = True,
) -> RoutingPlan:
    """Build an ordered engine chain covering `required` (or the derived set).

    The primary engine is the one covering the most required capabilities, so a
    single well-matched engine is preferred over an unnecessary chain. Leftover
    capabilities are then assigned, most constraining first, to the engines that
    support them.
    """
    requirements = derive_requirements(form_data)
    needed = frozenset(required) if required is not None else requirements.capabilities
    candidates = registry.engines_supporting({Capability.CHAT}) or registry.list()
    if not candidates:
        return RoutingPlan(unsupported=needed, requirements=requirements)

    if preferred_engine and registry.has(preferred_engine):
        candidates = [preferred_engine] + [name for name in candidates if name != preferred_engine]

    covers = {name: _engine_cover(name, registry, needed) for name in candidates}
    primary = max(
        candidates,
        key=lambda name: (len(covers[name]), -candidates.index(name)),
    )

    # Accumulate assignments before building the (frozen) selections, ordered
    # primary first, so the chain is deterministic and each engine's capability
    # set is final.
    ordered: list[str] = [primary]
    assignment: dict[str, set[Capability]] = {primary: set(covers[primary])}
    reason: dict[str, str] = {primary: f'primary engine covering {len(covers[primary])}/{len(needed)} capabilities'}

    remaining = set(needed) - assignment[primary]
    if allow_chaining:
        for capability in _CAPABILITY_PRIORITY:
            if capability not in remaining:
                continue
            extra = next((name for name in candidates if name != primary and capability in covers[name]), None)
            if extra is None:
                continue
            assignment.setdefault(extra, set()).add(capability)
            reason.setdefault(extra, f'follow-up for {capability.value}')
            if extra not in ordered:
                ordered.append(extra)
            remaining.discard(capability)

    selections = [
        EngineSelection(engine=name, capabilities=frozenset(assignment[name]), reason=reason[name]) for name in ordered
    ]

    return RoutingPlan(selections=selections, unsupported=frozenset(remaining), requirements=requirements)
