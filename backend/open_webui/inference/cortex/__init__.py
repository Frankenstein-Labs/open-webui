"""CORTEX engine bridge: capability-routed orchestration across engines.

Contracts mirror CORTEX-Software-Agent-SDK (`EngineAdapter`, `EngineRegistry`,
`CortexEvent`/protocol v1, `TaskGraph`, `Policy`, `AgentCapability`) so the
TypeScript orchestrator and this Python bridge stay interoperable.
"""

from open_webui.inference.cortex.capabilities import (
    AgentCapability,
    Capability,
    EngineAdapter,
    EngineContext,
    EngineRegistry,
    EngineSession,
)
from open_webui.inference.cortex.computer import (
    BaseComputer,
    ComputerError,
    ComputerKind,
    ComputerSpec,
    ExecResult,
    FileEntry,
)
from open_webui.inference.cortex.computers import ComputerRecord, ComputerRegistry, get_computer_registry
from open_webui.inference.cortex.orchestrator import CortexOrchestrator, EngineUnavailableError
from open_webui.inference.cortex.policy import CortexPolicy, CortexPolicyError
from open_webui.inference.cortex.providers import LlmProviderConfig, ProviderConfigError, resolve_provider_config
from open_webui.inference.cortex.protocol import PROTOCOL_VERSION, CortexEvent, make_event
from open_webui.inference.cortex.routing import (
    EngineSelection,
    RoutingPlan,
    TaskRequirements,
    derive_requirements,
    plan_route,
)

__all__ = [
    'PROTOCOL_VERSION',
    'AgentCapability',
    'BaseComputer',
    'Capability',
    'ComputerError',
    'ComputerKind',
    'ComputerRecord',
    'ComputerRegistry',
    'ComputerSpec',
    'CortexEvent',
    'CortexOrchestrator',
    'CortexPolicy',
    'CortexPolicyError',
    'EngineAdapter',
    'EngineContext',
    'EngineRegistry',
    'EngineSelection',
    'EngineSession',
    'EngineUnavailableError',
    'ExecResult',
    'FileEntry',
    'LlmProviderConfig',
    'ProviderConfigError',
    'RoutingPlan',
    'TaskRequirements',
    'derive_requirements',
    'get_computer_registry',
    'make_event',
    'plan_route',
    'resolve_provider_config',
]
