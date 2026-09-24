"""Secure runtime descriptors exposed to the CORTEX chat UI.

The chat UI already renders arbitrary ports through `PortPreview.svelte` /
`getPortProxyUrl`, and the backend already has a hardened reverse proxy in
`routers/terminals.py` (path sanitizing, JWT validation, HTTP + WebSocket).
Rather than inventing a second streaming path, AI-Manus screen/CDP endpoints
are described here so the *existing* proxy and preview components can be
reused.

Descriptors are templates, not live credentials: they carry the port and the
capability, and every URL is validated through `CortexPolicy` before it is
handed out. Screen streaming stays gated behind
`CORTEX_ALLOW_SCREEN_STREAMING`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from open_webui.inference.cortex.capabilities import Capability
from open_webui.inference.cortex.policy import CortexPolicy, CortexPolicyError

# Ports exposed by the ai-manus sandbox image (see sandbox/supervisord.conf).
AI_MANUS_RUNTIME_PORTS: dict[str, tuple[int, Capability]] = {
    'api': (8080, Capability.ARTIFACTS),
    'cdp': (9222, Capability.BROWSER),
    'vnc': (5901, Capability.SCREEN),
}


@dataclass(slots=True)
class RuntimeEndpoint:
    """A single reusable runtime endpoint (screen, browser, api)."""

    name: str
    port: int
    capability: Capability
    host: str
    scheme: str = 'http'
    description: str = ''

    @property
    def url(self) -> str:
        return f'{self.scheme}://{self.host}:{self.port}'


@dataclass(slots=True)
class RuntimeDescriptor:
    """Runtime endpoints for one engine session, already policy-validated."""

    engine: str
    session_id: str
    endpoints: list[RuntimeEndpoint] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)

    def endpoint(self, name: str) -> RuntimeEndpoint | None:
        return next((endpoint for endpoint in self.endpoints if endpoint.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            'engine': self.engine,
            'sessionId': self.session_id,
            'endpoints': [
                {
                    'name': endpoint.name,
                    'port': endpoint.port,
                    'capability': endpoint.capability.value,
                    'url': endpoint.url,
                    'description': endpoint.description,
                }
                for endpoint in self.endpoints
            ],
            # Kept explicit so an operator can see *why* the screen is absent
            # instead of assuming the engine failed to start it.
            'blocked': self.blocked,
        }


def describe_runtime(
    engine: str,
    session_id: str,
    host: str,
    *,
    policy: CortexPolicy | None = None,
    ports: dict[str, tuple[int, Capability]] | None = None,
) -> RuntimeDescriptor:
    """Build the runtime descriptor for a session, enforcing the policy.

    Endpoints refused by the policy are reported in `blocked` rather than
    raising, so one disabled capability (screen streaming) does not hide the
    rest of the runtime.
    """
    policy = policy or CortexPolicy.from_env()
    ports = ports if ports is not None else AI_MANUS_RUNTIME_PORTS
    descriptor = RuntimeDescriptor(engine=engine, session_id=session_id)

    for name, (port, capability) in ports.items():
        scheme = 'ws' if capability is Capability.SCREEN else 'http'
        try:
            policy.validate_endpoint(f'{scheme}://{host}:{port}', capability=capability)
        except CortexPolicyError as exc:
            descriptor.blocked[name] = exc.code
            continue
        descriptor.endpoints.append(
            RuntimeEndpoint(
                name=name,
                port=port,
                capability=capability,
                host=host,
                scheme=scheme,
                description=f'{capability.value} endpoint exposed by the {engine} sandbox',
            )
        )

    return descriptor
