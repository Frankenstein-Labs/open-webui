"""CORTEX security policy applied to every engine session.

Mirrors `packages/core/.../policy` in CORTEX-Software-Agent-SDK: the policy
lives in CORTEX, not in the engines. Adapters must consult it before exposing a
runtime endpoint or running a tool, which is what lets us *not* inherit
ai-manus' permissive defaults (passwordless VNC, `--disable-web-security`,
mounted Docker socket, sudo without password).

Nothing here is a substitute for network isolation: it is the in-process guard
that refuses obviously dangerous endpoints and enforces budgets.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from open_webui.inference.cortex.capabilities import Capability

# Endpoints reachable without an explicit allowlist entry.
_PRIVATE_LABELS = ('localhost', '127.0.0.1', '::1')


class CortexPolicyError(RuntimeError):
    """Raised when an action violates the CORTEX policy."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class CortexPolicy:
    """Session-scoped security policy: endpoints, tools, and budgets."""

    allowed_endpoint_hosts: frozenset[str] = field(default_factory=frozenset)
    allow_private_networks: bool = True
    allowed_tools: frozenset[str] | None = None  # None means "engine default"
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    max_tool_calls: int = 200
    max_runtime_seconds: int = 3600
    max_tokens: int = 0  # 0 means unbounded
    require_isolated_runtime: bool = True
    allow_screen_streaming: bool = False
    _tool_calls: int = 0

    @classmethod
    def from_env(cls) -> 'CortexPolicy':
        hosts = os.getenv('CORTEX_ALLOWED_ENDPOINT_HOSTS', '')
        return cls(
            allowed_endpoint_hosts=frozenset(host.strip().lower() for host in hosts.split(',') if host.strip()),
            allow_private_networks=os.getenv('CORTEX_ALLOW_PRIVATE_NETWORKS', 'true').lower() != 'false',
            denied_tools=frozenset(
                tool.strip() for tool in os.getenv('CORTEX_DENIED_TOOLS', '').split(',') if tool.strip()
            ),
            max_tool_calls=int(os.getenv('CORTEX_MAX_TOOL_CALLS', '200')),
            max_runtime_seconds=int(os.getenv('CORTEX_MAX_RUNTIME_SECONDS', '3600')),
            require_isolated_runtime=os.getenv('CORTEX_REQUIRE_ISOLATED_RUNTIME', 'true').lower() != 'false',
            allow_screen_streaming=os.getenv('CORTEX_ALLOW_SCREEN_STREAMING', 'false').lower() == 'true',
        )

    # ── endpoint validation ──────────────────────────────────────────────
    def validate_endpoint(self, url: str, *, capability: Capability | None = None) -> str:
        """Return a normalized endpoint URL or raise `CortexPolicyError`.

        Blocks the cloud metadata address and, unless explicitly allowed,
        routable addresses: an engine runtime must stay on the private network
        CORTEX controls.
        """
        parsed = urlparse(url)
        if parsed.scheme not in {'http', 'https', 'ws', 'wss'}:
            raise CortexPolicyError('endpoint_scheme', f'Unsupported endpoint scheme: {parsed.scheme!r}')
        host = (parsed.hostname or '').lower()
        if not host:
            raise CortexPolicyError('endpoint_host', f'Endpoint has no host: {url!r}')

        if capability is Capability.SCREEN and not self.allow_screen_streaming:
            raise CortexPolicyError('screen_streaming_disabled', 'Screen streaming is disabled by CORTEX policy')

        if host in self.allowed_endpoint_hosts or host in _PRIVATE_LABELS:
            return url.rstrip('/')

        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            # A DNS name that was not explicitly allowlisted is refused: CORTEX
            # must not let a task choose its own egress target.
            raise CortexPolicyError('endpoint_not_allowed', f'Host {host!r} is not in the CORTEX allowlist') from None

        if address.is_link_local or address.is_loopback or address.is_multicast or address.is_reserved:
            raise CortexPolicyError('endpoint_address', f'Address {host!r} is not permitted')
        if address.is_private and self.allow_private_networks:
            return url.rstrip('/')
        raise CortexPolicyError('endpoint_not_allowed', f'Address {host!r} is not in the CORTEX allowlist')

    # ── tool validation ──────────────────────────────────────────────────
    def authorize_tool(self, tool: str) -> None:
        if tool in self.denied_tools:
            raise CortexPolicyError('tool_denied', f'Tool {tool!r} is denied by CORTEX policy')
        if self.allowed_tools is not None and tool not in self.allowed_tools:
            raise CortexPolicyError('tool_not_allowed', f'Tool {tool!r} is not in the CORTEX allowlist')
        if self._tool_calls >= self.max_tool_calls:
            raise CortexPolicyError('budget_tool_calls', f'Tool-call budget exhausted ({self.max_tool_calls})')
        self._tool_calls += 1

    @property
    def tool_calls_used(self) -> int:
        return self._tool_calls

    def to_dict(self) -> dict[str, Any]:
        return {
            'allowedEndpointHosts': sorted(self.allowed_endpoint_hosts),
            'allowPrivateNetworks': self.allow_private_networks,
            'deniedTools': sorted(self.denied_tools),
            'maxToolCalls': self.max_tool_calls,
            'maxRuntimeSeconds': self.max_runtime_seconds,
            'requireIsolatedRuntime': self.require_isolated_runtime,
            'allowScreenStreaming': self.allow_screen_streaming,
        }
