"""Registry binding a shared computer to the engines of a task.

Two concerns live here.

**Ownership.** A computer is a process with a shell, so its identifier is a
security boundary, not just a key. Identifiers are generated with
``secrets.token_urlsafe`` (never a counter, never derived from a user id) and
every lookup goes through ``authorize_for_user``. Serving a terminal or screen
for someone else's computer is a privilege escalation, so ownership is checked
before any handle is returned rather than at the route.

**Lifecycle.** The computer is created once per task and attached to each engine
that takes part in the chain, then released when the task ends. Releasing is
idempotent and best effort: a task that fails must not strand a container.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import threading
from dataclasses import dataclass, field
from typing import Any

from open_webui.inference.cortex.computer import (
    BaseComputer,
    ComputerKind,
    ComputerSpec,
    ComputerState,
)
from open_webui.inference.cortex.computer_api import SandboxApiProvider
from open_webui.inference.cortex.computer_docker import DockerComputerProvider
from open_webui.inference.cortex.computer_local import LocalComputerProvider
from open_webui.inference.cortex.computer_policy import ComputerPolicy, ComputerPolicyError
from open_webui.inference.cortex.runtime import describe_runtime

log = logging.getLogger(__name__)

_ID_BYTES = 24


@dataclass(slots=True)
class ComputerRecord:
    """Registry entry: the computer plus who owns it."""

    id: str
    computer: BaseComputer
    user_id: str
    task_id: str
    conversation_id: str = ''
    workspace_id: str = ''
    engines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            'id': self.id,
            'userId': self.user_id,
            'taskId': self.task_id,
            'conversationId': self.conversation_id,
            'engines': sorted(self.engines),
            **self.computer.describe(),
        }


class ComputerRegistry:
    """In-memory registry of live computers, keyed by an unguessable id."""

    def __init__(
        self,
        policy: ComputerPolicy | None = None,
        *,
        providers: dict[ComputerKind, Any] | None = None,
    ) -> None:
        self.policy = policy or ComputerPolicy.from_env()
        self._providers = providers or {
            ComputerKind.DOCKER: DockerComputerProvider(),
            ComputerKind.LOCAL: LocalComputerProvider(),
            ComputerKind.API: SandboxApiProvider(),
        }
        self._records: dict[str, ComputerRecord] = {}
        self._lock = threading.Lock()

    # ── creation ─────────────────────────────────────────────────────────
    def build_spec(
        self,
        workspace_dir: str,
        *,
        kind: ComputerKind | None = None,
        image: str | None = None,
        network_enabled: bool = False,
        env: dict[str, str] | None = None,
    ) -> ComputerSpec:
        """Build a spec from the environment, already bounded by the policy."""
        return ComputerSpec(
            workspace_dir=workspace_dir,
            kind=kind or _default_kind(),
            image=image or _default_image(),
            cpus=self.policy.max_cpus,
            memory_mb=self.policy.max_memory_mb,
            network_enabled=network_enabled and self.policy.allow_network,
            read_only_root=True,
            max_commands=self.policy.max_commands,
            command_timeout_s=self.policy.command_timeout_s,
            max_output_bytes=self.policy.max_output_bytes,
            env=dict(env or {}),
        )

    async def create(
        self,
        *,
        user_id: str,
        task_id: str,
        workspace_dir: str,
        conversation_id: str = '',
        workspace_id: str = '',
        spec: ComputerSpec | None = None,
        kind: ComputerKind | None = None,
    ) -> ComputerRecord:
        """Provision a computer, start it, and register it under a fresh id."""
        spec = spec or self.build_spec(workspace_dir, kind=kind)
        self.policy.validate(spec)

        provider = self._providers.get(spec.kind)
        if provider is None:
            raise ComputerPolicyError('provider_missing', f'No computer provider for kind {spec.kind.value!r}')

        computer = provider.create(spec, self.policy)
        await computer.start()

        record = ComputerRecord(
            id=secrets.token_urlsafe(_ID_BYTES),
            computer=computer,
            user_id=str(user_id or ''),
            task_id=task_id,
            conversation_id=conversation_id,
            workspace_id=workspace_id,
        )
        with self._lock:
            self._records[record.id] = record
        log.info('CORTEX computer %s provisioned for task %s', record.id, task_id)
        return record

    # ── lookup / authorization ───────────────────────────────────────────
    def get(self, computer_id: str) -> ComputerRecord | None:
        with self._lock:
            return self._records.get(computer_id)

    def authorize_for_user(self, computer_id: str, user_id: str) -> ComputerRecord:
        """Return the record only if it belongs to *user_id*.

        A missing computer and another user's computer raise the same error, so
        the response cannot be used to probe which ids exist.
        """
        record = self.get(computer_id)
        if record is None or record.user_id != str(user_id or ''):
            raise ComputerPolicyError('computer_forbidden', 'Computer not found or not owned by this user')
        return record

    def attach_engine(self, computer_id: str, engine: str) -> None:
        """Record that an engine took part in the task using this computer."""
        record = self.get(computer_id)
        if record is not None and engine not in record.engines:
            record.engines.append(engine)

    # ── description for the UI ───────────────────────────────────────────
    def describe_for_user(self, computer_id: str, user_id: str) -> dict[str, Any]:
        """Status plus the runtime endpoints the existing proxy can expose.

        The endpoints are described, never tunnelled here: the hardened
        ``routers/terminals.py`` proxy and ``PortPreview.svelte`` already know
        how to stream a port, and this intentionally does not add a second path.
        """
        record = self.authorize_for_user(computer_id, user_id)
        payload = record.to_dict()
        payload['runtime'] = describe_runtime(
            'cortex-computer',
            record.id,
            _host_for(record.computer),
            policy=_cortex_policy_for_endpoints(),
        ).to_dict()
        return payload

    # ── teardown ─────────────────────────────────────────────────────────
    async def release(self, computer_id: str) -> None:
        record = self.get(computer_id)
        if record is None:
            return
        with self._lock:
            self._records.pop(computer_id, None)
        try:
            await asyncio.wait_for(record.computer.stop(), timeout=60)
        except Exception:  # teardown must never mask the task outcome
            log.warning('Failed to stop CORTEX computer %s cleanly', computer_id, exc_info=True)

    async def release_for_task(self, task_id: str) -> None:
        with self._lock:
            ids = [record_id for record_id, record in self._records.items() if record.task_id == task_id]
        for record_id in ids:
            await self.release(record_id)

    async def shutdown(self) -> None:
        for record_id in list(self._records):
            await self.release(record_id)

    def list_records(self) -> list[dict[str, Any]]:
        """Admin view; passwords and env values are never included."""
        with self._lock:
            return [record.to_dict() for record in self._records.values()]

    @property
    def size(self) -> int:
        return len(self._records)


def _default_kind() -> ComputerKind:
    import os

    value = os.getenv('CORTEX_COMPUTER_KIND', 'docker').strip().lower()
    try:
        return ComputerKind(value)
    except ValueError:
        return ComputerKind.DOCKER


def _default_image() -> str:
    import os

    return os.getenv('CORTEX_COMPUTER_IMAGE', 'cortex-computer:latest')


def _host_for(computer: BaseComputer) -> str:
    """Host an endpoint descriptor should point at.

    Containers are reached through the runtime's own network name; the local
    computer is reached on loopback. The value is a *description*: the proxy
    resolves it, and the endpoint policy still vets it.
    """
    if computer.kind is ComputerKind.DOCKER:
        return getattr(computer, 'container_name', lambda: 'cortex-computer')()
    return '127.0.0.1'


def _cortex_policy_for_endpoints() -> Any:
    from open_webui.inference.cortex.policy import CortexPolicy

    return CortexPolicy.from_env()


_GLOBAL_LOCK = threading.Lock()
_GLOBAL: ComputerRegistry | None = None


def get_computer_registry() -> ComputerRegistry:
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = ComputerRegistry()
        return _GLOBAL


def reset_computer_registry() -> None:
    """Test hook."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        _GLOBAL = None
