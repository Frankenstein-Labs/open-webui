"""Computer provider speaking the existing ai-manus sandbox plane.

ai-manus already ships the machine we want: a separate `sandbox` service exposing

  * ``POST /api/v1/shell/{exec,view,wait,write,kill}``   (persistent tmux shell)
  * ``POST /api/v1/file/{read,write,replace,search,find,exists,delete,list}``
  * ``GET  /api/v1/supervisor/status``
  * CDP on ``:9222`` and VNC (websockified) on ``:5901``

and its backend consumes any such sandbox through the ``Sandbox`` protocol. With
``SANDBOX_ADDRESS`` set it stops creating its own container and talks to that
address instead.

Rather than invent a parallel contract, this provider adopts that wire protocol.
The consequence is the thing we actually wanted: **one machine, several
engines.** CORTEX can drive it directly, and ai-manus can be pointed at the very
same one with ``SANDBOX_ADDRESS``, so a task is not executed on two different
computers depending on which engine happened to win the routing.

The transport is injected so the envelope parsing and path mapping are unit
testable without a running sandbox.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from open_webui.inference.cortex.computer import (
    BaseComputer,
    ComputerError,
    ComputerKind,
    ExecResult,
    FileEntry,
)

# Where the sandbox keeps the shared work area; both engines read it from here.
SANDBOX_WORKSPACE = '/workspace'

_API_PREFIX = '/api/v1'


@runtime_checkable
class SandboxTransport(Protocol):
    """Minimal HTTP surface of the sandbox service."""

    async def post(self, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]: ...


class HttpxSandboxTransport:
    """Default transport: one pooled httpx client per sandbox."""

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip('/')

    async def post(self, path: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        import httpx

        async with httpx.AsyncClient(base_url=self.base_url, timeout=timeout) as client:
            response = await client.post(f'{_API_PREFIX}{path}', json=payload)
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict):
            raise ComputerError('sandbox_protocol', 'Sandbox returned a non-object response')
        return body


class SandboxApiComputer(BaseComputer):
    """A computer driven through the ai-manus sandbox HTTP API."""

    kind = ComputerKind.API
    # The sandbox has no host directory: paths are absolute inside that machine.
    host_backed = False

    def __init__(self, spec: Any, policy: Any = None, transport: SandboxTransport | None = None) -> None:
        super().__init__(spec, policy)
        self._transport = transport

    # ── lifecycle ────────────────────────────────────────────────────────
    async def _start(self) -> None:
        # The sandbox is provisioned elsewhere (CORTEX docker provider, or an
        # operator-managed host). We only verify it answers before use.
        if self._transport is None:
            base_url = self._base_url()
            if self.policy is not None:
                self.policy.validate_endpoint(base_url)
            self._transport = HttpxSandboxTransport(base_url)
        body = await self._call('/supervisor/status', {}, timeout=30.0, allow_failure=True)
        if body is not None and not body.get('success', True):
            raise ComputerError('sandbox_unavailable', 'Sandbox supervisor did not report success')

    async def _stop(self) -> None:
        # Never destroy a shared sandbox from here: ai-manus may still be using
        # it, and ownership of teardown belongs to whoever provisioned it.
        pass

    # ── provider hooks ───────────────────────────────────────────────────
    async def _exec(self, command: str, timeout: float) -> ExecResult:
        # A stable session id per computer makes the shell persistent, which is
        # what lets a later engine in a chain see the earlier engine's shell
        # state instead of starting from an empty environment.
        body = await self._call(
            '/shell/exec',
            {'id': self._session_id(), 'exec_dir': self.root, 'command': command},
            timeout=timeout,
        )
        data = body.get('data') or {}
        returncode = data.get('returncode')
        return ExecResult(
            command=command,
            exit_code=int(returncode) if returncode is not None else (0 if body.get('success') else 1),
            stdout=str(data.get('output') or ''),
            # The sandbox folds stderr into `output`; error detail lives in the
            # envelope message, so keep it visible.
            stderr='' if body.get('success') else str(body.get('message') or ''),
        )

    async def _read(self, path: str) -> bytes:
        body = await self._call('/file/read', {'file': path}, timeout=self.spec.command_timeout_s)
        data = body.get('data') or {}
        return str(data.get('content') or '').encode('utf-8')

    async def _write(self, path: str, content: str) -> None:
        await self._call(
            '/file/write',
            {'file': path, 'content': content},
            timeout=self.spec.command_timeout_s,
        )

    async def _list(self, path: str) -> list[FileEntry]:
        body = await self._call('/file/find', {'path': path, 'glob': '*'}, timeout=self.spec.command_timeout_s)
        data = body.get('data') or {}
        entries: list[FileEntry] = []
        for name in data.get('files') or []:
            if isinstance(name, str):
                entries.append(FileEntry(path=name, is_dir=False))
        return entries

    # ── helpers ──────────────────────────────────────────────────────────
    def _base_url(self) -> str:
        url = self.spec.env.get('SANDBOX_BASE_URL') or os.getenv('CORTEX_COMPUTER_API_URL', '')
        if not url:
            raise ComputerError('sandbox_unconfigured', 'No sandbox base URL configured for the API computer')
        return url.rstrip('/')

    def computer_url(self) -> str:
        """The sandbox base URL, which is what an engine needs to reach it."""
        try:
            return self._base_url()
        except ComputerError:
            return ''

    def _session_id(self) -> str:
        return f'cortex-{self.root.strip("/").replace("/", "-")}' or 'cortex-session'

    async def _call(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float,
        allow_failure: bool = False,
    ) -> dict[str, Any]:
        if self._transport is None:
            raise ComputerError('not_started', 'The computer has not been started')
        try:
            body = await self._transport.post(path, payload, timeout=timeout)
        except ComputerError:
            raise
        except Exception as exc:
            if allow_failure:
                return {}
            raise ComputerError('sandbox_request', f'Sandbox request failed: {exc}') from exc
        if isinstance(body, dict) and body.get('success') is False and not allow_failure:
            raise ComputerError('sandbox_error', str(body.get('message') or 'Sandbox rejected the request'))
        return body


class SandboxApiProvider:
    """Provider for ``ComputerKind.API`` (the ai-manus sandbox plane)."""

    name = 'sandbox-api'

    def __init__(self, transport: SandboxTransport | None = None) -> None:
        self.transport = transport

    def create(self, spec: Any, policy: Any = None) -> SandboxApiComputer:
        if spec.kind is not ComputerKind.API:
            spec.kind = ComputerKind.API
        return SandboxApiComputer(spec, policy, transport=self.transport)
