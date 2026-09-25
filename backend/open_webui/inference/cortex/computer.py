"""Shared CORTEX computer plane: one sandbox both engines can use.

Historically the terminal, the filesystem, the browser and the screen lived
inside each engine: ai-manus shipped its own per-task container with permissive
defaults (passwordless VNC, ``--disable-web-security``, a mounted Docker
socket), and OpenHands ran in-process on the web host. That made the two
engines disagree about where the work happened and forced CORTEX to trust
whatever runtime an engine decided to start.

This module inverts that. CORTEX owns a *computer* -- a workspace plus the
process/filesystem handles -- and hands the very same handle to every engine
that takes part in a task. An engine is a reasoning loop; the computer is the
place the reasoning acts on. Swapping an engine therefore no longer swaps the
machine, and the security posture is decided once, here, instead of per engine.

The contract is deliberately small (exec, read, write, list) because that is
what the engines actually need, and every later addition is a capability in
``capabilities.py`` rather than a new engine-specific surface.
"""

from __future__ import annotations

import posixpath
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class ComputerKind(str, Enum):
    """How the computer is reached."""

    # A workspace directory on the host. Development only; refused by default.
    LOCAL = 'local'
    # A hardened container CORTEX starts and owns.
    DOCKER = 'docker'
    # An already-running sandbox reached over its HTTP API (the ai-manus
    # sandbox plane). Provisioned elsewhere; CORTEX only drives it.
    API = 'api'


class ComputerState(str, Enum):
    IDLE = 'idle'
    RUNNING = 'running'
    STOPPED = 'stopped'
    FAILED = 'failed'


class ComputerError(RuntimeError):
    """Raised for a refused or failed computer operation."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class ComputerSpec:
    """How to build the computer.

    Defaults are the tighterned posture: no network, read-only root filesystem,
    an unprivileged user and a bounded pid/command/output budget. A caller has
    to opt *in* to each relax, which is the opposite of the engine defaults we
    are moving away from.
    """

    workspace_dir: str
    kind: ComputerKind = ComputerKind.DOCKER
    image: str = 'cortex-computer:latest'
    cpus: float = 1.0
    memory_mb: int = 2048
    network_enabled: bool = False
    read_only_root: bool = True
    pids_limit: int = 256
    user: str = '1000:1000'
    tmpfs_size_mb: int = 256
    max_commands: int = 500
    max_output_bytes: int = 65536
    command_timeout_s: float = 120.0
    env: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'kind': self.kind.value,
            'image': self.image,
            'cpus': self.cpus,
            'memoryMb': self.memory_mb,
            'networkEnabled': self.network_enabled,
            'readOnlyRoot': self.read_only_root,
            'pidsLimit': self.pids_limit,
            'user': self.user,
            'maxCommands': self.max_commands,
            'maxOutputBytes': self.max_output_bytes,
            'commandTimeoutSeconds': self.command_timeout_s,
        }


@dataclass(slots=True)
class ExecResult:
    """Outcome of a command, with output already bounded."""

    command: str
    exit_code: int
    stdout: str = ''
    stderr: str = ''
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            'command': self.command,
            'exitCode': self.exit_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'truncated': self.truncated,
        }


@dataclass(slots=True)
class FileEntry:
    path: str
    size: int = 0
    is_dir: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {'path': self.path, 'size': self.size, 'isDir': self.is_dir}


class BaseComputer(ABC):
    """Workspace-scoped computer shared by the engines of a task.

    Subclasses implement the four private operations; everything shared --
    path containment, budget accounting and output bounding -- lives here so a
    provider cannot forget it.

    Path handling differs by kind. A host-backed provider (local, docker) stores
    the workspace in ``spec.workspace_dir`` and maps a relative path onto it. A
    remote provider (api) has no host directory: its ``root`` *is* the workspace
    path inside the remote machine, and ``resolve`` therefore yields the absolute
    remote path the API expects.
    """

    kind: ComputerKind
    #: True when ``workspace_dir`` is a path on the host rather than inside the
    #: remote machine. Host-backed providers join it with the relative path.
    host_backed: bool = True

    def __init__(self, spec: ComputerSpec, policy: Any = None) -> None:
        self.spec = spec
        self.policy = policy
        self.state = ComputerState.IDLE
        self._commands = 0
        # The workspace directory is the only place the computer may touch.
        self.root = posixpath.normpath(spec.workspace_dir)

    # ── lifecycle ────────────────────────────────────────────────────────
    async def start(self) -> None:
        await self._start()
        self.state = ComputerState.RUNNING

    async def stop(self) -> None:
        await self._stop()
        self.state = ComputerState.STOPPED

    async def _start(self) -> None:
        """Provider hook."""

    async def _stop(self) -> None:
        """Provider hook."""

    # ── public operations ────────────────────────────────────────────────
    async def exec(self, command: str, *, timeout: float | None = None) -> ExecResult:
        """Run a shell command inside the computer, subject to the budget."""
        if not command.strip():
            raise ComputerError('empty_command', 'Command must not be empty')
        self._authorize('computer.exec')
        if self._commands >= self.spec.max_commands:
            raise ComputerError(
                'budget_commands',
                f'Command budget exhausted ({self.spec.max_commands})',
            )
        self._commands += 1
        result = await self._exec(command, timeout or self.spec.command_timeout_s)
        return self._bound(result)

    async def read_file(self, path: str) -> str:
        self._authorize('computer.read_file')
        content = await self._read(self.resolve(path))
        if isinstance(content, bytes):
            return content.decode('utf-8', errors='replace')
        return content

    async def write_file(self, path: str, content: str) -> None:
        self._authorize('computer.write_file')
        await self._write(self.resolve(path), content)

    async def list_files(self, path: str = '.') -> list[FileEntry]:
        self._authorize('computer.list_files')
        return await self._list(self.resolve(path))

    # ── path containment ─────────────────────────────────────────────────
    def resolve(self, path: str) -> str:
        """Resolve a path, refusing any escape from the workspace.

        Absolute paths and ``..`` are rejected outright rather than clamped:
        silently rewriting ``/etc/passwd`` to a workspace file would hide an
        engine's attempt to leave the sandbox.

        A remote computer returns the absolute path *inside the remote machine*
        (``/workspace/foo``), which is what its API expects. A host-backed
        computer returns the path on the host. Both are containment-checked.
        """
        if not path:
            raise ComputerError('invalid_path', 'Path must not be empty')
        if '\\' in path:
            raise ComputerError('invalid_path', f'Backslashes are not allowed: {path!r}')

        if path.startswith('/'):
            if self.host_backed:
                raise ComputerError('absolute_path', f'Absolute paths are not allowed: {path!r}')
            # Remote: an absolute path is only accepted when it already points
            # inside the workspace. Anything else (``/etc/passwd``) is refused
            # rather than quietly remapped under the root, which would hide an
            # engine's attempt to leave the sandbox.
            if not path.startswith(f'{self.root}/') and path != self.root:
                raise ComputerError('absolute_path', f'Absolute paths must be inside {self.root}: {path!r}')
            relative = posixpath.relpath(path, self.root)
            if relative == '.':
                relative = ''
            normalized = posixpath.normpath(relative) if relative else '.'
        else:
            normalized = posixpath.normpath(path)

        if normalized == '..' or normalized.startswith('../'):
            raise ComputerError('path_traversal', f'Path escapes the workspace: {path!r}')

        full = posixpath.normpath(posixpath.join(self.root, normalized))
        if full != self.root and not full.startswith(f'{self.root}/'):
            raise ComputerError('path_traversal', f'Path escapes the workspace: {path!r}')
        return full

    # ── shared helpers ───────────────────────────────────────────────────
    def _authorize(self, tool: str) -> None:
        if self.policy is None:
            return
        authorize = getattr(self.policy, 'authorize_tool', None)
        if callable(authorize):
            authorize(tool)

    def _bound(self, result: ExecResult) -> ExecResult:
        limit = self.spec.max_output_bytes
        if len(result.stdout) <= limit and len(result.stderr) <= limit:
            return result
        result.stdout = result.stdout[:limit]
        result.stderr = result.stderr[:limit]
        result.truncated = True
        return result

    @property
    def commands_used(self) -> int:
        return self._commands

    def computer_url(self) -> str:
        """Base URL an engine should point at to reach this computer, if any.

        A host-backed computer has no network endpoint -- the engine runs in the
        same place -- so this is empty. A remote (API) computer returns the
        sandbox base URL, which is exactly what an engine that supports an
        external sandbox (ai-manus via ``SANDBOX_ADDRESS``) wants.
        """
        return ''

    def describe(self) -> dict[str, Any]:
        """Status surfaced to the UI; never carries credentials."""
        return {
            'kind': self.kind.value,
            'state': self.state.value,
            'workspace': self.spec.workspace_dir,
            'spec': self.spec.to_dict(),
            'commandsUsed': self._commands,
        }

    # ── provider hooks ───────────────────────────────────────────────────
    @abstractmethod
    async def _exec(self, command: str, timeout: float) -> ExecResult: ...

    @abstractmethod
    async def _read(self, path: str) -> str | bytes: ...

    @abstractmethod
    async def _write(self, path: str, content: str) -> None: ...

    @abstractmethod
    async def _list(self, path: str) -> list[FileEntry]: ...


@runtime_checkable
class ComputerProvider(Protocol):
    """Builds computers of one kind (local, docker, ...)."""

    name: str

    def create(self, spec: ComputerSpec, policy: Any = None) -> BaseComputer: ...


def default_tool_names() -> Iterable[str]:
    """Tool names the policy sees when a computer operation is authorized."""
    return ('computer.exec', 'computer.read_file', 'computer.write_file', 'computer.list_files')
