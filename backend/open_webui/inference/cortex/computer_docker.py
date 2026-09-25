"""Docker computer: the real isolation boundary, hardened by the CORTEX policy.

The engines' original defaults are exactly what we refuse here: a passwordless
VNC server, Chromium started with ``--disable-web-security``, a bind-mounted
Docker socket and passwordless ``sudo``. A container built from this module has
networking off unless the policy allows it, a read-only root with a bounded
``/tmp``, no new privileges, a dropped capability set and an unprivileged user.

Everything reaches Docker through a ``ContainerRunner`` rather than by calling
``docker`` inline. That keeps argv construction and output parsing unit-testable
(the tests drive a fake runner) and leaves room for a different transport later
(``podman``, a remote daemon) without touching this class.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import posixpath
from typing import Any, Protocol, runtime_checkable

from open_webui.inference.cortex.computer import (
    BaseComputer,
    ComputerError,
    ComputerKind,
    ExecResult,
    FileEntry,
)

WORKSPACE_MOUNT = '/workspace'


@runtime_checkable
class ContainerRunner(Protocol):
    """Executes a container-runtime CLI invocation."""

    async def run(self, argv: list[str], *, timeout: float, stdin: bytes | None = None) -> ExecResult: ...


class SubprocessContainerRunner:
    """Runs the ``docker`` CLI as a subprocess (the default transport)."""

    def __init__(self, binary: str = 'docker') -> None:
        self.binary = binary

    async def run(self, argv: list[str], *, timeout: float, stdin: bytes | None = None) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            self.binary,
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(input=stdin), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ExecResult(command=' '.join(argv), exit_code=124, stderr=f'Timed out after {timeout}s')
        return ExecResult(
            command=' '.join(argv),
            exit_code=process.returncode or 0,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
        )


class DockerComputer(BaseComputer):
    """A hardened container with one workspace bind-mounted at /workspace."""

    kind = ComputerKind.DOCKER

    def __init__(self, spec: Any, policy: Any = None, runner: ContainerRunner | None = None) -> None:
        super().__init__(spec, policy)
        self.runner = runner or SubprocessContainerRunner()
        self.container_id = ''

    # ── lifecycle ────────────────────────────────────────────────────────
    async def _start(self) -> None:
        os.makedirs(self.root, exist_ok=True)

        name = self.container_name()
        # Remove a leftover container with the same name so a retried session
        # does not fail on a name collision.
        await self.runner.run(['rm', '-f', name], timeout=30)

        inspect = await self.runner.run(['image', 'inspect', self.spec.image], timeout=60)
        if not inspect.ok:
            raise ComputerError('image_unavailable', f'Container image unavailable: {self.spec.image}')

        argv = ['run', '-d', '--name', name]
        argv += self._hardening_args()
        for key, value in self.spec.env.items():
            argv += ['-e', f'{key}={value}']
        argv += [
            '--cpus',
            str(self.spec.cpus),
            '--memory',
            f'{self.spec.memory_mb}m',
            '-w',
            WORKSPACE_MOUNT,
        ]
        argv += [self.spec.image, 'sleep', 'infinity']

        result = await self.runner.run(argv, timeout=120)
        if not result.ok:
            raise ComputerError('docker_start', f'Could not start the computer: {result.stderr.strip()}')
        self.container_id = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ''
        if not self.container_id:
            raise ComputerError('docker_start', 'Container runtime returned no container id')

    async def _stop(self) -> None:
        if not self.container_id:
            return
        await self.runner.run(['rm', '-f', self.container_id], timeout=60)
        self.container_id = ''

    # ── provider hooks ───────────────────────────────────────────────────
    async def _exec(self, command: str, timeout: float) -> ExecResult:
        argv = ['exec', '-i', self._require_container(), 'sh', '-lc', command]
        result = await self.runner.run(argv, timeout=timeout)
        return ExecResult(
            command=command,
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    async def _read(self, path: str) -> bytes:
        argv = ['exec', '-i', self._require_container(), 'sh', '-c', f'cat -- {_quote(self._inside(path))}']
        result = await self.runner.run(argv, timeout=self.spec.command_timeout_s)
        if not result.ok:
            raise ComputerError('read_failed', f'Could not read {path!r}: {result.stderr.strip()}')
        return result.stdout.encode('utf-8')

    async def _write(self, path: str, content: str) -> None:
        inside = self._inside(path)
        argv = [
            'exec',
            '-i',
            self._require_container(),
            'sh',
            '-c',
            f'mkdir -p -- {_quote(posixpath.dirname(inside))} && cat > {_quote(inside)}',
        ]
        result = await self.runner.run(argv, timeout=self.spec.command_timeout_s, stdin=content.encode('utf-8'))
        if not result.ok:
            raise ComputerError('write_failed', f'Could not write {path!r}: {result.stderr.strip()}')

    async def _list(self, path: str) -> list[FileEntry]:
        inside = self._inside(path)
        # -printf keeps the output machine-parseable; no reliance on ls columns.
        argv = [
            'exec',
            '-i',
            self._require_container(),
            'sh',
            '-c',
            f"find {_quote(inside)} -mindepth 1 -maxdepth 1 -printf '%P\\t%s\\t%y\\n'",
        ]
        result = await self.runner.run(argv, timeout=self.spec.command_timeout_s)
        if not result.ok:
            return []
        return _parse_find(result.stdout)

    # ── helpers ──────────────────────────────────────────────────────────
    def container_name(self) -> str:
        """Deterministic, filesystem-safe name so a retry does not leak containers."""
        digest = hashlib.sha256(self.root.encode('utf-8')).hexdigest()[:12]
        return f'cortex-computer-{digest}'

    def _hardening_args(self) -> list[str]:
        if self.policy is None:
            # No policy means no hardening opinion; the spec defaults still apply.
            args = ['--network', 'none'] if not self.spec.network_enabled else []
            return args
        return self.policy.docker_hardening_args(self.spec)

    def _inside(self, host_path: str) -> str:
        """Map a resolved host path to its location inside the container."""
        relative = posixpath.relpath(host_path, self.root)
        if relative == '.':
            return WORKSPACE_MOUNT
        return posixpath.normpath(posixpath.join(WORKSPACE_MOUNT, relative))

    def _require_container(self) -> str:
        if not self.container_id:
            raise ComputerError('not_started', 'The computer has not been started')
        return self.container_id


def _quote(value: str) -> str:
    """Single-quote a path for the container shell."""
    return "'" + value.replace("'", "'\\''") + "'"


def _parse_find(output: str) -> list[FileEntry]:
    entries: list[FileEntry] = []
    for line in output.splitlines():
        parts = line.split('\t')
        if len(parts) != 3:
            continue
        name, size, kind = parts
        if not name:
            continue
        try:
            parsed_size = int(size)
        except ValueError:
            parsed_size = 0
        entries.append(FileEntry(path=name, size=parsed_size, is_dir=kind == 'd'))
    return entries


class DockerComputerProvider:
    """Provider for ``ComputerKind.DOCKER``."""

    name = 'docker'

    def __init__(self, runner: ContainerRunner | None = None) -> None:
        self.runner = runner

    def create(self, spec: Any, policy: Any = None) -> DockerComputer:
        if spec.kind is not ComputerKind.DOCKER:
            spec.kind = ComputerKind.DOCKER
        return DockerComputer(spec, policy, runner=self.runner)
