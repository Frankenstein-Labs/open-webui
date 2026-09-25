"""Local computer: a workspace directory on the host, for development and tests.

This is *not* an isolation boundary and is refused by default. It exists so the
orchestration can be exercised end to end without a Docker daemon, and so a
developer can run a task locally on purpose with
``CORTEX_COMPUTER_ALLOW_LOCAL=true``.

Even so it keeps the same containment as the containerised computer: every path
goes through ``BaseComputer.resolve`` and the working directory is always the
workspace root, so an engine cannot wander the host filesystem just because the
computer happens to be local.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from typing import Any

from open_webui.inference.cortex.computer import BaseComputer, ComputerKind, ExecResult, FileEntry

_CONTROL_CHARS = ('\x00',)


class LocalComputer(BaseComputer):
    """Runs commands with the backend's own OS process, inside one workspace."""

    kind = ComputerKind.LOCAL

    def __init__(self, spec: Any, policy: Any = None) -> None:
        super().__init__(spec, policy)

    async def _start(self) -> None:
        os.makedirs(self.root, exist_ok=True)

    async def _exec(self, command: str, timeout: float) -> ExecResult:
        env = {**os.environ, **{str(k): str(v) for k, v in self.spec.env.items()}}
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=self.root,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            # A new session so a command cannot signal the backend's process group.
            start_new_session=True,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            return ExecResult(
                command=command,
                exit_code=124,
                stderr=f'Command timed out after {timeout}s',
            )
        return ExecResult(
            command=command,
            exit_code=process.returncode or 0,
            stdout=stdout.decode('utf-8', errors='replace'),
            stderr=stderr.decode('utf-8', errors='replace'),
        )

    async def _read(self, path: str) -> bytes:
        for char in _CONTROL_CHARS:
            if char in path:
                raise ValueError('Invalid path')
        with open(path, 'rb') as handle:
            return handle.read()

    async def _write(self, path: str, content: str) -> None:
        os.makedirs(os.path.dirname(path) or self.root, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as handle:
            handle.write(content)

    async def _list(self, path: str) -> list[FileEntry]:
        if not os.path.isdir(path):
            return []
        entries: list[FileEntry] = []
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            entries.append(
                FileEntry(
                    path=os.path.relpath(full, self.root),
                    size=os.path.getsize(full) if os.path.isfile(full) else 0,
                    is_dir=os.path.isdir(full),
                )
            )
        return entries

    async def _stop(self) -> None:
        # The workspace is kept: it is the task's output, and removing it would
        # discard exactly what the next engine in the chain needs to read.
        pass

    def remove_workspace(self) -> None:
        """Explicit destructive helper, never called by the lifecycle."""
        shutil.rmtree(self.root, ignore_errors=True)


class LocalComputerProvider:
    """Provider for ``ComputerKind.LOCAL``."""

    name = 'local'

    def create(self, spec: Any, policy: Any = None) -> LocalComputer:
        if spec.kind is not ComputerKind.LOCAL:
            spec.kind = ComputerKind.LOCAL
        return LocalComputer(spec, policy)
