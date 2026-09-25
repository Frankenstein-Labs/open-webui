"""Security posture for the shared CORTEX computer.

``CortexPolicy`` guards *endpoints and tools*; this guards the *machine*. They
are separate on purpose: a task can be allowed to reach a host yet forbidden
from having networking inside its own sandbox, and an operator should be able
to reason about each independently.

The defaults encode what we refuse to inherit from the engines: networking off,
read-only root, a non-root user, dropped capabilities and no ability to start
further containers. Docker-specific hardening is expressed as the argv
fragments to apply, so the argument-building is unit-testable without a Docker
daemon.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


class ComputerPolicyError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


# Capabilities a sandboxed workload is allowed to keep. Everything else is
# dropped; these are enough to run a shell, a filesystem and a browser.
_DROP_ALL_CAPS = 'ALL'
_KEPT_CAPS = ('CHOWN', 'DAC_OVERRIDE', 'FOWNER', 'SETUID', 'SETGID')


@dataclass(slots=True)
class ComputerPolicy:
    """What the computer is allowed to be and do."""

    allow_local: bool = False
    allow_api: bool = False
    allowed_images: frozenset[str] = field(default_factory=frozenset)
    allow_network: bool = False
    allow_read_only_root: bool = True
    require_non_root: bool = True
    max_cpus: float = 2.0
    max_memory_mb: int = 4096
    max_commands: int = 500
    command_timeout_s: float = 120.0
    max_output_bytes: int = 65536
    workspace_root: str = ''

    @classmethod
    def from_env(cls) -> 'ComputerPolicy':
        images = os.getenv('CORTEX_COMPUTER_ALLOWED_IMAGES', '')
        return cls(
            allow_local=os.getenv('CORTEX_COMPUTER_ALLOW_LOCAL', 'false').lower() == 'true',
            allow_api=os.getenv('CORTEX_COMPUTER_ALLOW_API', 'false').lower() == 'true',
            allowed_images=frozenset(image.strip() for image in images.split(',') if image.strip()),
            allow_network=os.getenv('CORTEX_COMPUTER_ALLOW_NETWORK', 'false').lower() == 'true',
            require_non_root=os.getenv('CORTEX_COMPUTER_REQUIRE_NON_ROOT', 'true').lower() != 'false',
            max_cpus=float(os.getenv('CORTEX_COMPUTER_MAX_CPUS', '2.0')),
            max_memory_mb=int(os.getenv('CORTEX_COMPUTER_MAX_MEMORY_MB', '4096')),
            max_commands=int(os.getenv('CORTEX_COMPUTER_MAX_COMMANDS', '500')),
            command_timeout_s=float(os.getenv('CORTEX_COMPUTER_COMMAND_TIMEOUT_S', '120')),
            max_output_bytes=int(os.getenv('CORTEX_COMPUTER_MAX_OUTPUT_BYTES', '65536')),
            workspace_root=os.getenv('CORTEX_COMPUTER_WORKSPACE_ROOT', ''),
        )

    # ── validation ───────────────────────────────────────────────────────
    def validate(self, spec: Any) -> None:
        """Reject a spec that asks for more than the policy permits."""
        kind = getattr(getattr(spec, 'kind', None), 'value', None)

        if kind == 'docker':
            if self.allowed_images and spec.image not in self.allowed_images:
                raise ComputerPolicyError(
                    'image_not_allowed',
                    f'Image {spec.image!r} is not in CORTEX_COMPUTER_ALLOWED_IMAGES',
                )
            if spec.network_enabled and not self.allow_network:
                raise ComputerPolicyError('network_refused', 'Networking is disabled by CORTEX computer policy')
            if not spec.read_only_root and self.allow_read_only_root is True:
                # allow_read_only_root acts as a floor: a writable root is only
                # possible when the operator has turned the floor off.
                raise ComputerPolicyError('read_only_root_required', 'A writable root filesystem is not permitted')
        elif kind == 'local' and not self.allow_local:
            raise ComputerPolicyError(
                'local_refused',
                'The local computer is disabled; set CORTEX_COMPUTER_ALLOW_LOCAL=true for development only',
            )
        elif kind == 'api' and not self.allow_api:
            raise ComputerPolicyError(
                'api_refused',
                'Remote API computers are disabled; set CORTEX_COMPUTER_ALLOW_API=true when a trusted sandbox plane exists',
            )

        if spec.cpus > self.max_cpus:
            raise ComputerPolicyError('cpu_limit', f'Requested {spec.cpus} CPUs, maximum is {self.max_cpus}')
        if spec.memory_mb > self.max_memory_mb:
            raise ComputerPolicyError('memory_limit', f'Requested {spec.memory_mb} MB, maximum is {self.max_memory_mb}')
        if self.require_non_root and spec.user.split(':')[0] in {'0', 'root', ''}:
            raise ComputerPolicyError('root_user_refused', 'The computer must run as a non-root user')
        if spec.max_commands > self.max_commands:
            raise ComputerPolicyError('command_budget', f'Command budget exceeds the policy maximum')
        if spec.max_output_bytes > self.max_output_bytes:
            raise ComputerPolicyError('output_limit', 'Output budget exceeds the policy maximum')

    # ── docker hardening ─────────────────────────────────────────────────
    def docker_hardening_args(self, spec: Any) -> list[str]:
        """argv fragments that harden a container for this spec.

        Returned as a list so the caller can splice them into a run command and
        tests can assert them without a daemon.
        """
        args = [
            '--cap-drop',
            _DROP_ALL_CAPS,
        ]
        for capability in _KEPT_CAPS:
            args += ['--cap-add', capability]
        args += [
            '--security-opt',
            'no-new-privileges',
            '--pids-limit',
            str(spec.pids_limit),
            '--user',
            spec.user,
        ]
        if spec.read_only_root:
            args += ['--read-only', '--tmpfs', f'/tmp:size={spec.tmpfs_size_mb}m']
        if not spec.network_enabled:
            args += ['--network', 'none']
        # Never mount the container runtime socket: that is the privilege
        # escalation the engines' defaults handed to the workload.
        args += ['--volume', f'{spec.workspace_dir}:/workspace']
        return args

    def to_dict(self) -> dict[str, Any]:
        return {
            'allowLocal': self.allow_local,
            'allowApi': self.allow_api,
            'allowedImages': sorted(self.allowed_images),
            'allowNetwork': self.allow_network,
            'requireNonRoot': self.require_non_root,
            'maxCpus': self.max_cpus,
            'maxMemoryMb': self.max_memory_mb,
            'maxCommands': self.max_commands,
            'commandTimeoutSeconds': self.command_timeout_s,
            'maxOutputBytes': self.max_output_bytes,
        }
