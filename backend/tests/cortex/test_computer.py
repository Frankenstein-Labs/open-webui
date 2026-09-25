"""Tests for the shared computer core: containment, budgets, policy, local run."""

from __future__ import annotations

import asyncio

import pytest
from conftest import _backend  # noqa: F401  (ensures open_webui is importable)

from open_webui.inference.cortex.computer import (
    BaseComputer,
    ComputerError,
    ComputerKind,
    ComputerSpec,
    ComputerState,
)
from open_webui.inference.cortex.computer_local import LocalComputerProvider
from open_webui.inference.cortex.computer_policy import ComputerPolicy, ComputerPolicyError


def run(coro):
    return asyncio.run(coro)


def local_spec(tmp_path, **overrides):
    defaults = {'workspace_dir': str(tmp_path), 'kind': ComputerKind.LOCAL}
    defaults.update(overrides)
    return ComputerSpec(**defaults)


def local_policy(**overrides):
    defaults = {'allow_local': True}
    defaults.update(overrides)
    return ComputerPolicy(**defaults)


# ── path containment ─────────────────────────────────────────────────────
def test_resolve_keeps_paths_inside_the_workspace(tmp_path):
    computer = LocalComputerProvider().create(local_spec(tmp_path), local_policy())
    assert computer.resolve('src/app.py') == f'{tmp_path}/src/app.py'
    assert computer.resolve('./src/../src/app.py') == f'{tmp_path}/src/app.py'


@pytest.mark.parametrize('path', ['/etc/passwd', '../escape', 'a/../../escape', ''])
def test_resolve_refuses_escapes(tmp_path, path):
    computer = LocalComputerProvider().create(local_spec(tmp_path), local_policy())
    with pytest.raises(ComputerError):
        computer.resolve(path)


def test_resolve_refuses_backslashes(tmp_path):
    computer = LocalComputerProvider().create(local_spec(tmp_path), local_policy())
    with pytest.raises(ComputerError) as excinfo:
        computer.resolve('a\\..\\b')
    assert excinfo.value.code == 'invalid_path'


def test_remote_computer_accepts_absolute_paths_inside_its_root():
    from open_webui.inference.cortex.computer_api import SandboxApiComputer

    spec = ComputerSpec(workspace_dir='/workspace', kind=ComputerKind.API)
    computer = SandboxApiComputer(spec)
    assert computer.resolve('/workspace/src/app.py') == '/workspace/src/app.py'
    assert computer.resolve('src/app.py') == '/workspace/src/app.py'
    with pytest.raises(ComputerError):
        computer.resolve('/etc/passwd')


# ── budgets and output bounding ──────────────────────────────────────────
def test_command_budget_is_enforced(tmp_path):
    computer = LocalComputerProvider().create(local_spec(tmp_path, max_commands=1), local_policy())
    run(computer.start())
    first = run(computer.exec('echo one'))
    assert first.ok
    with pytest.raises(ComputerError) as excinfo:
        run(computer.exec('echo two'))
    assert excinfo.value.code == 'budget_commands'


def test_output_is_bounded_and_flagged(tmp_path):
    computer = LocalComputerProvider().create(local_spec(tmp_path, max_output_bytes=16), local_policy())
    run(computer.start())
    result = run(computer.exec('printf "0123456789abcdefghij"'))
    assert result.truncated is True
    assert len(result.stdout) == 16


def test_empty_command_is_refused(tmp_path):
    computer = LocalComputerProvider().create(local_spec(tmp_path), local_policy())
    run(computer.start())
    with pytest.raises(ComputerError) as excinfo:
        run(computer.exec('   '))
    assert excinfo.value.code == 'empty_command'


# ── policy ───────────────────────────────────────────────────────────────
def test_policy_refuses_local_by_default():
    policy = ComputerPolicy()
    with pytest.raises(ComputerPolicyError) as excinfo:
        policy.validate(ComputerSpec(workspace_dir='/tmp/ws', kind=ComputerKind.LOCAL))
    assert excinfo.value.code == 'local_refused'


def test_policy_refuses_api_by_default():
    policy = ComputerPolicy()
    with pytest.raises(ComputerPolicyError) as excinfo:
        policy.validate(ComputerSpec(workspace_dir='/workspace', kind=ComputerKind.API))
    assert excinfo.value.code == 'api_refused'


def test_policy_refuses_network_and_unknown_images():
    policy = ComputerPolicy(allowed_images=frozenset({'cortex-computer:latest'}))
    with pytest.raises(ComputerPolicyError) as network:
        policy.validate(ComputerSpec(workspace_dir='/tmp/ws', kind=ComputerKind.DOCKER, network_enabled=True))
    assert network.value.code == 'network_refused'

    with pytest.raises(ComputerPolicyError) as image:
        policy.validate(ComputerSpec(workspace_dir='/tmp/ws', kind=ComputerKind.DOCKER, image='evil:latest'))
    assert image.value.code == 'image_not_allowed'


def test_policy_refuses_root_user_and_excess_resources():
    policy = ComputerPolicy()
    with pytest.raises(ComputerPolicyError) as root:
        policy.validate(ComputerSpec(workspace_dir='/tmp/ws', kind=ComputerKind.DOCKER, user='0:0'))
    assert root.value.code == 'root_user_refused'

    with pytest.raises(ComputerPolicyError) as cpus:
        policy.validate(ComputerSpec(workspace_dir='/tmp/ws', kind=ComputerKind.DOCKER, cpus=99.0))
    assert cpus.value.code == 'cpu_limit'


def test_docker_hardening_args_are_restrictive():
    policy = ComputerPolicy()
    spec = ComputerSpec(workspace_dir='/tmp/ws', kind=ComputerKind.DOCKER)
    args = policy.docker_hardening_args(spec)
    joined = ' '.join(args)

    assert '--cap-drop ALL' in joined
    assert '--security-opt no-new-privileges' in joined
    assert '--network none' in joined
    assert '--read-only' in joined
    assert '--user 1000:1000' in joined
    assert '--pids-limit 256' in joined
    # The Docker socket must never be mounted: that is the privilege escalation
    # the engines' defaults handed to the workload.
    assert 'docker.sock' not in joined


def test_docker_hardening_allows_network_only_when_asked():
    policy = ComputerPolicy(allow_network=True)
    spec = ComputerSpec(workspace_dir='/tmp/ws', kind=ComputerKind.DOCKER, network_enabled=True)
    assert '--network none' not in ' '.join(policy.docker_hardening_args(spec))


# ── local computer end to end ────────────────────────────────────────────
def test_local_computer_executes_and_roundtrips_files(tmp_path):
    computer = LocalComputerProvider().create(local_spec(tmp_path), local_policy())
    run(computer.start())
    assert computer.state is ComputerState.RUNNING

    run(computer.write_file('notes/hello.txt', 'hello cortex'))
    assert run(computer.read_file('notes/hello.txt')) == 'hello cortex'

    listing = run(computer.list_files('notes'))
    assert [entry.path for entry in listing] == ['notes/hello.txt']
    assert listing[0].size == len('hello cortex')

    result = run(computer.exec('cat notes/hello.txt'))
    assert result.ok
    assert 'hello cortex' in result.stdout

    assert computer.commands_used == 1
    run(computer.stop())
    assert computer.state is ComputerState.STOPPED


def test_local_computer_reports_nonzero_exit(tmp_path):
    computer = LocalComputerProvider().create(local_spec(tmp_path), local_policy())
    run(computer.start())
    result = run(computer.exec('exit 3'))
    assert result.exit_code == 3
    assert not result.ok


def test_describe_exposes_no_credentials(tmp_path):
    spec = local_spec(tmp_path, env={'SECRET_TOKEN': 'super-secret-value'})
    computer = LocalComputerProvider().create(spec, local_policy())
    described = computer.describe()
    assert 'SECRET_TOKEN' not in str(described)
    assert 'super-secret-value' not in str(described)


def test_base_computer_cannot_be_instantiated_without_hooks(tmp_path):
    class Incomplete(BaseComputer):
        kind = ComputerKind.LOCAL

    with pytest.raises(TypeError):
        Incomplete(local_spec(tmp_path))
