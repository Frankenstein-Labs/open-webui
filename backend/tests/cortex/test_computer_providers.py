"""Tests for the containerised and remote-API computer providers.

Docker and a live sandbox are not available in CI, so the *transport* is faked
here (a scripted runner / transport). Everything above it -- argv construction,
hardening flags, envelope parsing, exit-code mapping and path mapping -- is the
real code path under test.
"""

from __future__ import annotations

import asyncio

import pytest
from conftest import _backend  # noqa: F401

from open_webui.inference.cortex.computer import ComputerError, ComputerKind, ComputerSpec, ExecResult
from open_webui.inference.cortex.computer_api import SandboxApiComputer
from open_webui.inference.cortex.computer_docker import DockerComputer
from open_webui.inference.cortex.computer_policy import ComputerPolicy


def run(coro):
    return asyncio.run(coro)


class ScriptedRunner:
    """Records argv and replays canned results in order."""

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[list[str]] = []
        self.stdin: list[bytes | None] = []

    async def run(self, argv, *, timeout, stdin=None):
        self.calls.append(list(argv))
        self.stdin.append(stdin)
        if self.results:
            return self.results.pop(0)
        return ExecResult(command=' '.join(argv), exit_code=0)


def ok(stdout='', stderr=''):
    return ExecResult(command='', exit_code=0, stdout=stdout, stderr=stderr)


def fail(stderr='boom'):
    return ExecResult(command='', exit_code=1, stderr=stderr)


def docker_computer(tmp_path, runner, **spec_overrides):
    overrides = {'workspace_dir': str(tmp_path), 'kind': ComputerKind.DOCKER}
    overrides.update(spec_overrides)
    spec = ComputerSpec(**overrides)
    return DockerComputer(spec, ComputerPolicy(), runner=runner)


# ── docker: startup and hardening ────────────────────────────────────────
def test_docker_start_inspects_image_and_runs_hardened_container(tmp_path):
    runner = ScriptedRunner([ok(), ok(), ok('container-abc123')])
    computer = docker_computer(tmp_path, runner)
    run(computer.start())

    assert computer.container_id == 'container-abc123'
    # rm -f leftover, image inspect, docker run
    assert runner.calls[0][:2] == ['rm', '-f']
    assert runner.calls[1][:2] == ['image', 'inspect']

    run_argv = runner.calls[2]
    joined = ' '.join(run_argv)
    assert run_argv[0] == 'run'
    assert '--cap-drop ALL' in joined
    assert '--security-opt no-new-privileges' in joined
    assert '--network none' in joined
    assert '--read-only' in joined
    assert f'{tmp_path}:/workspace' in joined
    assert 'docker.sock' not in joined


def test_docker_start_fails_loudly_when_image_is_missing(tmp_path):
    runner = ScriptedRunner([ok(), fail('no such image')])
    computer = docker_computer(tmp_path, runner)
    with pytest.raises(ComputerError) as excinfo:
        run(computer.start())
    assert excinfo.value.code == 'image_unavailable'


def test_docker_start_fails_when_runtime_returns_no_id(tmp_path):
    runner = ScriptedRunner([ok(), ok(), ok('')])
    computer = docker_computer(tmp_path, runner)
    with pytest.raises(ComputerError) as excinfo:
        run(computer.start())
    assert excinfo.value.code == 'docker_start'


def test_docker_exec_maps_exit_code_and_streams(tmp_path):
    runner = ScriptedRunner([ok(), ok(), ok('cid'), ExecResult('', 7, 'out', 'err')])
    computer = docker_computer(tmp_path, runner)
    run(computer.start())
    result = run(computer.exec('false'))
    assert result.exit_code == 7
    assert result.stdout == 'out'
    assert result.stderr == 'err'


def test_docker_read_and_write_use_the_container_path(tmp_path):
    runner = ScriptedRunner([ok(), ok(), ok('cid'), ok('file-content'), ok()])
    computer = docker_computer(tmp_path, runner)
    run(computer.start())

    assert run(computer.read_file('src/app.py')) == 'file-content'
    read_argv = runner.calls[3]
    assert '/workspace/src/app.py' in ' '.join(read_argv)

    run(computer.write_file('src/app.py', 'new'))
    write_argv = runner.calls[4]
    assert '/workspace/src/app.py' in ' '.join(write_argv)
    assert runner.stdin[4] == b'new'


def test_docker_list_parses_find_output(tmp_path):
    runner = ScriptedRunner([ok(), ok(), ok('cid'), ok('a.txt\t3\tf\nsub\t0\td\n')])
    computer = docker_computer(tmp_path, runner)
    run(computer.start())
    entries = run(computer.list_files('.'))
    assert [(entry.path, entry.is_dir, entry.size) for entry in entries] == [
        ('a.txt', False, 3),
        ('sub', True, 0),
    ]


def test_docker_container_name_is_deterministic(tmp_path):
    a = docker_computer(tmp_path, ScriptedRunner([]))
    b = docker_computer(tmp_path, ScriptedRunner([]))
    assert a.container_name() == b.container_name()
    assert a.container_name().startswith('cortex-computer-')


def test_docker_stop_removes_the_container_once(tmp_path):
    runner = ScriptedRunner([ok(), ok(), ok('cid'), ok()])
    computer = docker_computer(tmp_path, runner)
    run(computer.start())
    run(computer.stop())
    assert runner.calls[-1][:2] == ['rm', '-f']
    run(computer.stop())
    # A second stop is a no-op: nothing else was invoked.
    assert len(runner.calls) == 4


# ── api: envelope parsing over the ai-manus sandbox protocol ─────────────
class ScriptedTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    async def post(self, path, payload, *, timeout):
        self.calls.append((path, payload))
        if self.responses:
            return self.responses.pop(0)
        return {'success': True, 'data': {}}


def ok_envelope(data):
    return {'success': True, 'message': 'ok', 'data': data}


def error_envelope(message):
    return {'success': False, 'message': message, 'data': None}


def api_computer(transport):
    spec = ComputerSpec(workspace_dir='/workspace', kind=ComputerKind.API)
    return SandboxApiComputer(spec, ComputerPolicy(allow_api=True), transport=transport)


def test_api_start_checks_supervisor(tmp_path):
    transport = ScriptedTransport([ok_envelope([{'name': 'app', 'statename': 'RUNNING'}])])
    computer = api_computer(transport)
    run(computer.start())
    assert transport.calls[0][0] == '/supervisor/status'


def test_api_exec_maps_envelope_to_result():
    transport = ScriptedTransport(
        [
            ok_envelope([]),
            ok_envelope(
                {'session_id': 's', 'command': 'ls', 'status': 'completed', 'returncode': 0, 'output': 'files'}
            ),
        ]
    )
    computer = api_computer(transport)
    run(computer.start())
    result = run(computer.exec('ls'))
    assert result.ok
    assert result.stdout == 'files'
    path, payload = transport.calls[1]
    assert path == '/shell/exec'
    assert payload['command'] == 'ls'
    # The session id is stable per computer, so the shell stays persistent.
    assert payload['id']


def test_api_exec_reports_sandbox_failure():
    transport = ScriptedTransport([ok_envelope([]), error_envelope('tmux exploded')])
    computer = api_computer(transport)
    run(computer.start())
    with pytest.raises(ComputerError) as excinfo:
        run(computer.exec('ls'))
    assert excinfo.value.code == 'sandbox_error'
    assert 'tmux exploded' in str(excinfo.value)


def test_api_file_roundtrip_uses_absolute_paths():
    transport = ScriptedTransport(
        [ok_envelope([]), ok_envelope({'content': 'body', 'file': '/workspace/a.txt'}), ok_envelope({})]
    )
    computer = api_computer(transport)
    run(computer.start())
    assert run(computer.read_file('a.txt')) == 'body'
    assert transport.calls[1][0] == '/file/read'
    assert transport.calls[1][1]['file'] == '/workspace/a.txt'


def test_api_list_maps_find_result():
    transport = ScriptedTransport([ok_envelope([]), ok_envelope({'path': '/workspace', 'files': ['a.txt', 'b.txt']})])
    computer = api_computer(transport)
    run(computer.start())
    entries = run(computer.list_files('.'))
    assert [entry.path for entry in entries] == ['a.txt', 'b.txt']


def test_api_computer_requires_a_base_url():
    spec = ComputerSpec(workspace_dir='/workspace', kind=ComputerKind.API)
    computer = SandboxApiComputer(spec, ComputerPolicy(allow_api=True))
    # No injected transport and no configured URL: starting must fail clearly
    # rather than attempting an HTTP call to an empty host.
    import os

    os.environ.pop('CORTEX_COMPUTER_API_URL', None)
    with pytest.raises(ComputerError) as excinfo:
        run(computer.start())
    assert excinfo.value.code == 'sandbox_unconfigured'
