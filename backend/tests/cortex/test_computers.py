"""Tests for the computer registry: ownership, lifecycle and engine attachment."""

from __future__ import annotations

import asyncio

import pytest
from conftest import _backend  # noqa: F401

from open_webui.inference.cortex.adapters.ai_manus import AiManusAdapter
from open_webui.inference.cortex.adapters.base import attach_computer
from open_webui.inference.cortex.computer import ComputerKind, ComputerSpec
from open_webui.inference.cortex.computer_local import LocalComputerProvider
from open_webui.inference.cortex.computer_policy import ComputerPolicy, ComputerPolicyError
from open_webui.inference.cortex.computers import ComputerRegistry


def run(coro):
    return asyncio.run(coro)


def local_registry(**policy_overrides):
    policy_overrides.setdefault('allow_local', True)
    return ComputerRegistry(
        ComputerPolicy(**policy_overrides),
        providers={ComputerKind.LOCAL: LocalComputerProvider()},
    )


def spec(tmp_path):
    return ComputerSpec(workspace_dir=str(tmp_path), kind=ComputerKind.LOCAL)


# ── creation and identity ────────────────────────────────────────────────
def test_create_registers_a_live_computer(tmp_path):
    registry = local_registry()
    record = run(registry.create(user_id='user-1', task_id='task-1', workspace_dir=str(tmp_path), spec=spec(tmp_path)))
    assert record.computer.state.value == 'running'
    assert registry.size == 1
    assert registry.get(record.id) is record


def test_identifiers_are_unguessable_and_unique(tmp_path):
    registry = local_registry()
    ids = {
        run(registry.create(user_id='u', task_id='t', workspace_dir=str(tmp_path), spec=spec(tmp_path))).id
        for _ in range(25)
    }
    assert len(ids) == 25
    # token_urlsafe(24) is ~32 chars; anything short would be enumerable.
    assert all(len(value) >= 32 for value in ids)


def test_creation_is_refused_when_policy_forbids_the_kind(tmp_path):
    registry = ComputerRegistry(ComputerPolicy(), providers={ComputerKind.LOCAL: LocalComputerProvider()})
    with pytest.raises(ComputerPolicyError) as excinfo:
        run(registry.create(user_id='u', task_id='t', workspace_dir=str(tmp_path), spec=spec(tmp_path)))
    assert excinfo.value.code == 'local_refused'


# ── ownership ────────────────────────────────────────────────────────────
def test_authorize_accepts_the_owner(tmp_path):
    registry = local_registry()
    record = run(registry.create(user_id='owner', task_id='t', workspace_dir=str(tmp_path), spec=spec(tmp_path)))
    assert registry.authorize_for_user(record.id, 'owner') is record


def test_authorize_refuses_another_user_and_unknown_ids_indistinguishably(tmp_path):
    registry = local_registry()
    record = run(registry.create(user_id='owner', task_id='t', workspace_dir=str(tmp_path), spec=spec(tmp_path)))

    with pytest.raises(ComputerPolicyError) as foreign:
        registry.authorize_for_user(record.id, 'intruder')
    with pytest.raises(ComputerPolicyError) as unknown:
        registry.authorize_for_user('does-not-exist', 'intruder')

    # Same code and message: the response cannot be used to probe which ids exist.
    assert foreign.value.code == unknown.value.code
    assert str(foreign.value) == str(unknown.value)


def test_attach_engine_is_idempotent(tmp_path):
    registry = local_registry()
    record = run(registry.create(user_id='u', task_id='t', workspace_dir=str(tmp_path), spec=spec(tmp_path)))
    registry.attach_engine(record.id, 'openhands')
    registry.attach_engine(record.id, 'openhands')
    registry.attach_engine(record.id, 'ai-manus')
    assert record.engines == ['openhands', 'ai-manus']


# ── lifecycle ────────────────────────────────────────────────────────────
def test_release_is_idempotent_and_removes_the_record(tmp_path):
    registry = local_registry()
    record = run(registry.create(user_id='u', task_id='t', workspace_dir=str(tmp_path), spec=spec(tmp_path)))
    run(registry.release(record.id))
    assert registry.get(record.id) is None
    run(registry.release(record.id))
    assert registry.size == 0


def test_release_for_task_only_touches_that_task(tmp_path):
    registry = local_registry()
    run(registry.create(user_id='u', task_id='task-a', workspace_dir=str(tmp_path), spec=spec(tmp_path)))
    run(registry.create(user_id='u', task_id='task-b', workspace_dir=str(tmp_path), spec=spec(tmp_path)))
    run(registry.release_for_task('task-a'))
    assert registry.size == 1


def test_list_records_never_leaks_secrets(tmp_path):
    registry = local_registry()
    secret_spec = ComputerSpec(workspace_dir=str(tmp_path), kind=ComputerKind.LOCAL, env={'API_KEY': 'top-secret'})
    run(registry.create(user_id='u', task_id='t', workspace_dir=str(tmp_path), spec=secret_spec))
    assert 'top-secret' not in str(registry.list_records())


# ── engine attachment helper ─────────────────────────────────────────────
class _RecordingAdapter:
    """Stands in for a real engine adapter; records what it was told."""

    name = 'recording'

    def __init__(self):
        self.kwargs = []

    def configure_computer(self, **kwargs):
        self.kwargs.append(kwargs)
        return 'configured'


def test_attach_computer_prefers_native_env_configuration(tmp_path):
    registry = local_registry()
    record = run(registry.create(user_id='u', task_id='t', workspace_dir=str(tmp_path), spec=spec(tmp_path)))
    adapter = _RecordingAdapter()

    result = attach_computer(adapter, record, metadata={})
    assert result == 'configured'
    # ai-manus and OpenHands both configure via a base URL; the local computer
    # has none, so nothing is passed.
    assert adapter.kwargs == [{'base_url': ''}]


def test_attach_computer_sets_engine_agnostic_metadata_when_no_hook(tmp_path):
    registry = local_registry()
    record = run(registry.create(user_id='u', task_id='t', workspace_dir=str(tmp_path), spec=spec(tmp_path)))

    class Plain:
        name = 'plain'

    metadata: dict = {}
    attach_computer(Plain(), record, metadata=metadata)
    assert metadata['computer']['id'] == record.id
    assert metadata['computer']['kind'] == 'local'


def test_attach_computer_returns_computer_url_for_remote_kinds(tmp_path):
    from open_webui.inference.cortex.computers import ComputerRecord

    record = ComputerRecord(
        id='cid',
        computer=_FakeComputer(ComputerKind.API),
        user_id='u',
        task_id='t',
    )
    adapter = _RecordingAdapter()
    attach_computer(adapter, record, metadata={})
    assert adapter.kwargs[0]['base_url'] == 'http://sandbox:8080'


class _FakeComputer:
    def __init__(self, kind):
        self.kind = kind
        self.state = None

    def computer_url(self):
        return 'http://sandbox:8080' if self.kind is ComputerKind.API else ''


# ── the ai-manus adapter accepts native configuration ────────────────────
def test_ai_manus_adapter_accepts_base_url_override():
    adapter = AiManusAdapter()
    adapter.configure_computer(base_url='http://cortex-computer.internal:8080')
    assert adapter._base_url_override == 'http://cortex-computer.internal:8080'
