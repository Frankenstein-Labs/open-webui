"""Tests for LLM provider resolution.

Focus: OpenRouter is explicitly supported, `OPENHANDS_*` overrides keep working,
and a credential can never surface through provider metadata.
"""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from open_webui.inference.cortex.providers import (  # noqa: E402
    DEFAULT_OPENROUTER_BASE_URL,
    ProviderConfigError,
    redact,
    resolve_provider_config,
)

_ENV_KEYS = (
    'CORTEX_LLM_PROVIDER',
    'CORTEX_LLM_MODEL',
    'CORTEX_LLM_API_KEY',
    'CORTEX_LLM_BASE_URL',
    'OPENROUTER_API_KEY',
    'OPENROUTER_MODEL',
    'OPENROUTER_BASE_URL',
    'OPENROUTER_SITE_URL',
    'OPENROUTER_APP_NAME',
    'OPENAI_API_KEY',
    'OPENAI_MODEL',
    'OPENAI_API_BASE',
    'ANTHROPIC_API_KEY',
    'ANTHROPIC_MODEL',
    'OPENHANDS_API_KEY',
    'OPENHANDS_BASE_URL',
    'OPENHANDS_MODEL',
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    yield


def test_openrouter_selected_when_only_openrouter_key_present(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'sk-or-v1-example')
    config = resolve_provider_config()
    assert config.provider == 'openrouter'
    assert config.base_url == DEFAULT_OPENROUTER_BASE_URL
    assert config.model.startswith('openrouter/')


def test_openrouter_selected_explicitly(monkeypatch):
    monkeypatch.setenv('CORTEX_LLM_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'example-key')
    monkeypatch.setenv('OPENROUTER_MODEL', 'deepseek/deepseek-chat')
    config = resolve_provider_config()
    assert config.provider == 'openrouter'
    assert config.model == 'openrouter/deepseek/deepseek-chat'


def test_openrouter_model_keeps_existing_prefix(monkeypatch):
    config = resolve_provider_config(provider='openrouter', model='openrouter/x/y', api_key='k')
    assert config.model == 'openrouter/x/y'


def test_openrouter_attribution_headers_are_forwarded(monkeypatch):
    monkeypatch.setenv('OPENROUTER_SITE_URL', 'https://cortex.example')
    config = resolve_provider_config(provider='openrouter', api_key='k')
    kwargs = config.to_llm_kwargs()
    assert kwargs['openrouter_site_url'] == 'https://cortex.example'
    assert kwargs['openrouter_app_name'] == 'CORTEX Web'
    assert kwargs['base_url'] == DEFAULT_OPENROUTER_BASE_URL


def test_openshands_env_overrides_still_win(monkeypatch):
    monkeypatch.setenv('CORTEX_LLM_PROVIDER', 'openrouter')
    monkeypatch.setenv('OPENROUTER_API_KEY', 'openrouter-key')
    monkeypatch.setenv('OPENHANDS_API_KEY', 'override-key')
    monkeypatch.setenv('OPENHANDS_MODEL', 'override/model')
    monkeypatch.setenv('OPENHANDS_BASE_URL', 'https://custom.example/api/v1')
    config = resolve_provider_config()
    assert config.api_key == 'override-key'
    assert config.model == 'openrouter/override/model'
    assert config.base_url == 'https://custom.example/api/v1'


def test_legacy_openai_path_is_unchanged(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-key')
    config = resolve_provider_config()
    assert config.provider == 'openai'
    assert config.base_url is None
    assert config.model == 'gpt-4o'


def test_local_provider_requires_a_base_url(monkeypatch):
    monkeypatch.setenv('CORTEX_LLM_PROVIDER', 'local')
    monkeypatch.setenv('CORTEX_LLM_MODEL', 'my-model')
    with pytest.raises(ProviderConfigError, match='CORTEX_LLM_BASE_URL'):
        resolve_provider_config()


def test_local_provider_does_not_require_an_api_key(monkeypatch):
    config = resolve_provider_config(
        provider='local',
        model='my-model',
        base_url='http://localhost:8000/v1',
    )
    assert config.api_key == 'not-required'
    assert config.model == 'my-model'


def test_missing_credential_raises_with_a_clear_message():
    with pytest.raises(ProviderConfigError, match='no API key'):
        resolve_provider_config()


def test_unknown_provider_is_rejected():
    with pytest.raises(ProviderConfigError, match='Unsupported'):
        resolve_provider_config(provider='gemini')


def test_metadata_never_contains_the_api_key():
    config = resolve_provider_config(provider='openrouter', api_key='sk-or-v1-supersecret')
    metadata = config.to_metadata()
    assert 'supersecret' not in str(metadata)
    assert metadata['hasApiKey'] is True
    assert 'api_key' not in metadata


def test_repr_never_contains_the_api_key():
    config = resolve_provider_config(provider='openrouter', api_key='sk-or-v1-supersecret')
    assert 'supersecret' not in repr(config)


def test_redact_masks_secret_keys_and_key_like_values():
    payload = {
        'api_key': 'sk-or-v1-abc',
        'nested': {'Authorization': 'sk-xyz', 'model': 'openrouter/x'},
        'model': 'openrouter/x',
        'apiKey': 'anything',
    }
    masked = redact(payload)
    assert masked['api_key'] == '<redacted>'
    assert masked['nested']['Authorization'] == '<redacted>'
    assert masked['nested']['model'] == 'openrouter/x'
    assert masked['model'] == 'openrouter/x'
    assert masked['apiKey'] == '<redacted>'
