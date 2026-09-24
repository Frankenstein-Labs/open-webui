"""LLM provider resolution for the CORTEX engine adapters.

Engines need a model and its credentials. OpenRouter is a first-class,
explicitly supported provider here: it is selected by
`CORTEX_LLM_PROVIDER=openrouter`, or inferred when `OPENROUTER_API_KEY` is the
only provider credential present.

Credentials are read from the process environment only. Nothing in this module
writes, logs, or returns a key in a form that reaches an event stream — use
`redact` before putting provider metadata anywhere observable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1'
DEFAULT_OPENROUTER_MODEL = 'anthropic/claude-sonnet-4.5'
DEFAULT_OPENAI_MODEL = 'gpt-4o'

SUPPORTED_PROVIDERS = ('openrouter', 'openai', 'anthropic', 'local')


class ProviderConfigError(RuntimeError):
    """Raised when a provider cannot be resolved from the environment."""


@dataclass(slots=True)
class LlmProviderConfig:
    """Resolved provider settings ready to build an `LLM`."""

    provider: str
    model: str
    api_key: str = field(repr=False)
    base_url: str | None = None
    site_url: str | None = None
    app_name: str | None = None

    def to_llm_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for `openhands.sdk.LLM`.

        `api_key` is intentionally included here (the SDK needs it) but excluded
        from `repr`/`to_metadata`, so it cannot leak through logs or events.
        """
        kwargs: dict[str, Any] = {
            'model': self.model,
            'api_key': self.api_key,
            'usage_id': 'cortex-web',
            'drop_params': True,
        }
        if self.base_url:
            kwargs['base_url'] = self.base_url
        if self.provider == 'openrouter':
            if self.site_url:
                kwargs['openrouter_site_url'] = self.site_url
            if self.app_name:
                kwargs['openrouter_app_name'] = self.app_name
        return kwargs

    def to_metadata(self) -> dict[str, Any]:
        """Safe-to-observe description: never contains the API key."""
        return {
            'provider': self.provider,
            'model': self.model,
            'baseUrl': self.base_url,
            'hasApiKey': bool(self.api_key),
        }


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, '').strip()
        if value:
            return value
    return ''


def _normalize_model(provider: str, model: str) -> str:
    """Apply the LiteLLM provider prefix expected by the OpenHands SDK.

    OpenRouter models are addressed as `openrouter/<vendor>/<model>`. A model
    that already carries a provider prefix is left untouched so an operator can
    pin a different route deliberately.
    """
    if provider == 'openrouter' and not model.startswith('openrouter/'):
        return f'openrouter/{model}'
    return model


def resolve_provider_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LlmProviderConfig:
    """Resolve provider settings from explicit arguments then the environment.

    Precedence for each field: explicit argument > `OPENHANDS_*` override >
    provider-specific variable > generic fallback. This keeps the pre-existing
    `OPENHANDS_*` configuration working unchanged.
    """
    provider = (provider or os.getenv('CORTEX_LLM_PROVIDER', '')).strip().lower()
    if not provider:
        provider = 'openrouter' if os.getenv('OPENROUTER_API_KEY') else 'openai'
    if provider not in SUPPORTED_PROVIDERS:
        raise ProviderConfigError(
            f'Unsupported CORTEX_LLM_PROVIDER {provider!r}; expected one of {", ".join(SUPPORTED_PROVIDERS)}'
        )

    if provider == 'openrouter':
        resolved_model = model or _first_env('OPENHANDS_MODEL', 'OPENROUTER_MODEL', 'CORTEX_LLM_MODEL')
        resolved_model = resolved_model or DEFAULT_OPENROUTER_MODEL
        resolved_key = api_key or _first_env('OPENHANDS_API_KEY', 'OPENROUTER_API_KEY')
        resolved_base = (
            base_url or _first_env('OPENHANDS_BASE_URL', 'OPENROUTER_BASE_URL') or DEFAULT_OPENROUTER_BASE_URL
        )
        site_url = os.getenv('OPENROUTER_SITE_URL', '').strip() or None
        app_name = os.getenv('OPENROUTER_APP_NAME', '').strip() or 'CORTEX Web'
    elif provider == 'openai':
        resolved_model = model or _first_env('OPENHANDS_MODEL', 'OPENAI_MODEL') or DEFAULT_OPENAI_MODEL
        resolved_key = api_key or _first_env('OPENHANDS_API_KEY', 'OPENAI_API_KEY')
        resolved_base = base_url or _first_env('OPENHANDS_BASE_URL', 'OPENAI_API_BASE') or None
        site_url = app_name = None
    elif provider == 'anthropic':
        resolved_model = model or _first_env('OPENHANDS_MODEL', 'ANTHROPIC_MODEL') or 'claude-sonnet-4-5'
        resolved_key = api_key or _first_env('OPENHANDS_API_KEY', 'ANTHROPIC_API_KEY')
        resolved_base = base_url or _first_env('OPENHANDS_BASE_URL', 'ANTHROPIC_API_BASE') or None
        site_url = app_name = None
    else:  # local: any OpenAI-compatible server (vLLM, Ollama, LM Studio, ...)
        resolved_model = model or _first_env('OPENHANDS_MODEL', 'CORTEX_LLM_MODEL')
        resolved_key = api_key or _first_env('OPENHANDS_API_KEY', 'CORTEX_LLM_API_KEY') or 'not-required'
        resolved_base = base_url or _first_env('OPENHANDS_BASE_URL', 'CORTEX_LLM_BASE_URL')
        if not resolved_base:
            raise ProviderConfigError('The local provider requires CORTEX_LLM_BASE_URL (or OPENHANDS_BASE_URL)')
        if not resolved_model:
            raise ProviderConfigError('The local provider requires CORTEX_LLM_MODEL (or OPENHANDS_MODEL)')
        site_url = app_name = None

    if not resolved_model:
        raise ProviderConfigError(f'Provider {provider!r} resolved no model')
    if not resolved_key:
        raise ProviderConfigError(
            f'Provider {provider!r} resolved no API key; set the provider credential in the environment'
        )

    return LlmProviderConfig(
        provider=provider,
        model=_normalize_model(provider, resolved_model),
        api_key=resolved_key,
        base_url=resolved_base,
        site_url=site_url,
        app_name=app_name,
    )


def redact(value: Any) -> Any:
    """Replace anything that looks like an API key with a placeholder.

    Applied to provider metadata before it is attached to a CORTEX event, so an
    accidental `str(config)` can never publish a credential.
    """
    if isinstance(value, dict):
        return {key: ('<redacted>' if _looks_secret_key(key) else redact(item)) for key, item in value.items()}
    if isinstance(value, str) and value.startswith(('sk-', 'sk-or-', 'sk-ant-')):
        return '<redacted>'
    return value


def _looks_secret_key(key: str) -> bool:
    lowered = key.lower()
    return 'key' in lowered or 'token' in lowered or 'secret' in lowered or 'password' in lowered
