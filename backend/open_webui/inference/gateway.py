"""Neutral inference gateway used while the OpenDevin engine is prepared.

The previous Ollama/OpenAI provider implementations have intentionally been
removed. This module keeps a small explicit compatibility contract so that
Open WebUI's files, retrieval, tools, MCP and administration routes can still
start without silently invoking a model provider.
"""

from typing import Any

from fastapi import HTTPException, status
from fastapi.responses import StreamingResponse

from open_webui.inference.engine_router import (
    EngineConfigurationError,
    generate_chat_completion as route_chat_completion,
)

ENGINE_REMOVED_MESSAGE = (
    'Le moteur d’inférence OpenWebUI a été supprimé. Le moteur OpenDevin/OpenHands sera branché ultérieurement.'
)


class InferenceEngineUnavailable(RuntimeError):
    """Raised by non-HTTP call sites when no inference engine is configured."""

    def __init__(self) -> None:
        super().__init__(ENGINE_REMOVED_MESSAGE)


def inference_engine_unavailable() -> None:
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=ENGINE_REMOVED_MESSAGE,
    )


class GenerateEmbedForm:
    """Legacy request shape kept until the OpenDevin adapter is installed."""

    def __init__(self, **_: Any) -> None:
        pass


async def generate_chat_completion(
    request: Any = None,
    form_data: dict[str, Any] | None = None,
    user: Any = None,
    **_: Any,
) -> Any:
    """Route chat tasks to ai-manus or OpenHands using the task-aware selector."""
    del request
    if form_data is None:
        inference_engine_unavailable()
    # The router reads the mode from metadata first, then the top level.
    mode = form_data.get('conversation_mode')
    if mode and 'conversation_mode' not in (form_data.get('metadata') or {}):
        form_data['metadata'] = {**(form_data.get('metadata') or {}), 'conversation_mode': mode}

    # CORTEX Agent orchestration is opt-in and capability-routed. When the
    # feature flag is off, the legacy single-engine path below is unchanged.
    from open_webui.inference.cortex.bridge import should_orchestrate

    if should_orchestrate(form_data):
        return await _orchestrated_completion(form_data, user)

    try:
        result = await route_chat_completion(
            form_data,
            stream=bool(form_data.get('stream')),
        )
    except EngineConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    if hasattr(result, '__aiter__'):
        return StreamingResponse(result, media_type='text/event-stream')
    return result


async def embed(*_: Any, **__: Any) -> Any:
    inference_engine_unavailable()


async def _orchestrated_completion(form_data: dict[str, Any], user: Any) -> Any:
    """Delegate to the CORTEX orchestrator in Agent mode (feature-flagged)."""
    from fastapi.responses import StreamingResponse

    from open_webui.inference.cortex.bridge import route_agent_completion

    result = await route_agent_completion(form_data, user, stream=bool(form_data.get('stream')))
    if hasattr(result, '__aiter__'):
        return StreamingResponse(result, media_type='text/event-stream')
    return result


def describe_cortex_engines() -> dict[str, Any]:
    """Diagnostics: registered engines and their declared capabilities."""
    from open_webui.inference.cortex.bridge import describe_engines

    return describe_engines()


async def embeddings(*_: Any, **__: Any) -> Any:
    inference_engine_unavailable()


async def get_all_models(*_: Any, **__: Any) -> dict[str, list[Any]]:
    raise InferenceEngineUnavailable()


async def get_all_models_responses(*_: Any, **__: Any) -> list[Any]:
    raise InferenceEngineUnavailable()


async def count_anthropic_tokens(*_: Any, **__: Any) -> int:
    raise InferenceEngineUnavailable()


async def get_anthropic_token_count_target(*_: Any, **__: Any) -> Any:
    raise InferenceEngineUnavailable()


async def get_openai_connection(*_: Any, **__: Any) -> tuple[str, str, dict[str, Any]]:
    raise InferenceEngineUnavailable()


async def publish_model_provider_request_failed(*_: Any, **__: Any) -> None:
    raise InferenceEngineUnavailable()


def _clean_proxy_headers(headers: Any) -> dict[str, Any]:
    return dict(headers)
