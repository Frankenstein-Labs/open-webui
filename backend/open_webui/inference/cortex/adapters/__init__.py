"""Engine adapters. Engine-specific types stay behind these modules."""

from open_webui.inference.cortex.adapters.ai_manus import AiManusAdapter
from open_webui.inference.cortex.adapters.base import BaseEngineAdapter
from open_webui.inference.cortex.adapters.openhands import OpenHandsAdapter

__all__ = ['AiManusAdapter', 'BaseEngineAdapter', 'OpenHandsAdapter']
