"""Test bootstrap for the CORTEX bridge suite.

CI runs `pytest` from the repository root with only the test dependencies
installed. Importing `open_webui` by name would execute
`open_webui/__init__.py`, which imports `typer` and `uvicorn` (CLI
dependencies) and fails there.

This module installs `open_webui` as a filesystem-backed package whose
`__init__.py` is never executed, then lets the normal import machinery resolve
`open_webui.inference.cortex`. It runs at import time because pytest imports
conftest.py before collecting the test modules.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import sys
import types
from pathlib import Path


def _locate_backend() -> Path:
    """Find the backend root by walking up to the directory holding open_webui."""
    candidate = Path(__file__).resolve().parent
    while candidate != candidate.parent:
        if (candidate / 'open_webui' / 'inference').is_dir():
            return candidate
        candidate = candidate.parent
    raise RuntimeError('Could not locate the backend root containing open_webui/inference')


_backend = _locate_backend()

if 'open_webui' not in sys.modules:
    _package = types.ModuleType('open_webui')
    _package.__path__ = [str(_backend / 'open_webui')]
    _package.__package__ = 'open_webui'
    _spec = importlib.machinery.ModuleSpec('open_webui', loader=None, is_package=True)
    _spec.submodule_search_locations = [str(_backend / 'open_webui')]
    _package.__spec__ = _spec
    sys.modules['open_webui'] = _package

if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

# The ordinary machinery can now import the subpackages, whose __init__ files
# only depend on the standard library and sibling modules.
importlib.import_module('open_webui.inference.cortex')
