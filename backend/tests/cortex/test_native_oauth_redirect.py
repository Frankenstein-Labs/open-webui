"""Tests for the native OAuth deep-link allowlist.

The Android client cannot read the session cookie the OAuth callback would
otherwise set, so it asks the server to hand the token back over a `cortex://`
deep link. That hand-off is a token-redirection primitive, so the validator
that gates it is worth pinning down.
"""

from __future__ import annotations

import pytest

# `conftest.py` installs open_webui as a filesystem-backed package, so this
# import resolves without executing the CLI-heavy `open_webui/__init__.py`.
from open_webui.utils.native_redirect import is_allowed_native_redirect as allowed


def test_accepts_the_registered_deep_link():
    assert allowed('cortex://oauth/callback') is True


def test_accepts_a_deep_link_with_a_query():
    assert allowed('cortex://oauth/callback?token=x') is True


def test_rejects_a_web_origin():
    """A web redirect would leak the token to whoever controls that origin."""
    assert allowed('https://attacker.example/callback') is False
    assert allowed('http://attacker.example/callback') is False


def test_rejects_an_unregistered_scheme():
    assert allowed('other://oauth/callback') is False
    assert allowed('file:///etc/passwd') is False


def test_rejects_schemes_that_happen_to_start_with_the_allowed_one():
    """`cortexevil:` must not pass a naive prefix check."""
    assert allowed('cortexevil://oauth/callback') is False


def test_rejects_input_without_a_destination():
    assert allowed('cortex:') is False
    assert allowed('cortex://') is False


def test_rejects_garbage():
    assert allowed('') is False
    assert allowed('not a url') is False


def test_extra_schemes_come_from_the_environment(monkeypatch):
    monkeypatch.setenv('CORTEX_NATIVE_REDIRECT_SCHEMES', 'cortex, cortexdev')
    assert allowed('cortexdev://oauth/callback') is True
    assert allowed('other://oauth/callback') is False


def test_environment_can_narrow_the_allowlist(monkeypatch):
    monkeypatch.setenv('CORTEX_NATIVE_REDIRECT_SCHEMES', 'cortexdev')
    assert allowed('cortex://oauth/callback') is False
