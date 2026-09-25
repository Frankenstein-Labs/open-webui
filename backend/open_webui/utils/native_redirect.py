"""Allowlist for native OAuth deep-link hand-off.

The Android client cannot read the session cookie an OAuth callback normally
sets, so it asks the server to deliver the token over a `cortex://` deep link
(see `handle_login` / `handle_callback` in `utils.oauth`). That hand-off is a
token-redirection primitive, so the destination is validated here rather than
in the OAuth flow, where importing the web stack would be unavoidable.
"""

from __future__ import annotations

import os
import urllib.parse

#: Schemes the shipped app registers. `cortex` matches the `<data
#: android:scheme>` in AndroidManifest.xml; operators of other native builds can
#: extend the list without a code change.
DEFAULT_NATIVE_REDIRECT_SCHEMES = 'cortex'


def is_allowed_native_redirect(uri: str) -> bool:
    """Whether `uri` may receive an OAuth token from the native app.

    Only schemes the app registers are accepted, and the URI must carry an
    actual destination, so a bare `cortex:` cannot be used.
    """
    schemes = {
        scheme.strip().lower()
        for scheme in os.getenv(
            'CORTEX_NATIVE_REDIRECT_SCHEMES', DEFAULT_NATIVE_REDIRECT_SCHEMES
        ).split(',')
        if scheme.strip()
    }
    try:
        parsed = urllib.parse.urlparse(uri)
    except ValueError:
        return False
    return parsed.scheme.lower() in schemes and bool(parsed.netloc or parsed.path)
