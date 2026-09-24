"""Tests for the secure runtime descriptors.

These assert that the reused port-preview path never receives an endpoint the
policy refused, and that screen streaming is off by default.
"""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from open_webui.inference.cortex.capabilities import Capability  # noqa: E402
from open_webui.inference.cortex.policy import CortexPolicy  # noqa: E402
from open_webui.inference.cortex.runtime import describe_runtime  # noqa: E402


def test_runtime_describes_cdp_and_api_but_blocks_screen_by_default():
    policy = CortexPolicy(allowed_endpoint_hosts=frozenset({'sandbox'}), allow_screen_streaming=False)
    descriptor = describe_runtime('ai-manus', 'session-1', 'sandbox', policy=policy)

    names = {endpoint.name for endpoint in descriptor.endpoints}
    assert names == {'api', 'cdp'}
    assert descriptor.blocked == {'vnc': 'screen_streaming_disabled'}
    assert descriptor.endpoint('cdp').port == 9222
    assert descriptor.endpoint('cdp').capability is Capability.BROWSER


def test_screen_endpoint_is_exposed_when_policy_allows_it():
    policy = CortexPolicy(allowed_endpoint_hosts=frozenset({'sandbox'}), allow_screen_streaming=True)
    descriptor = describe_runtime('ai-manus', 'session-1', 'sandbox', policy=policy)

    names = {endpoint.name for endpoint in descriptor.endpoints}
    assert names == {'api', 'cdp', 'vnc'}
    vnc = descriptor.endpoint('vnc')
    assert vnc.scheme == 'ws'
    assert vnc.port == 5901
    assert descriptor.blocked == {}


def test_unlisted_host_blocks_every_endpoint():
    policy = CortexPolicy(allowed_endpoint_hosts=frozenset(), allow_screen_streaming=True)
    descriptor = describe_runtime('ai-manus', 'session-1', 'random-host', policy=policy)
    assert descriptor.endpoints == []
    assert set(descriptor.blocked) == {'api', 'cdp', 'vnc'}
    assert set(descriptor.blocked.values()) == {'endpoint_not_allowed'}


def test_descriptor_is_serializable_for_the_chat_ui():
    policy = CortexPolicy(allowed_endpoint_hosts=frozenset({'sandbox'}))
    payload = describe_runtime('ai-manus', 'session-1', 'sandbox', policy=policy).to_dict()
    assert payload['engine'] == 'ai-manus'
    assert payload['sessionId'] == 'session-1'
    assert {entry['name'] for entry in payload['endpoints']} == {'api', 'cdp'}
    assert payload['blocked']['vnc'] == 'screen_streaming_disabled'
