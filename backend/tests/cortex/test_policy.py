"""Tests for the CORTEX security policy.

The policy is what stops us inheriting ai-manus' permissive defaults, so these
tests assert the refusals, not just the happy path.
"""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from open_webui.inference.cortex.capabilities import Capability  # noqa: E402
from open_webui.inference.cortex.policy import CortexPolicy, CortexPolicyError  # noqa: E402


def test_allowlisted_host_is_accepted():
    policy = CortexPolicy(allowed_endpoint_hosts=frozenset({'ai-manus'}))
    assert policy.validate_endpoint('http://ai-manus:8000') == 'http://ai-manus:8000'


def test_unlisted_dns_host_is_refused():
    policy = CortexPolicy(allowed_endpoint_hosts=frozenset({'ai-manus'}))
    with pytest.raises(CortexPolicyError) as excinfo:
        policy.validate_endpoint('https://evil.example.com')
    assert excinfo.value.code == 'endpoint_not_allowed'


def test_link_local_metadata_address_is_refused():
    # 169.254.169.254 is the cloud metadata endpoint: never reachable.
    policy = CortexPolicy()
    with pytest.raises(CortexPolicyError) as excinfo:
        policy.validate_endpoint('http://169.254.169.254/latest/meta-data/')
    assert excinfo.value.code == 'endpoint_address'


def test_public_address_is_refused_even_with_private_networks_allowed():
    policy = CortexPolicy(allow_private_networks=True)
    with pytest.raises(CortexPolicyError):
        policy.validate_endpoint('http://93.184.216.34')


def test_private_address_allowed_when_configured():
    policy = CortexPolicy(allow_private_networks=True)
    assert policy.validate_endpoint('http://10.0.0.5:8080') == 'http://10.0.0.5:8080'


def test_private_address_refused_when_private_networks_disabled():
    policy = CortexPolicy(allow_private_networks=False)
    with pytest.raises(CortexPolicyError):
        policy.validate_endpoint('http://10.0.0.5:8080')


def test_unsupported_scheme_is_refused():
    policy = CortexPolicy()
    with pytest.raises(CortexPolicyError) as excinfo:
        policy.validate_endpoint('file:///etc/passwd')
    assert excinfo.value.code == 'endpoint_scheme'


def test_screen_streaming_is_refused_unless_enabled():
    policy = CortexPolicy(allow_screen_streaming=False, allowed_endpoint_hosts=frozenset({'ai-manus'}))
    with pytest.raises(CortexPolicyError) as excinfo:
        policy.validate_endpoint('ws://ai-manus:5901', capability=Capability.SCREEN)
    assert excinfo.value.code == 'screen_streaming_disabled'

    permissive = CortexPolicy(allow_screen_streaming=True, allowed_endpoint_hosts=frozenset({'ai-manus'}))
    assert permissive.validate_endpoint('ws://ai-manus:5901', capability=Capability.SCREEN) == 'ws://ai-manus:5901'


def test_denied_tool_is_refused():
    policy = CortexPolicy(denied_tools=frozenset({'shell'}))
    with pytest.raises(CortexPolicyError) as excinfo:
        policy.authorize_tool('shell')
    assert excinfo.value.code == 'tool_denied'


def test_tool_budget_is_enforced():
    policy = CortexPolicy(max_tool_calls=2)
    policy.authorize_tool('a')
    policy.authorize_tool('b')
    with pytest.raises(CortexPolicyError) as excinfo:
        policy.authorize_tool('c')
    assert excinfo.value.code == 'budget_tool_calls'
    assert policy.tool_calls_used == 2


def test_policy_is_serializable_for_events():
    policy = CortexPolicy(allowed_endpoint_hosts=frozenset({'ai-manus'}), max_tool_calls=5)
    payload = policy.to_dict()
    assert payload['allowedEndpointHosts'] == ['ai-manus']
    assert payload['maxToolCalls'] == 5
    assert payload['allowScreenStreaming'] is False
