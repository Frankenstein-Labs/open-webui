import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / 'open_webui' / 'inference' / 'engine_router.py'
spec = importlib.util.spec_from_file_location('engine_router_under_test', MODULE_PATH)
engine_router = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = engine_router
assert spec.loader is not None
spec.loader.exec_module(engine_router)


def test_computer_tasks_use_ai_manus():
    decision = engine_router.choose_engine(
        {'messages': [{'role': 'user', 'content': 'open a browser and click the website'}]}
    )
    assert decision.name == 'ai-manus'


def test_software_tasks_use_openhands():
    decision = engine_router.choose_engine(
        {'messages': [{'role': 'user', 'content': 'refactor this repository and create a commit'}]}
    )
    assert decision.name == 'openhands'


def test_explicit_engine_wins():
    decision = engine_router.choose_engine(
        {'metadata': {'engine': 'ai-manus'}, 'messages': [{'role': 'user', 'content': 'refactor code'}]}
    )
    assert decision.name == 'ai-manus'


def test_explicit_engine_wins_over_mode():
    decision = engine_router.choose_engine(
        {
            'metadata': {'engine': 'ai-manus'},
            'conversation_mode': 'discussion',
            'messages': [{'role': 'user', 'content': 'refactor code'}],
        }
    )
    assert decision.name == 'ai-manus'


def test_discussion_mode_stays_conversational():
    decision = engine_router.choose_engine(
        {
            'conversation_mode': 'discussion',
            'messages': [{'role': 'user', 'content': 'open a browser and click the website'}],
        }
    )
    assert decision.name == 'openhands'
    assert 'discussion' in decision.reason


def test_agent_mode_routes_software_tasks_to_openhands():
    decision = engine_router.choose_engine(
        {
            'conversation_mode': 'agent',
            'messages': [{'role': 'user', 'content': 'refactor this repository and create a commit'}],
        }
    )
    assert decision.name == 'openhands'
    assert 'agent mode' in decision.reason


def test_agent_mode_routes_computer_tasks_to_ai_manus():
    decision = engine_router.choose_engine(
        {
            'conversation_mode': 'agent',
            'messages': [{'role': 'user', 'content': 'open a browser and click the website'}],
        }
    )
    assert decision.name == 'ai-manus'
    assert 'agent mode' in decision.reason


def test_agent_mode_reads_mode_from_metadata():
    decision = engine_router.choose_engine(
        {
            'metadata': {'conversation_mode': 'agent'},
            'messages': [{'role': 'user', 'content': 'refactor this repository'}],
        }
    )
    assert decision.name == 'openhands'


def test_agent_mode_general_task_uses_configured_default():
    decision = engine_router.choose_engine(
        {
            'conversation_mode': 'agent',
            'messages': [{'role': 'user', 'content': 'summarise the quarterly report'}],
        }
    )
    assert decision.name in {'openhands', 'ai-manus'}
    assert 'agent mode' in decision.reason


def test_absent_mode_keeps_intent_based_routing():
    decision = engine_router.choose_engine(
        {'messages': [{'role': 'user', 'content': 'open a browser and click the website'}]}
    )
    assert decision.name == 'ai-manus'
    assert 'agent mode' not in decision.reason


if __name__ == '__main__':
    test_computer_tasks_use_ai_manus()
    test_software_tasks_use_openhands()
    test_explicit_engine_wins()
    test_explicit_engine_wins_over_mode()
    test_discussion_mode_stays_conversational()
    test_agent_mode_routes_software_tasks_to_openhands()
    test_agent_mode_routes_computer_tasks_to_ai_manus()
    test_agent_mode_reads_mode_from_metadata()
    test_agent_mode_general_task_uses_configured_default()
    test_absent_mode_keeps_intent_based_routing()
    print('engine routing: OK')
