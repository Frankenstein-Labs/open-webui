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


if __name__ == '__main__':
    test_computer_tasks_use_ai_manus()
    test_software_tasks_use_openhands()
    test_explicit_engine_wins()
    print('engine routing: OK')
