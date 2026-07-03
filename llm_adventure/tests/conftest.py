"""Shared pytest fixtures for llm_adventure tests."""

import json
import os
import sys

import pytest

_LLM_ADVENTURE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _LLM_ADVENTURE_DIR not in sys.path:
    sys.path.insert(0, _LLM_ADVENTURE_DIR)

from engine.bible_store import set_world_bible
from engine.state import GameState, StateManager


_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def state_mgr():
    return StateManager(GameState(player_name="Tester", location="Start"))


@pytest.fixture
def minimal_bible():
    """Three-room bible with one unreachable gap potential."""
    return {
        "objectives": ["A", "B", "C", "D"],
        "locations": [
            {"name": "Start", "description": "start", "exits": ["Mid"]},
            {"name": "Mid", "description": "middle", "exits": ["Start", "End"]},
            {"name": "End", "description": "end", "exits": ["Mid"]},
            {"name": "Side", "description": "isolated", "exits": []},
            {"name": "Vault", "description": "vault", "exits": ["Mid"]},
        ],
        "key_items": [
            {"name": "key", "purpose": "opens"},
            {"name": "gem", "purpose": "treasure"},
            {"name": "rope", "purpose": "climb"},
            {"name": "map", "purpose": "navigate"},
        ],
        "win_condition": {
            "required_items": ["gem"],
            "required_location": "Start",
            "description": "Return with the gem.",
        },
        "item_locations": {
            "key": "Mid: on table",
            "gem": "Vault: pedestal",
            "rope": "Start: floor",
            "map": "Mid: wall",
        },
        "solution_chain": [
            {"step": 1, "location": "Mid", "requires_item": None, "blocker": "find key", "result": "Pick up key"},
            {"step": 2, "location": "Vault", "requires_item": "key", "blocker": "locked door", "result": "get gem"},
            {"step": 3, "location": "Start", "requires_item": "gem", "blocker": "return", "result": "win"},
        ],
        "progression_hints": ["hint one", "hint two"],
        "npcs": [],
        "monsters": [],
        "riddles": [],
    }


@pytest.fixture
def default_cave_bible():
    path = os.path.normpath(
        os.path.join(_LLM_ADVENTURE_DIR, "..", "browser_adventure", "default_cave.json")
    )
    if not os.path.isfile(path):
        pytest.skip("default_cave.json not found")
    with open(path, "r", encoding="utf-8") as f:
        wb = json.load(f)
    from engine.world_bible.normalize import normalize_browser_chain
    from engine.world_bible.repair import auto_repair_world_bible
    from engine.world_bible.win_condition import parse_win_condition

    if "puzzle_chain" in wb and "solution_chain" not in wb:
        wb["solution_chain"] = wb.pop("puzzle_chain")
    normalize_browser_chain(wb)
    parse_win_condition(wb)
    auto_repair_world_bible(wb)
    return wb


@pytest.fixture(autouse=True)
def _clear_world_bible():
    set_world_bible(None)
    yield
    set_world_bible(None)
