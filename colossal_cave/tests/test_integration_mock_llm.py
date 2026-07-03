"""Integration tests with mocked Ollama — imports game module after UI guard."""

import importlib
import json
import os
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")

_COLOSSAL_CAVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLOSSAL_CAVE_DIR not in sys.path:
    sys.path.insert(0, _COLOSSAL_CAVE_DIR)

_MINIMAL_ROOM = {
    "description": "A test room.",
    "riddles": [],
    "hints": [],
    "magic_items": [],
    "action_words": [],
}


@pytest.fixture(scope="module")
def game_module():
    """Import main game module and load adventure data (weapons, treasures, etc.)."""
    mod_name = "Colossal_Cave_Aug_2_25"
    if mod_name not in sys.modules:
        importlib.import_module(mod_name)
    mod = sys.modules[mod_name]
    mod.load_adventure_data()
    return mod


@pytest.fixture
def test_room(game_module):
    return game_module.Room(_MINIMAL_ROOM.copy(), 99)


@pytest.fixture
def test_player(game_module, test_room):
    """Player with equip_self skipped so tests stay deterministic."""
    with patch.object(game_module.Player, "equip_self"):
        player = game_module.Player(name="Hero", current_room=test_room)
    return player


@pytest.fixture
def mock_ollama_take_sword():
    return json.dumps({
        "chosen_action": ["take"],
        "entity_categories": {"room_items": ["Iron Sword"]},
        "confidence": 0.95,
    })


def test_preprocess_imported_from_engine(game_module):
    assert game_module.preprocess_command("attck") == "attack"


def test_find_item_smart_delegates_to_engine(game_module):
    weapon = game_module.Weapon("Iron Sword", 20, runllm=False)
    player = game_module.Player.__new__(game_module.Player)
    found = player._find_item_smart("iron", [weapon])
    assert found.name == "Iron Sword"


@patch("Colossal_Cave_Aug_2_25.ollama_generate")
def test_llm_process_command_take(mock_gen, game_module, test_room, test_player, mock_ollama_take_sword):
    mock_gen.return_value = mock_ollama_take_sword
    sword = game_module.Weapon("Iron Sword", 15, runllm=False)
    test_room.items.append(sword)
    test_player.current_room = test_room

    result = game_module.llm_process_command("take sword", test_player, test_room)
    assert "take" in result["chosen_action"]
    assert result["confidence"] >= 0.9
    mock_gen.assert_called_once()


@patch("Colossal_Cave_Aug_2_25.ollama_generate")
def test_execute_action_deterministic_inventory(mock_gen, game_module, test_player):
    """inventory action should not call Ollama."""
    game_module.playerOne = test_player

    entities = {k: [] for k in [
        "location", "room_items", "monsters", "npcs", "npc_items",
        "my_items", "my_magic", "weapons", "my_armor", "worn_armor", "directions", "values",
    ]}
    result = game_module.execute_action(["inventory"], entities, "inventory", confidence=1.0)
    mock_gen.assert_not_called()
    assert result is not None


def test_session_sync_roundtrip(game_module):
    from engine.session import GameSession, sync_globals_from_session, sync_session_from_globals

    session = sync_session_from_globals(
        rooms=game_module.rooms,
        player_one=game_module.playerOne,
        old_room=getattr(game_module, "old_room", None),
        cheat_mode=getattr(game_module, "cheat_mode", False),
        magicword=getattr(game_module, "magicword", ""),
        trophies=getattr(game_module, "trophies", []),
        active_game=getattr(game_module, "active_game", False),
        art_style_g=getattr(game_module, "art_style_g", ""),
    )
    assert isinstance(session, GameSession)
    back = sync_globals_from_session(session)
    assert back[0] is game_module.rooms
