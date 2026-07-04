"""Tests for execute_action live path."""

from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


def test_remove_armor_uses_worn_armor_key(game_module, test_player):
    """Regression: LLM fills worn_armor, not wear_armor."""
    game_module.playerOne = test_player
    armor = game_module.Armor("Chain Mail", 10, runllm=False)
    test_player.items.append(armor)
    test_player.wear_armor(armor)

    entities = {k: [] for k in [
        "location", "room_items", "monsters", "npcs", "npc_items",
        "my_items", "my_magic", "weapons", "my_armor", "worn_armor", "directions", "values",
    ]}
    entities["worn_armor"] = ["Chain Mail"]

    result = game_module.execute_action(["remove"], entities, "remove chain", confidence=1.0)
    assert "took off" in result.lower() or "Chain Mail" in result


def test_inventory_action_no_ollama(game_module, test_player):
    game_module.playerOne = test_player
    entities = {k: [] for k in [
        "location", "room_items", "monsters", "npcs", "npc_items",
        "my_items", "my_magic", "weapons", "my_armor", "worn_armor", "directions", "values",
    ]}
    with patch.object(game_module, "ollama_generate") as mock_gen:
        result = game_module.execute_action(["inventory"], entities, "inventory", confidence=1.0)
        mock_gen.assert_not_called()
    assert result is not None


def test_map_action(game_module, test_player):
    game_module.playerOne = test_player
    entities = {k: [] for k in [
        "location", "room_items", "monsters", "npcs", "npc_items",
        "my_items", "my_magic", "weapons", "my_armor", "worn_armor", "directions", "values",
    ]}
    result = game_module.execute_action(["map"], entities, "map", confidence=1.0)
    assert "Room" in result
