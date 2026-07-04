"""Tests for Player core methods."""

from unittest.mock import patch

import pytest

pytest.importorskip("torch")


def test_player_move(game_module, minimal_game_world):
    player = minimal_game_world["player"]
    result = player.move("east")
    assert player.current_room.number == 2
    assert "east" in result.lower()


def test_player_take_fuzzy(game_module, minimal_game_world):
    player = minimal_game_world["player"]
    player.current_room = minimal_game_world["r2"]
    text, _ = player.take(["iron"])
    assert "Iron Sword" in text
    assert any(i.name == "Iron Sword" for i in player.items)


def test_player_attack_kills_monster(game_module, minimal_game_world):
    player = minimal_game_world["player"]
    sword = minimal_game_world["sword"]
    player.items.append(sword)
    player.current_room = minimal_game_world["r3"]
    with patch.object(game_module, "ollama_generate", return_value="The goblin staggers."):
        result = player.attack("Goblin", "Iron Sword")
    assert "killed" in result.lower() or "hit" in result.lower()


def test_solve_puzzle_success(game_module, minimal_game_world):
    player = minimal_game_world["player"]
    magic = minimal_game_world["magic"]
    player.items.append(magic)
    player.current_room = minimal_game_world["r3"]
    minimal_game_world["r3"].riddle_magic_items = ["Piano Key"]
    game_module.magicword = "testword"
    result = player.solve_puzzle("Piano Key")
    assert "solved" in result.lower()
    assert minimal_game_world["r3"].riddle_solved is True
