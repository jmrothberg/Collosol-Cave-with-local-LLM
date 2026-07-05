"""P0 fast-path integration — assert ollama_generate not called."""

from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


def _setup_armor_player(game_module, minimal_game_world):
    gm = game_module
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    plate = gm.Armor("Plate Armor", 10, runllm=False)
    padded = gm.Armor("Padded Armor", 3, runllm=False)
    player.items.extend([plate, padded])
    player.current_room = room
    gm.playerOne = player
    gm.cheat_mode = False
    gm.magicword = "testword"
    return player, room


@patch("colossal_cave_July_5_26.ollama_generate")
def test_wear_the_armor_fast(mock_gen, game_module, minimal_game_world):
    player, room = _setup_armor_player(game_module, minimal_game_world)
    result = game_module.process_game_command("wear the armor", player, room)
    assert "put on" in str(result).lower() or "Plate" in str(result) or "Padded" in str(result)
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_wear_plate_armor_fast(mock_gen, game_module, minimal_game_world):
    player, room = _setup_armor_player(game_module, minimal_game_world)
    result = game_module.process_game_command("wear plate armor", player, room)
    assert "Plate Armor" in str(result)
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_remove_armor_fast(mock_gen, game_module, minimal_game_world):
    player, room = _setup_armor_player(game_module, minimal_game_world)
    game_module.process_game_command("wear plate armor", player, room)
    result = game_module.process_game_command("remove armor", player, room)
    assert "took off" in str(result).lower() or "Plate" in str(result)
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_use_healing_potion_fast(mock_gen, game_module, minimal_game_world):
    gm = game_module
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    potion = gm.Magic("Healing Potion", 10, runllm=False)
    player.items.append(potion)
    player.current_room = room
    result = gm.process_game_command("use healing potion", player, room)
    assert "restored" in str(result).lower() or "used" in str(result).lower()
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_solve_named_item_fast(mock_gen, game_module, minimal_game_world):
    gm = game_module
    player = minimal_game_world["player"]
    room = minimal_game_world["r3"]
    magic = minimal_game_world["magic"]
    player.items.append(magic)
    player.current_room = room
    result = gm.process_game_command("solve Piano Key", player, room)
    assert "solved" in str(result).lower() or "evaporates" in str(result).lower()
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26._add_combat_flavor", side_effect=lambda base, *a, **k: base)
@patch("colossal_cave_July_5_26.ollama_generate")
def test_kill_goblin_fast(mock_gen, _flavor, game_module, minimal_game_world):
    gm = game_module
    player = minimal_game_world["player"]
    room = minimal_game_world["r3"]
    player.current_room = room
    weapon = gm.Weapon("Iron Sword", 20, runllm=False)
    player.items.append(weapon)
    result = gm.process_game_command("kill goblin with iron sword", player, room)
    assert "hit" in str(result).lower() or "damage" in str(result).lower()
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_put_down_item_fast(mock_gen, game_module, minimal_game_world):
    gm = game_module
    player = minimal_game_world["player"]
    sword = minimal_game_world["sword"]
    player.items.append(sword)
    player.current_room = minimal_game_world["r1"]
    result = gm.process_game_command("put down iron sword", player, player.current_room)
    assert "Iron Sword" in str(result) or "dropped" in str(result).lower()
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_xyzzy_cheat_fast(mock_gen, game_module, minimal_game_world):
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    result = game_module.process_game_command("xyzzy", player, room)
    assert "cheat" in str(result).lower()
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_health_cheat_fast(mock_gen, game_module, minimal_game_world):
    gm = game_module
    gm.cheat_mode = True
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    result = gm.process_game_command("health 100", player, room)
    assert "100" in str(result)
    mock_gen.assert_not_called()
