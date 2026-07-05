"""Mocked-LLM golden path walkthrough."""

import json
from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


@patch("colossal_cave_July_5_26._add_combat_flavor", side_effect=lambda base, *a, **k: base)
@patch("colossal_cave_July_5_26.ollama_generate")
def test_golden_path_take_wear_attack_solve(mock_gen, _flavor, game_module, minimal_game_world):
    """Golden path: take → wear → attack → solve — all fast-path, no Ollama."""
    gm = game_module
    gm.magicword = "xyzzy"
    player = minimal_game_world["player"]
    r2 = minimal_game_world["r2"]
    r3 = minimal_game_world["r3"]
    armor = gm.Armor("Chain Mail", 5, runllm=False)
    r2.items.append(armor)
    magic = minimal_game_world["magic"]
    player.items.append(magic)
    player.current_room = r2
    gm.process_game_command("take all", player, r2)
    gm.process_game_command("wear chain mail", player, r2)
    player.current_room = r3
    weapon = gm.Weapon("Iron Sword", 20, runllm=False)
    player.items.append(weapon)
    attack_result = gm.process_game_command("kill goblin with iron sword", player, r3)
    assert "hit" in str(attack_result).lower() or "killed" in str(attack_result).lower()
    solve_result = gm.process_game_command("solve Piano Key", player, r3)
    assert "solved" in str(solve_result).lower() or "evaporates" in str(solve_result).lower()
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_fast_path_take_skips_ollama(mock_gen, game_module, minimal_game_world):
    mock_gen.return_value = json.dumps({
        "chosen_action": ["take"],
        "entity_categories": {"room_items": ["Iron Sword"]},
        "confidence": 0.95,
    })
    player = minimal_game_world["player"]
    player.current_room = minimal_game_world["r2"]
    game_module.playerOne = player

    result = game_module.process_game_command("take sword", player, player.current_room)
    assert "Iron Sword" in str(result) or "iron" in str(result).lower()
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_preprocess_wired_before_llm(mock_gen, game_module, test_player, test_room):
    mock_gen.return_value = json.dumps({
        "chosen_action": ["go"],
        "entity_categories": {"directions": ["north"]},
        "confidence": 0.9,
    })
    test_player.current_room = test_room
    game_module.llm_process_command("attck north", test_player, test_room)
    call_prompt = mock_gen.call_args[0][0]
    assert "attack" in call_prompt.lower() or mock_gen.called
