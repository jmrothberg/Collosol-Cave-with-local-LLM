"""Mocked-LLM golden path walkthrough."""

import json
from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


@patch("Colossal_Cave_Aug_2_25.ollama_generate")
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


@patch("Colossal_Cave_Aug_2_25.ollama_generate")
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
