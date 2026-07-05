"""Error paths should not call Ollama for simple suggestions."""

from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


@patch("colossal_cave_July_5_26.ollama_generate")
def test_drop_wrong_item_no_llm(mock_gen, game_module, minimal_game_world):
    player = minimal_game_world["player"]
    player.items.append(minimal_game_world["sword"])
    player.current_room = minimal_game_world["r1"]
    result = player.leave(["nonexistent sword"])
    assert "don't have" in str(result).lower() or "carry" in str(result).lower()
    mock_gen.assert_not_called()


@patch("colossal_cave_July_5_26.ollama_generate")
def test_unrecognized_command_deterministic(mock_gen, game_module, minimal_game_world):
    gm = game_module
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    gm.playerOne = player
    # Force through execute path with empty action
    result = gm.get_llm_error_suggestions("gibberish xyz", None, {})
    assert "don't understand" in result.lower()
    assert "look" in result.lower()
    mock_gen.assert_not_called()
