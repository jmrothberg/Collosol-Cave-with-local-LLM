"""riddle status should not call Ollama on easy."""

from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


@patch("colossal_cave_July_5_26.ollama_generate")
def test_riddle_status_uses_static_hint(mock_gen, game_module, minimal_game_world):
    gm = game_module
    gm.playerOne = minimal_game_world["player"]
    gm.rooms = minimal_game_world["rooms"]
    gm.difficulty_g = "easy"
    gm.playerOne.current_room = minimal_game_world["r3"]
    result = gm.check_riddle_status()
    assert "Think music" in result or "HINT" in result
    mock_gen.assert_not_called()
