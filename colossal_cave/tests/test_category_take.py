"""Category take/drop — take all weapons should not take everything."""

from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


@pytest.mark.parametrize("category_take", ["take all weapons", "take all"])
def test_category_take_commands(category_take, game_module, minimal_game_world):
    """Category take and take all go through fast-path when magicword is set."""
    from unittest.mock import patch
    gm = game_module
    gm.magicword = "xyzzy"
    player = minimal_game_world["player"]
    room = minimal_game_world["r2"]
    armor = gm.Armor("Chain Mail", 5, runllm=False)
    room.items.append(armor)
    player.current_room = room
    with patch.object(gm, "ollama_generate") as mock_gen:
        gm.process_game_command(category_take, player, room)
        mock_gen.assert_not_called()
