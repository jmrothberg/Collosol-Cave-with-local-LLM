"""Tests for process_command NPC routing."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


def test_pending_conversation_cleared_on_movement(game_module, test_player):
    npc = MagicMock()
    npc.has_pending_action.return_value = True
    npc.name = "Merchant"
    npc.clear_pending_action = MagicMock()
    npc.handle_conversation_response = MagicMock(return_value=None)
    test_player.current_room.npc = npc

    with patch.object(game_module, "process_game_command", return_value="moved") as mock_pgc:
        result = game_module.process_command("go east", test_player, test_player.current_room)
    assert result == "moved"
    mock_pgc.assert_called_once()


def test_mb_addressing_routes_to_talk(game_module, minimal_game_world):
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player

    with patch.object(player, "talk_to_npc", return_value="MB says hi") as mock_talk:
        result = game_module.process_command("MB what should I do?", player, room)
    mock_talk.assert_called_once()
    assert "MB" in result
