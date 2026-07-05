"""Tests for process_command NPC routing."""

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")

_UNPARSED_LLM = {
    "chosen_action": [],
    "entities": {
        "location": [], "room_items": [], "monsters": [], "npcs": [], "npc_items": [],
        "my_items": [], "my_magic": [], "weapons": [], "my_armor": [], "worn_armor": [],
        "directions": [], "values": [],
    },
    "confidence": 0.3,
}


def _mock_unparsed_game_path(game_module):
    """Fast-path miss + game LLM could not pick an action."""
    return (
        patch.object(game_module, "try_local_command", return_value=None),
        patch.object(game_module, "llm_process_command", return_value=_UNPARSED_LLM),
    )


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

    fast_mock, llm_mock = _mock_unparsed_game_path(game_module)
    with fast_mock, llm_mock, patch.object(player, "talk_to_npc", return_value="MB says hi") as mock_talk:
        result = game_module.process_command("MB what should I do?", player, room)
    mock_talk.assert_called_once_with("MB what should I do?")
    assert "MB" in result


@pytest.mark.parametrize(
    "command",
    [
        "MB help",
        "MB help me figure out this game",
        "Master Beast, guide me",
        "beast hello",
        "help me figure out this game",
        "what should I do next",
    ],
)
def test_unparsed_command_falls_back_to_npc_talk(game_module, minimal_game_world, command):
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player

    fast_mock, llm_mock = _mock_unparsed_game_path(game_module)
    with fast_mock, llm_mock, patch.object(player, "talk_to_npc", return_value="MB says: Sure, I can help.") as mock_talk:
        result = game_module.process_command(command, player, room)
    mock_talk.assert_called_once_with(command)
    assert "MB" in result


def test_unparsed_command_without_npc_stays_error(game_module, minimal_game_world):
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    room.npc = None
    player.current_room = room
    game_module.playerOne = player

    fast_mock, llm_mock = _mock_unparsed_game_path(game_module)
    with fast_mock, llm_mock, patch.object(player, "talk_to_npc") as mock_talk:
        result = game_module.process_command("help me figure out this game", player, room)
    mock_talk.assert_not_called()
    assert "don't understand" in result.lower()


def test_attack_beast_stays_game_command(game_module, minimal_game_world):
    """Combat commands should not fall back to NPC chat."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player

    with patch.object(game_module, "try_local_command", return_value="You swing wildly.") as mock_fast:
        with patch.object(player, "talk_to_npc") as mock_talk:
            result = game_module.process_command("attack beast", player, room)
    mock_fast.assert_called_once()
    mock_talk.assert_not_called()
    assert "swing" in result
