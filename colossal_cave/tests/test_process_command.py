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


def test_ask_mb_skips_game_json_parser(game_module, minimal_game_world):
    """MB-addressed chat must not go through game-action JSON (entity_categories: null crash)."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]

    with patch.object(game_module, "try_local_command", return_value=None):
        with patch.object(game_module, "llm_process_command") as mock_llm:
            with patch.object(game_module, "_talk_to_master_beast", return_value="MB says hi") as mock_mb:
                result = game_module.process_command("ask MB for help", player, room)
    mock_llm.assert_not_called()
    mock_mb.assert_called_once_with(player, "ask MB for help")
    assert "MB" in result


@pytest.mark.parametrize(
    "command",
    [
        "MB help",
        "MB help me figure out this game",
        "Master Beast, guide me",
        "beast hello",
    ],
)
def test_mb_addressed_routes_to_talk(game_module, minimal_game_world, command):
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]

    with patch.object(game_module, "try_local_command", return_value=None):
        with patch.object(game_module, "llm_process_command") as mock_llm:
            with patch.object(game_module, "_talk_to_master_beast", return_value="MB says: Sure.") as mock_mb:
                result = game_module.process_command(command, player, room)
    mock_llm.assert_not_called()
    mock_mb.assert_called_once_with(player, command)
    assert "MB" in result


@pytest.mark.parametrize(
    "command",
    [
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
    game_module.rooms = minimal_game_world["rooms"]

    fast_mock, llm_mock = _mock_unparsed_game_path(game_module)
    with fast_mock, llm_mock, patch.object(game_module, "_talk_to_master_beast", return_value="MB says: Sure, I can help.") as mock_mb:
        result = game_module.process_command(command, player, room)
    mock_mb.assert_called_once_with(player, command)
    assert "MB" in result


def test_help_with_riddle_reaches_mb_from_other_room(game_module, minimal_game_world):
    """MB in room 1 answers help/riddle questions even when player is elsewhere."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r2"]
    room.npc = None
    player.current_room = room
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    minimal_game_world["r1"].npc = mb
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]

    with patch.object(game_module, "try_local_command", return_value=None):
        with patch.object(game_module, "llm_process_command") as mock_llm:
            with patch.object(game_module, "_talk_to_master_beast", return_value="MB says: riddle hint") as mock_mb:
                result = game_module.process_command("help me with the riddle", player, room)
    mock_llm.assert_not_called()
    mock_mb.assert_called_once_with(player, "help me with the riddle")
    assert "MB" in result


def test_unparsed_command_without_npc_stays_error(game_module, minimal_game_world):
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    room.npc = None
    player.current_room = room
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]
    minimal_game_world["r1"].npc = None

    fast_mock, llm_mock = _mock_unparsed_game_path(game_module)
    with fast_mock, llm_mock, patch.object(game_module, "_get_master_beast_npc", return_value=None):
        result = game_module.process_command("help me figure out this game", player, room)
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


def test_mb_talk_survives_dialogue_error(game_module, minimal_game_world):
    """MB dialogue errors must fall back to offline guidance, not a hard crash."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player

    with patch.object(game_module, "try_local_command", return_value=None):
        with patch.object(
            mb,
            "_llM_generate_self_dialogue_inner",
            side_effect=RuntimeError("boom"),
        ):
            result = game_module.process_command("ask MB for help", player, room)
    assert "MB" in result
    assert "don't understand" not in result.lower()


def test_llm_null_entity_categories_does_not_crash(game_module, minimal_game_world):
    """Regression: entity_categories:null must not crash execute_action path."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    room.npc = None
    player.current_room = room
    game_module.playerOne = player

    null_entities_llm = {
        "chosen_action": ["inventory"],
        "entities": {
            "location": [], "room_items": [], "monsters": [], "npcs": [], "npc_items": [],
            "my_items": [], "my_magic": [], "weapons": [], "my_armor": [], "worn_armor": [],
            "directions": [], "values": [],
        },
        "confidence": 0.95,
    }
    with patch.object(game_module, "try_local_command", return_value=None):
        with patch.object(game_module, "llm_process_command", return_value=null_entities_llm):
            result = game_module.process_game_command("inventory", player, room)
    assert isinstance(result, str)
    assert "nothing" in result.lower() or "carrying" in result.lower() or "items" in result.lower()


def test_gather_all_items_uses_fast_path_not_mb(game_module, minimal_game_world):
    """'gather all items' is a take command, not MB chat."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r2"]
    sword = minimal_game_world["sword"]
    room.items.append(sword)
    player.current_room = room
    game_module.playerOne = player

    with patch.object(player, "talk_to_npc") as mock_talk:
        result = game_module.process_command("gather all the items in the room", player, room)
    mock_talk.assert_not_called()
    console = result[0] if isinstance(result, tuple) else result
    assert "Taking all items" in console or "Iron Sword" in console


def test_yes_routes_to_mb_pending_when_remote(game_module, minimal_game_world):
    """Bare 'yes' must reach MB pending in room 1 when player is in another room."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r2"]
    room.npc = None
    player.current_room = room
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    minimal_game_world["r1"].npc = mb
    mb.set_pending_action(
        "information",
        "explain: which direction to go",
        {"info_details": "which direction", "offer_snippet": "Would you like me to tell you which direction to go?"},
    )
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]

    with patch.object(mb, "handle_conversation_response", return_value="MB says: Go east.\n") as mock_handle:
        with patch.object(game_module, "process_game_command") as mock_pgc:
            result = game_module.process_command("yes", player, room)
    mock_handle.assert_called_once_with("yes")
    mock_pgc.assert_not_called()
    assert "MB" in result


def test_information_accept_fulfills_direction(game_module, minimal_game_world):
    """Accepting INFORMATION offer must fulfill via MB dialogue, not detector metadata."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]
    mb.set_pending_action(
        "information",
        "Would you like me to tell you which direction to go?",
        {
            "offer_snippet": "Would you like me to tell you which direction to go?",
            "original_dialogue": "ask mb how to win",
        },
    )

    with patch.object(
        mb,
        "llM_generate_self_dialogue",
        return_value="MB says: Head **go east** to room 10 for your first treasure.\n",
    ) as mock_dialogue:
        result = mb.handle_conversation_response("yes")
    mock_dialogue.assert_called_once()
    fulfill_arg = mock_dialogue.call_args[0][0]
    assert "accepted your offer" in fulfill_arg.lower()
    assert "which direction" in fulfill_arg.lower()
    assert "What would you like to know?" not in result
    assert "go east" in result.lower()
    assert not mb.has_pending_action()


def test_information_accept_ignores_meta_details(game_module, minimal_game_world):
    """Legacy pending with detector meta details must still fulfill from offer_snippet."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]
    meta = "The NPC is asking if the player wants to know the specific direction to find the first treasure."
    mb.set_pending_action(
        "information",
        "Would you like me to tell you exactly which direction to head first to find your first treasure?",
        {
            "info_details": meta,
            "offer_snippet": "Would you like me to tell you exactly which direction to head first to find your first treasure?",
            "original_dialogue": "ask mb how to win",
        },
    )

    with patch.object(
        mb,
        "llM_generate_self_dialogue",
        return_value="MB says: Start with **go east** to room 10.\n",
    ):
        result = mb.handle_conversation_response("yes!")
    assert "npc is asking" not in result.lower()
    assert "go east" in result.lower()
    assert not mb.has_pending_action()


def test_quest_accept_with_meta_details_fulfills_for_mb(game_module, minimal_game_world):
    """QUEST misclassified with meta details should fulfill like any accepted offer."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]
    meta = "The NPC is asking if the player wants specific information regarding which direction to head."
    mb.set_pending_action(
        "quest",
        "Would you like me to tell you which direction to go?",
        {
            "quest_details": meta,
            "info_details": meta,
            "offer_snippet": "Would you like me to tell you which direction to go?",
            "original_dialogue": "ask mb how to win",
        },
    )

    with patch.object(
        mb,
        "llM_generate_self_dialogue",
        return_value="MB says: Try **go south** to room 13.\n",
    ):
        result = mb.handle_conversation_response("yes")
    assert "quest accepted" not in result.lower()
    assert "npc is asking" not in result.lower()
    assert "go south" in result.lower()
    assert not mb.has_pending_action()


def test_mb_dialogue_survives_null_monster_items(game_module, minimal_game_world, monkeypatch):
    """Legacy saves with monster.items=None must not crash MB knowledge gathering."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player
    # Simulate corrupted save data
    for r in minimal_game_world["rooms"].values():
        r.items = None
        for m in getattr(r, "monsters", None) or []:
            m.items = None

    monkeypatch.setattr(game_module, "ollama_generate", lambda *a, **k: "Win by solving all riddles and collecting trophies.")
    with patch.object(game_module, "try_local_command", return_value=None):
        result = game_module.process_command("ask MB how to win", player, room)
    assert "MB" in result
    assert "don't understand" not in result.lower()
