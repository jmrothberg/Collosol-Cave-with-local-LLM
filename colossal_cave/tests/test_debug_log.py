"""Tests for engine.debug_log persistent play logging."""

import os

from engine.debug_log import (
    consume_command_meta,
    get_debug_log_path,
    log_play_turn,
    set_command_meta,
    start_debug_session,
)


def test_debug_log_writes_turn_context(tmp_path, minimal_game_world):
    world = minimal_game_world
    player = world["player"]
    room = world["r1"]
    game_dir = str(tmp_path / "games")
    os.makedirs(game_dir, exist_ok=True)

    path = start_debug_session(game_dir, "test_session")
    assert path.endswith("_play_debug.log")
    assert get_debug_log_path() == path

    set_command_meta(path="fast_path", preprocessed="go east", action="move")
    log_play_turn(
        command="go east",
        preprocessed="go east",
        path="fast_path",
        room_before=room,
        room_after=world["r2"],
        player=player,
        result="Jack moved east.\n",
    )
    meta = consume_command_meta()
    assert meta["path"] == "fast_path"

    with open(path, encoding="utf-8") as f:
        text = f.read()
    assert "SESSION NEW" in text
    assert "command_raw='go east'" in text
    assert "route=fast_path" in text
    assert "exit east" in text
    assert "room_before" in text
    assert "room_after" in text
