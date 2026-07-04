"""Tests for engine.fast_path.try_local_command."""

from unittest.mock import patch

from engine.fast_path import (
    bare_solve_magic_items,
    movement_blocked_by_monster,
    normalize_direction,
    try_local_command,
)
from tests.conftest import _StubRoom


def test_normalize_direction():
    assert normalize_direction("n") == "north"
    assert normalize_direction("WEST") == "west"


def test_movement_blocked_by_monster():
    r1 = _StubRoom(1)
    r2 = _StubRoom(2)
    r1.add_connection("east", r2)
    r2.monsters.append(type("M", (), {"name": "Dragon"})())
    msg = movement_blocked_by_monster(r1, "east")
    assert msg is not None
    assert "Dragon" in msg
    assert "room 2" in msg


def test_go_east_not_blocked_by_monsters_in_dest(minimal_game_world):
    """Player can enter a room with monsters and fight them there."""
    world = minimal_game_world
    player = world["player"]
    room = world["r1"]
    dest = world["r2"]
    dest.monsters.append(type("M", (), {"name": "Dragon", "health": 50})())
    helpers = {"move": player.move}
    result = try_local_command("go east", player, room, helpers)
    assert "moved east" in result.lower()
    assert player.current_room.number == 2
    assert "Dragon" in result


def test_try_local_inventory_skips_llm(minimal_game_world):
    world = minimal_game_world
    player = world["player"]
    room = world["r1"]
    helpers = {"inventory": player.inventory_living}
    result = try_local_command("inventory", player, room, helpers)
    assert result is not None
    assert "Hero" in result or "nothing" in result.lower()


def test_try_local_go_east(minimal_game_world):
    world = minimal_game_world
    player = world["player"]
    room = world["r1"]
    helpers = {"move": player.move}
    result = try_local_command("go east", player, room, helpers)
    assert "east" in result.lower()


def test_bare_solve_match(minimal_game_world):
    world = minimal_game_world
    player = world["player"]
    player.items.append(world["magic"])
    world["r3"].riddle_magic_items = ["Piano Key"]
    match = bare_solve_magic_items(world["r3"], player.items)
    assert match == ("Piano Key", None)
