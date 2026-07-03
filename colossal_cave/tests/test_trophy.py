"""Tests for engine.trophy.check_trophy."""

from engine.trophy import check_trophy


def test_trophy_not_awarded_when_condition_fails(three_room_world, minimal_player):
    trophies = []
    # Room 3 unexplored → condition lambda room: not room.explored is True for room 3
    console, _, _ = check_trophy(
        "explorer",
        lambda room: not room.explored,
        "you explored everything",
        three_room_world,
        trophies,
        minimal_player,
    )
    assert "Congratulations" not in console
    assert "explorer" not in trophies


def test_trophy_awarded_when_all_explored(three_room_world, minimal_player):
    for room in three_room_world.values():
        room.explored = True
    trophies = []
    console, _, _ = check_trophy(
        "explorer",
        lambda room: not room.explored,
        "you explored everything",
        three_room_world,
        trophies,
        minimal_player,
    )
    assert "Congratulations" in console
    assert "explorer" in trophies


def test_trophy_not_duplicated(three_room_world, minimal_player):
    for room in three_room_world.values():
        room.explored = True
    trophies = ["explorer"]
    console, _, _ = check_trophy(
        "explorer",
        lambda room: not room.explored,
        "you explored everything",
        three_room_world,
        trophies,
        minimal_player,
    )
    assert console == ""
    assert trophies.count("explorer") == 1
