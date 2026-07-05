"""Tests for engine.ux_helpers.adjacent_cues and format_room_exits."""

from engine.ux_helpers import adjacent_cues, format_room_exits
from tests.conftest import _StubRoom


def test_adjacent_cues_monster_growl():
    r1 = _StubRoom(1)
    r2 = _StubRoom(2)
    r1.add_connection("east", r2)
    r2.monsters.append(type("M", (), {"name": "Goblin"})())
    cue = adjacent_cues(r1, "easy")
    assert "growl" in cue.lower() or "goblin" in cue.lower()


def test_adjacent_cues_hard_mode_silent():
    r1 = _StubRoom(1)
    r2 = _StubRoom(2)
    r1.add_connection("east", r2)
    r2.monsters.append(type("M", (), {"name": "Goblin"})())
    assert adjacent_cues(r1, "hard") == ""


def test_format_room_exits_lists_directions():
    r1 = _StubRoom(1)
    r2 = _StubRoom(2)
    r3 = _StubRoom(3)
    r1.add_connection("north", r2)
    r1.add_connection("east", r3)
    text = format_room_exits(r1, "easy")
    assert "Exits from here:" in text
    assert "go east (room 3)" in text
    assert "go north (room 2)" in text
