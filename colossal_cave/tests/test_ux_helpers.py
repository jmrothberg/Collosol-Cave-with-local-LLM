"""Additional tests for engine.ux_helpers."""

from types import SimpleNamespace

from engine.ux_helpers import build_status_header, format_session_postmortem, reset_adventure_log
from tests.conftest import _StubRoom


def test_build_status_header_easy():
    player = SimpleNamespace(health=200, wealth=50, items=[])
    room = _StubRoom(3)
    room.description = "Glass Hall"
    header = build_status_header(player, room, "*growl*", "easy")
    assert "Room 3" in header
    assert "200" in header


def test_format_session_postmortem():
    player = SimpleNamespace(health=0, wealth=100, kills=2)
    rooms = {1: _StubRoom(1, explored=True), 2: _StubRoom(2, explored=False)}
    text = format_session_postmortem(player, rooms, ["explorer"], 10, "VICTORY")
    assert "VICTORY" in text
    assert "Turns played: 10" in text


def test_reset_adventure_log_clears():
    from engine.ux_helpers import append_adventure_log, get_adventure_log_text
    reset_adventure_log()
    append_adventure_log("look", "You see a cave.")
    assert "look" in get_adventure_log_text()
    reset_adventure_log()
    assert "begins" in get_adventure_log_text()
