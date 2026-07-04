"""Tests for auto-surface riddle eligibility."""

from engine.ux_helpers import riddle_eligibility_text
from tests.conftest import _StubRoom


def test_riddle_eligibility_shows_solve_command():
    room = _StubRoom(1)
    room.riddles = ["A riddle"]
    room.riddle_magic_items = ["Piano Key"]
    item = type("Magic", (), {"name": "Piano Key"})()
    text = riddle_eligibility_text(room, [item])
    assert "solve" in text.lower()
    assert "Piano Key" in text
