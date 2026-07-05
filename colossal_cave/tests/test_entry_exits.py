"""Tests for exit list ordering on room entry."""

from engine.ux_helpers import format_room_exits
from tests.conftest import _StubRoom


def test_entry_exits_prepended_before_movement_text():
    """Simulate room-entry console assembly: exits first, then move result."""
    r10 = _StubRoom(10)
    r11 = _StubRoom(11)
    r10.add_connection("north", r11)
    r11.add_connection("south", r10)

    entry_exits = format_room_exits(r11, "easy")
    move_result = "Jack moved north.\n"
    console = entry_exits + move_result

    assert console.startswith("Exits from here:")
    assert console.index("Exits from here:") < console.index("Jack moved")
