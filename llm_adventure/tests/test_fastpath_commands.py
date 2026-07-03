"""Tests for fast-path commands and blockers."""

from engine.bible_store import set_world_bible
from engine.commands import handle_look_command, post_turn_bookkeeping, try_local_command
from engine.session import seed_game_state_from_world_bible
from engine.world_bible.win_condition import check_win_condition


def test_inventory_fastpath(state_mgr):
    state_mgr.state.inventory = ["torch"]
    result = try_local_command(state_mgr, "inventory")
    assert result is not None
    text, _debug = result
    assert "torch" in text


def test_take_and_win(default_cave_bible, state_mgr):
    set_world_bible(default_cave_bible)
    seed_game_state_from_world_bible(state_mgr)
    # Place win item and go to win location via fastpath
    wc = default_cave_bible["win_condition"]
    win_item = wc["required_items"][0]
    win_loc = wc["required_location"]
    state_mgr.state.location = win_loc
    state_mgr.state.inventory = [win_item]
    post_turn_bookkeeping(state_mgr)
    assert check_win_condition(state_mgr) is not None


def test_movement_blocked_by_monster(state_mgr):
    wb = {
        "locations": [
            {"name": "A", "description": "start", "exits": ["B"]},
            {"name": "B", "description": "danger", "exits": ["A"]},
        ],
        "monsters": [{"name": "Troll", "location": "B", "difficulty": "high", "weakness": "fire"}],
        "key_items": [],
        "solution_chain": [],
    }
    set_world_bible(wb)
    state_mgr.state.location = "A"
    state_mgr.state.known_map = {
        "A": {"exits": ["B"], "notes": ""},
        "B": {"exits": ["A"], "notes": ""},
    }
    result = try_local_command(state_mgr, "go B")
    assert result is not None
    text, _debug = result
    assert "cannot enter" in text.lower() or "blocks" in text.lower()


def test_dark_room_look(state_mgr):
    wb = {
        "locations": [{"name": "Dark Cave", "description": "pitch black tunnel", "exits": []}],
    }
    set_world_bible(wb)
    state_mgr.state.location = "Dark Cave"
    text = handle_look_command(state_mgr)
    assert "pitch dark" in text.lower()
