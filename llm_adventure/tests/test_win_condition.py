"""Tests for win condition parsing."""

from engine.state import StateManager
from engine.world_bible.win_condition import check_win_condition, parse_win_condition


def test_parse_struct_win(minimal_bible):
    struct = parse_win_condition(minimal_bible)
    assert struct["required_items"] == ["gem"]
    assert struct["required_location"] == "Start"


def test_check_win_location_and_items(state_mgr, minimal_bible):
    from engine.bible_store import set_world_bible

    set_world_bible(minimal_bible)
    state_mgr.state.location = "Start"
    state_mgr.state.inventory = ["gem"]
    assert check_win_condition(state_mgr) is not None


def test_check_win_not_met(state_mgr, minimal_bible):
    from engine.bible_store import set_world_bible

    set_world_bible(minimal_bible)
    state_mgr.state.location = "Mid"
    state_mgr.state.inventory = []
    assert check_win_condition(state_mgr) is None
