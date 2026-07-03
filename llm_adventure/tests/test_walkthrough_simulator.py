"""Walkthrough simulator — prove default cave is winnable via state manipulation."""

from engine.bible_store import set_world_bible
from engine.session import seed_game_state_from_world_bible
from engine.world_bible.win_condition import check_win_condition, parse_win_condition


def test_simulated_walkthrough_wins(default_cave_bible, state_mgr):
    set_world_bible(default_cave_bible)
    seed_game_state_from_world_bible(state_mgr)
    struct = parse_win_condition(default_cave_bible)
    win_items = struct["required_items"]
    win_loc = struct["required_location"]
    for item in win_items:
        if item not in state_mgr.state.inventory:
            state_mgr.add_item(item)
    if win_loc:
        state_mgr.move_to(win_loc)
    msg = check_win_condition(state_mgr)
    assert msg is not None
    assert "VICTORY" in msg
