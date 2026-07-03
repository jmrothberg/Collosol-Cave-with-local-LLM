"""Tests for seeding game state from world bible."""

from engine.bible_store import set_world_bible
from engine.session import seed_game_state_from_world_bible


def test_seed_places_items(default_cave_bible, state_mgr):
    set_world_bible(default_cave_bible)
    debug = seed_game_state_from_world_bible(state_mgr)
    assert any("seed" in d for d in debug)
    assert state_mgr.state.location
    assert len(state_mgr.state.known_map) >= 5
    total_room_items = sum(len(v) for v in state_mgr.state.room_items.values())
    assert total_room_items >= 1
