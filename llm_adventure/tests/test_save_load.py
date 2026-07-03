"""Save/load round-trip tests."""

import os

from engine.bible_store import get_world_bible, set_world_bible
from engine.state import GameState, StateManager


def test_save_load_roundtrip(state_mgr, minimal_bible, tmp_path, monkeypatch):
    from config import paths

    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    monkeypatch.setattr(paths, "SAVE_DIR", save_dir)
    monkeypatch.setattr(paths, "IN_GAME_IMG_DIR", str(tmp_path / "imgs"))
    os.makedirs(paths.IN_GAME_IMG_DIR, exist_ok=True)

    set_world_bible(minimal_bible)
    state_mgr.state.player_name = "SaveTest"
    state_mgr.state.location = "Mid"
    state_mgr.state.inventory = ["key"]

    path = state_mgr.save(label="test_roundtrip")
    assert os.path.isfile(path)

    loaded = StateManager(GameState())
    loaded.load(path)
    assert loaded.state.player_name == "SaveTest"
    assert loaded.state.location == "Mid"
    assert "key" in loaded.state.inventory
    assert get_world_bible() is not None
