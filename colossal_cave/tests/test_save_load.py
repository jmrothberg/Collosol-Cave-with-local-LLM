"""Save/load round-trip tests using pickle."""

import os
import pickle


def test_pickle_roundtrip_simple_state(tmp_save_dir):
    """Verify pickle preserves basic game state dict."""
    state = {"health": 100, "wealth": 50, "location": 1, "inventory": ["sword"]}
    path = os.path.join(tmp_save_dir, "Adv_test.pkl")
    with open(path, "wb") as f:
        pickle.dump(state, f)

    with open(path, "rb") as f:
        loaded = pickle.load(f)

    assert loaded["health"] == 100
    assert loaded["location"] == 1
    assert "sword" in loaded["inventory"]


def test_pickle_file_created(tmp_save_dir, three_room_world):
    path = os.path.join(tmp_save_dir, "Adv_roundtrip.pkl")
    # Pickle only plain data (room numbers), not stub objects with lambdas
    room_numbers = list(three_room_world.keys())
    with open(path, "wb") as f:
        pickle.dump({"rooms": room_numbers, "old_room": 2}, f)
    assert os.path.isfile(path)

    with open(path, "rb") as f:
        data = pickle.load(f)
    assert data["old_room"] == 2
    assert 1 in data["rooms"]
