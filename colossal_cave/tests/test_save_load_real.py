"""Real save/load round-trip with Room and Player objects."""

import os
import pickle
from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


def test_save_load_roundtrip_dict_payload(game_module, minimal_game_world, tmp_path, monkeypatch):
    player = minimal_game_world["player"]
    rooms = minimal_game_world["rooms"]
    game_module.playerOne = player
    game_module.rooms = rooms
    game_module.old_room = rooms[1]
    game_module.trophies = ["explorer"]
    game_module.cheat_mode = True
    game_module.magicword = "testword"
    game_module.difficulty_g = "easy"
    game_module.art_style_g = "teststyle"

    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    monkeypatch.setattr(game_module, "game_dir", save_dir)

    name = game_module.save_game_state("pytest_save")
    assert "pytest_save" in str(name)

    # Reset globals
    game_module.rooms = {}
    game_module.playerOne = None

    loaded_rooms, loaded_player, loaded_old = game_module.load_game_state("pytest_save_3_teststyle")
    assert loaded_player is not None
    assert loaded_rooms is not None
    assert loaded_player.name == "Hero"
    assert game_module.trophies == ["explorer"]
    assert game_module.cheat_mode is True


def test_legacy_tuple_pickle_still_loads(game_module, minimal_game_world, tmp_path, monkeypatch):
    player = minimal_game_world["player"]
    rooms = minimal_game_world["rooms"]
    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "Adv_legacy_3_art.pkl")
    with open(path, "wb") as f:
        pickle.dump((rooms, player, rooms[1]), f)
    monkeypatch.setattr(game_module, "game_dir", save_dir)
    loaded_rooms, loaded_player, _ = game_module.load_game_state("legacy_3_art")
    assert loaded_player.name == "Hero"
