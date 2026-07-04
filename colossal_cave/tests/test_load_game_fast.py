"""Load-Game should restore state without heavy model setup or MB LLM intro."""
import os
import pickle
from unittest.mock import MagicMock

import pytest

pytest.importorskip("gradio")


def test_load_game_skips_llm_diffuser_setup(game_module, minimal_game_world, tmp_path, monkeypatch):
    """Load-Game restores pickle only — models load on first Play-Game turn."""
    gm = game_module
    player = minimal_game_world["player"]
    rooms = minimal_game_world["rooms"]
    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    monkeypatch.setattr(gm, "game_dir", save_dir)

    save_key = "resume_test"
    with open(os.path.join(save_dir, f"Adv_{save_key}.pkl"), "wb") as f:
        pickle.dump({
            "rooms": rooms,
            "playerOne": player,
            "old_room": rooms[1],
            "trophies": [],
            "cheat_mode": False,
            "magicword": "xyzzy",
            "difficulty_g": "easy",
            "adventure_log": "",
        }, f)

    gm._llm_diffuser_ready = False
    setup_mock = MagicMock()
    monkeypatch.setattr(gm, "set_up_llm_diffuser", setup_mock)
    monkeypatch.setattr(gm, "_ensure_master_beast", lambda r: None)
    monkeypatch.setattr(gm, "_patch_all_npcs", lambda r: None)
    monkeypatch.setattr(gm, "_migrate_legacy_videos", lambda r: None)

    outputs = gm.talk_to_functions(
        command=" ",
        name_of_game=save_key,
        difficulty="Easy",
        game_state="Load-Game",
        new_game_name="",
        number_of_rooms="9",
        llm_diff=False,
        llm_mon=False,
        llm_item=False,
        name_of_llm="test-model",
        name_of_diffuser="test-diffuser",
        art_style="",
        generate_videos=False,
    )

    setup_mock.assert_not_called()
    assert gm.playerOne is not None
    assert "Loaded game" in outputs[4]  # console_output
    assert gm.npc_introduced is True
