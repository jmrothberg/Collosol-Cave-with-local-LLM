"""Load-Game should restore state without heavy model setup or MB LLM intro."""
import os
import pickle
from unittest.mock import MagicMock


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
        play_llm_name="gemma-4-12B-it-8bit",
    )

    setup_mock.assert_not_called()
    assert gm.playerOne is not None
    assert "Loaded game" in outputs[4]  # console_output
    assert gm.npc_introduced is True


def test_monster_combat_after_load_without_initialize(game_module, minimal_game_world, tmp_path, monkeypatch):
    """Loaded games must not crash on first monster-room turn (you_started_it defined)."""
    gm = game_module
    player = minimal_game_world["player"]
    rooms = minimal_game_world["rooms"]
    player.current_room = rooms[2]
    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    monkeypatch.setattr(gm, "game_dir", save_dir)
    with open(os.path.join(save_dir, "Adv_combat_load.pkl"), "wb") as f:
        pickle.dump({"rooms": rooms, "playerOne": player, "old_room": rooms[2],
                     "trophies": [], "cheat_mode": False, "magicword": "xyzzy",
                     "difficulty_g": "easy", "adventure_log": ""}, f)

    if hasattr(gm, "you_started_it"):
        del gm.you_started_it  # simulate Load-Game skipping initialize_game_variables

    monkeypatch.setattr(gm, "_ensure_llm_diffuser_ready", lambda *a, **k: None)
    monkeypatch.setattr(gm, "_ensure_master_beast", lambda r: None)
    monkeypatch.setattr(gm, "_patch_all_npcs", lambda r: None)
    monkeypatch.setattr(gm, "_migrate_legacy_videos", lambda r: None)
    monkeypatch.setattr(gm, "draw_map_png", lambda *a, **k: None)
    monkeypatch.setattr(gm, "check_trophy", lambda *a, **k: ("", [], []))
    monkeypatch.setattr(gm, "save_game_state", lambda *a, **k: "quicksave")

    # Load then walk into monster room — must not NameError on you_started_it
    gm.talk_to_functions(" ", "combat_load", "Easy", "Load-Game", "", "", False, False, False,
                         "test-model", "test-diffuser", "", False, play_llm_name="gemma-4-12B-it-8bit")
    outputs = gm.talk_to_functions("go north", "combat_load", "Easy", "Play-Game", "", "", False, False, False,
                                   "test-model", "test-diffuser", "", False, play_llm_name="gemma-4-12B-it-8bit")
    assert hasattr(gm, "you_started_it")
    assert "moved north" in outputs[4].lower() or "Goblin" in outputs[4]
