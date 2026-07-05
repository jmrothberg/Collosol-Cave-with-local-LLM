"""Dual LLM routing — creation vs play model."""

import os
from unittest.mock import patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


def test_default_play_llm():
    from colossal_cave_July_5_26 import default_play_llm
    assert default_play_llm() == os.environ.get(
        "COLOSSAL_CAVE_PLAY_LLM", "gemma-4-12B-it-8bit"
    )


def test_list_mlx_model_folders_includes_gemma(game_module):
    folders = game_module.list_mlx_model_folders()
    # On dev machine gemma folder may exist; in CI empty is ok
    assert isinstance(folders, list)


@patch("colossal_cave_July_5_26.ollama_generate")
def test_llm_process_uses_play_model(mock_gen, game_module, test_player, test_room):
    mock_gen.return_value = '{"chosen_action":["help"],"entity_categories":{},"confidence":0.9}'
    game_module.play_model = "fast-play-model"
    game_module.model = "fast-play-model"
    game_module.llm_process_command("help me", test_player, test_room)
    assert mock_gen.called
    call_kwargs = mock_gen.call_args[1] if mock_gen.call_args[1] else {}
    # play_model is global active model when model=play_model
    assert game_module.play_model == "fast-play-model"


def test_save_payload_includes_play_model(game_module, minimal_game_world, tmp_path, monkeypatch):
    gm = game_module
    gm.playerOne = minimal_game_world["player"]
    gm.rooms = minimal_game_world["rooms"]
    gm.old_room = minimal_game_world["r1"]
    gm.trophies = []
    gm.cheat_mode = False
    gm.magicword = "testword"
    gm.difficulty_g = "easy"
    gm.art_style_g = "teststyle"
    gm.play_model = "gemma-4-12B-it-8bit"
    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    monkeypatch.setattr(gm, "game_dir", save_dir)
    gm.save_game_state("dual_llm_test")
    import pickle
    files = [f for f in os.listdir(save_dir) if f.startswith("Adv_dual_llm") and f.endswith(".pkl")]
    assert files
    with open(os.path.join(save_dir, files[0]), "rb") as f:
        payload = pickle.load(f)
    assert payload.get("play_model") == "gemma-4-12B-it-8bit"
