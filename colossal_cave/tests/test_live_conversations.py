"""Live MLX conversation tests — load a real save and run player commands."""

import os

import pytest

_SAVE_NAME = os.environ.get("COLOSSAL_CAVE_TEST_SAVE", "July_5_2026_16_")
_PLAY_MODEL = os.environ.get("COLOSSAL_CAVE_PLAY_LLM", "gemma-4-12B-it-8bit")
_BAD_PHRASES = (
    "Error generating response",
    "I don't understand",
    "MB'S SPECIAL GUIDANCE",  # offline-only MB fallback
    "MLX_ERROR",
)


def _mlx_model_ready(game_module):
    path = game_module.resolve_mlx_folder_path(_PLAY_MODEL)
    if path and os.path.isfile(os.path.join(path, "config.json")):
        return True
    return game_module._llm_model_available(_PLAY_MODEL)


def _load_user_save(game_module):
    game_module.play_model = _PLAY_MODEL
    game_module.model = _PLAY_MODEL
    game_module.rooms, game_module.playerOne, game_module.old_room = (
        game_module.load_game_state(_SAVE_NAME)
    )
    assert game_module.rooms is not None
    assert game_module.playerOne is not None


def _assert_good_response(text, label):
    assert text, f"{label}: empty response"
    for bad in _BAD_PHRASES:
        assert bad not in text, f"{label}: hit fallback/error phrase {bad!r} in {text[:200]!r}"


@pytest.fixture
def loaded_save(game_module):
    if not _mlx_model_ready(game_module):
        pytest.skip(f"MLX play model {_PLAY_MODEL!r} not available")
    _load_user_save(game_module)
    return game_module


@pytest.mark.mlx
def test_gradio_talk_to_functions_mb_loaded_save(game_module):
    """Full Gradio play turn on loaded save — adventure log must show real MB, not offline."""
    from engine.ux_helpers import get_adventure_log_text, reset_adventure_log

    gm = game_module
    if not _mlx_model_ready(gm):
        pytest.skip(f"MLX play model {_PLAY_MODEL!r} not available")
    reset_adventure_log()
    gm.load_adventure_data()
    gm.play_model = _PLAY_MODEL
    gm.model = _PLAY_MODEL
    gm.talk_to_functions(
        command="",
        name_of_game="July_5_2026_16_",
        difficulty="Easy",
        game_state="Load-Game",
        new_game_name="",
        number_of_rooms="25",
        llm_diff=False,
        llm_mon=False,
        llm_item=False,
        name_of_llm=_PLAY_MODEL,
        name_of_diffuser="FLUX2-klein-9B-mlx-8bit",
        art_style="",
        generate_videos=False,
        play_llm_name=_PLAY_MODEL,
    )
    gm.talk_to_functions(
        command="hi, how do i win at this adventure",
        name_of_game="July_5_2026_16_",
        difficulty="Easy",
        game_state="Play-Game",
        new_game_name="",
        number_of_rooms="25",
        llm_diff=False,
        llm_mon=False,
        llm_item=False,
        name_of_llm=_PLAY_MODEL,
        name_of_diffuser="FLUX2-klein-9B-mlx-8bit",
        art_style="",
        generate_videos=False,
        play_llm_name=_PLAY_MODEL,
    )
    log = get_adventure_log_text()
    _assert_good_response(log, "gradio loaded save MB")
    assert "MB says:" in log


@pytest.mark.mlx
def test_mb_new_game_conversation(game_module):
    """New-Game path: MB must not hit NameError on difficulty_g (user-reported offline fallback)."""
    gm = game_module
    if not _mlx_model_ready(gm):
        pytest.skip(f"MLX play model {_PLAY_MODEL!r} not available")
    gm.initialize_game_variables()
    gm.load_adventure_data()
    gm.play_model = _PLAY_MODEL
    gm.model = _PLAY_MODEL
    # Deliberately do NOT set difficulty_g — mirrors bug before module-level init
    if hasattr(gm, "difficulty_g"):
        del gm.difficulty_g
    gm.new_game(num_rooms=25, runllm_diff=False, name_of_game="pytest_mb_new")
    gm._ensure_llm_diffuser_ready(
        diff_name="FLUX2-klein-9B-mlx-8bit",
        llm_name=_PLAY_MODEL,
        play_llm_name=_PLAY_MODEL,
    )
    out = gm.process_command(
        "MB how can i win this adventure?",
        gm.playerOne,
        gm.playerOne.current_room,
    )
    _assert_good_response(out, "MB new game")
    assert "MB says:" in out


@pytest.mark.mlx
def test_mb_win_conversation(loaded_save):
    gm = loaded_save
    out = gm.process_command(
        "ask MB how to win the game",
        gm.playerOne,
        gm.playerOne.current_room,
    )
    _assert_good_response(out, "MB win")
    assert "MB" in out or "mb" in out.lower()


@pytest.mark.mlx
def test_help_with_riddle_conversation(loaded_save):
    gm = loaded_save
    out = gm.process_command(
        "help me with the riddle",
        gm.playerOne,
        gm.playerOne.current_room,
    )
    _assert_good_response(out, "riddle help")


@pytest.mark.mlx
def test_examin_item_conversation(loaded_save):
    gm = loaded_save
    items = gm._safe_player_items(gm.playerOne)
    if not items:
        pytest.skip("No items in player inventory for this save")
    target = items[0]
    target.description = "Error generating response"
    out = gm.process_command(
        f"examin the {target.name}",
        gm.playerOne,
        gm.playerOne.current_room,
    )
    _assert_good_response(out, "examin item")
    assert "Error generating response" not in out


@pytest.mark.mlx
def test_study_item_conversation(loaded_save):
    gm = loaded_save
    items = gm._safe_player_items(gm.playerOne)
    if not items:
        pytest.skip("No items in player inventory for this save")
    target = items[0]
    target.description = "Error generating response"
    out = gm.process_command(
        f"study the {target.name}",
        gm.playerOne,
        gm.playerOne.current_room,
    )
    _assert_good_response(out, "study item")
    assert "Error generating response" not in out
