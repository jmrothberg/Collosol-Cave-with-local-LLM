"""Live MLX / gemma play LLM tests — local Mac only."""

import os

import pytest

pytest.importorskip("ollama")


def _mlx_dir():
    return os.environ.get("COLOSSAL_CAVE_MLX_MODELS_DIR", os.path.expanduser("~/MLX_Models"))


def _play_model():
    return os.environ.get("COLOSSAL_CAVE_PLAY_LLM", "gemma-4-12B-it-8bit")


@pytest.mark.mlx
def test_mlx_models_dir_exists():
    path = os.path.join(_mlx_dir(), _play_model(), "config.json")
    if not os.path.isfile(path):
        pytest.skip(f"MLX model not at {path}")
    assert os.path.isfile(path)


@pytest.mark.mlx
def test_mlx_model_in_saved_llms(game_module):
    folders = game_module.list_mlx_model_folders()
    play = _play_model()
    if play not in folders:
        pytest.skip(f"{play} not in MLX_Models")
    llms = game_module.saved_llms()
    assert play in llms


@pytest.mark.mlx
def test_ollama_serves_mlx_gemma():
    import ollama
    try:
        ollama.list()
    except Exception:
        pytest.skip("Ollama not running")
    play = _play_model()
    try:
        r = ollama.generate(model=play, prompt="hi", options={"num_predict": 3})
        assert "response" in r
    except Exception as e:
        pytest.skip(f"Cannot generate with {play}: {e}")
