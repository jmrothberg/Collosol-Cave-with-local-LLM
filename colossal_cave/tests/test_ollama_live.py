"""Optional live Ollama tests — skipped in CI unless ollama is running."""

import os

import pytest

pytest.importorskip("ollama")


def _ollama_running():
    try:
        import ollama
        ollama.list()
        return True
    except Exception:
        return False


def _play_model_name():
    return os.environ.get("COLOSSAL_CAVE_PLAY_LLM", "gemma-4-12B-it-8bit")


@pytest.mark.ollama
def test_ollama_list_models():
    import ollama
    if not _ollama_running():
        pytest.skip("Ollama not running")
    models = ollama.list()
    assert models is not None


@pytest.mark.ollama
def test_play_model_registered_or_mlx_path_exists():
    play_llm = _play_model_name()
    mlx_path = os.path.join(
        os.environ.get("COLOSSAL_CAVE_MLX_MODELS_DIR", os.path.expanduser("~/MLX_Models")),
        play_llm,
        "config.json",
    )
    if os.path.isfile(mlx_path):
        return
    if not _ollama_running():
        pytest.skip("Ollama not running")
    import ollama
    response = ollama.list()
    names = []
    if hasattr(response, "models"):
        for m in response.models:
            names.append(getattr(m, "model", str(m)))
    elif isinstance(response, dict):
        for m in response.get("models", []):
            names.append(m.get("name", str(m)) if isinstance(m, dict) else str(m))
    if not any(play_llm in n for n in names):
        pytest.skip(f"Play model {play_llm} not in ollama list")


@pytest.mark.ollama
@pytest.mark.mlx
def test_ollama_smoke_generate_play_model():
    if not _ollama_running():
        pytest.skip("Ollama not running")
    import ollama
    play_llm = _play_model_name()
    try:
        response = ollama.generate(model=play_llm, prompt="Say OK", options={"num_predict": 5})
        assert response.get("response", "").strip() != ""
    except Exception as e:
        pytest.skip(f"Play model {play_llm} not available: {e}")
