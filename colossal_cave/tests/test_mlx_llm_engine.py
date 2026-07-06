"""Tests for direct MLX folder loading (engine/mlx_llm.py)."""

import json
import os

import pytest


@pytest.fixture
def mlx_tmp_dir(tmp_path, monkeypatch):
    """Minimal MLX-style folder with config.json."""
    folder = tmp_path / "test-mlx-model"
    folder.mkdir()
    (folder / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("COLOSSAL_CAVE_MLX_MODELS_DIR", str(tmp_path))
    return folder


def test_resolve_mlx_folder_path(mlx_tmp_dir):
    from engine.mlx_llm import resolve_mlx_folder_path

    assert resolve_mlx_folder_path("test-mlx-model") == str(mlx_tmp_dir.resolve())
    assert resolve_mlx_folder_path("ollama:qwen3.6:latest") is None
    assert resolve_mlx_folder_path("missing-folder") is None


def test_model_backend_prefers_mlx(mlx_tmp_dir):
    from engine.mlx_llm import model_backend

    assert model_backend("test-mlx-model") == "mlx"
    assert model_backend("ollama:gemma2:latest") == "ollama"


def test_ollama_generate_routes_to_mlx(game_module, mlx_tmp_dir, monkeypatch):
    gm = game_module
    monkeypatch.setattr(gm, "resolve_mlx_folder_path", lambda name: str(mlx_tmp_dir) if name == "test-mlx-model" else None)
    monkeypatch.setattr(gm, "mlx_generate_text", lambda *a, **k: "MLX says hello")
    out = gm.ollama_generate("User: hi", llm_model="test-mlx-model")
    assert out == "MLX says hello"


def test_llm_available_via_mlx_without_ollama(game_module, monkeypatch):
    gm = game_module
    monkeypatch.setattr(gm, "check_ollama_running", lambda: False)
    monkeypatch.setattr(
        gm,
        "resolve_mlx_folder_path",
        lambda name: "/fake/gemma" if name == "gemma-4-12B-it-8bit" else None,
    )
    # CHANGE (CI): model_backend is imported from engine.mlx_llm — must mock on gm, not rely on ~/MLX_Models
    monkeypatch.setattr(
        gm,
        "model_backend",
        lambda name: "mlx" if name == "gemma-4-12B-it-8bit" else None,
    )
    assert gm._llm_model_available("gemma-4-12B-it-8bit") is True
    assert gm.model_backend("gemma-4-12B-it-8bit") == "mlx"


def test_new_game_blocked_when_no_backend(game_module, monkeypatch):
    gm = game_module
    monkeypatch.setattr(gm, "check_ollama_running", lambda: False)
    monkeypatch.setattr(gm, "_llm_model_available", lambda name: False)
    outputs = gm.talk_to_functions(
        " ", "", "Easy", "New-Game", "testgame", "9", False, False, False,
        "missing-model", "test-diffuser", "", False, play_llm_name="missing-model",
    )
    assert "not available" in outputs[4]


def test_saved_llms_lists_mlx_first(game_module, mlx_tmp_dir, monkeypatch):
    gm = game_module
    monkeypatch.setattr(gm, "list_mlx_model_folders", lambda: ["test-mlx-model", "other-mlx"])
    monkeypatch.setattr(gm, "_ollama_installed_model_names", lambda: ["qwen3.6:27b"])
    choices = gm.saved_llms()
    assert choices[0] in ("test-mlx-model", gm.default_play_llm())
    assert "test-mlx-model" in choices
    assert "ollama:qwen3.6:27b" in choices
