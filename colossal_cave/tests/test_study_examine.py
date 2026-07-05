"""Study / examine commands — typos, bad LLM text, regeneration."""

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")


def test_preprocess_examin_typo():
    from engine.preprocess import preprocess_command

    assert preprocess_command("examin the ivory flute") == "examine the ivory flute"


def test_study_examin_ivory_flute_fast_path(game_module, test_player):
    flute = game_module.Treasure("Ivory Flute", 861, runllm=False)
    flute.description = "Error generating response"
    test_player.items.append(flute)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            game_module,
            "ollama_generate",
            lambda *a, **k: "A pale carved flute that hums softly in the dark.",
        )
        result, _images = test_player.study("Ivory Flute", "study the ivory flute")

    assert "Error generating" not in result
    assert "Ivory Flute" in result
    assert "861" in result


def test_examin_routes_via_process_command(game_module, test_player):
    flute = game_module.Treasure("Ivory Flute", 861, runllm=False)
    flute.description = "Error generating response"
    test_player.items.append(flute)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            game_module,
            "ollama_generate",
            lambda *a, **k: "An ancient ivory flute etched with cave runes.",
        )
        out = game_module.process_command(
            "examin the ivory flute", test_player, test_player.current_room
        )
        result = out[0] if isinstance(out, tuple) else out

    assert "Error generating" not in result
    assert "Ivory Flute" in result


def test_sanitize_llm_text_fallback(game_module):
    assert game_module._sanitize_llm_text("Error generating response", "ok") == "ok"
    assert game_module._sanitize_llm_text("  nice prose  ", "") == "nice prose"
