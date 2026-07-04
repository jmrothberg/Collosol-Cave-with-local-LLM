"""Optional live Ollama tests — skipped in CI unless ollama is running."""

import pytest

pytest.importorskip("ollama")


@pytest.mark.ollama
def test_ollama_list_models():
    import ollama
    try:
        models = ollama.list()
        assert models is not None
    except Exception:
        pytest.skip("Ollama not running")
