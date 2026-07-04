"""MB dialogue — thinking-model strip and empty-LLM fallback."""


def test_strip_llm_thinking_keeps_post_think_answer(game_module):
    think_open = chr(60) + "think" + chr(62)
    think_close = chr(60) + "/think" + chr(62)
    raw = f"{think_open}planning...{think_close}Go east and gather magic items."
    out = game_module._strip_llm_thinking(raw)
    assert "Go east" in out
    assert "planning" not in out


def test_mb_offline_guidance_when_llm_empty(game_module, minimal_game_world, monkeypatch):
    """MB must not return 'MB says:' with nothing after a failed LLM call."""
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    mb = game_module.MasterBeast(current_room=1, llm_npc=False)
    room.npc = mb
    player.current_room = room
    game_module.playerOne = player
    game_module.rooms = minimal_game_world["rooms"]
    game_module.difficulty_g = "easy"

    monkeypatch.setattr(game_module, "ollama_generate", lambda *a, **k: "")
    reply = mb.llM_generate_self_dialogue("what should I do", player.get_item_names())
    assert "MB says" in reply or "🌟 MB" in reply
    assert len(reply.strip()) > 40
    assert "go east" in reply.lower() or "explore" in reply.lower() or "magic" in reply.lower()
