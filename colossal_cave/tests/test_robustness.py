"""Regression tests for defensive paths — null neighbors, bad LLM JSON, trophy edge cases."""

import os
import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("gradio")
pytest.importorskip("torch")

from engine.graph_map import sanitize_connected_rooms, relayout_rooms
from engine.items import safe_items
from engine.llm_commands import parse_llm_json_response
from engine.map_utils import draw_map_png
from engine.trophy import check_trophy
from engine.ux_helpers import adjacent_cues, format_room_exits


def test_safe_items_none_entity():
    assert safe_items(None) == []


def test_safe_items_none_items_attr():
    assert safe_items(SimpleNamespace(items=None)) == []


def test_parse_scalar_entity_category_coerced_to_list():
    raw = '{"chosen_action":["take"],"entity_categories":{"room_items":"sword"},"confidence":0.9}'
    result = parse_llm_json_response(raw)
    assert result["entities"]["room_items"] == ["sword"]


def test_sanitize_connected_rooms_drops_none_neighbors(three_room_world):
    three_room_world[1].connected_rooms["south"] = None
    sanitize_connected_rooms(three_room_world)
    assert "south" not in three_room_world[1].connected_rooms


def test_format_room_exits_skips_none_neighbor(three_room_world):
    three_room_world[1].connected_rooms["south"] = None
    text = format_room_exits(three_room_world[1], "easy")
    assert "south" not in text.lower()
    assert "east" in text.lower()


def test_adjacent_cues_skips_none_neighbor(three_room_world):
    three_room_world[1].connected_rooms["south"] = None
    cues = adjacent_cues(three_room_world[1], "easy")
    assert "south" not in cues.lower()


def test_relayout_rooms_tolerates_none_neighbor(three_room_world):
    three_room_world[1].connected_rooms["south"] = None
    relayout_rooms(three_room_world)
    assert "south" not in three_room_world[1].connected_rooms


def test_draw_map_png_with_none_neighbor(three_room_world, tmp_path):
    three_room_world[1].connected_rooms["south"] = None
    out = tmp_path / "none_neighbor.png"
    img = draw_map_png(three_room_world, output_path=str(out), current_room_number=1)
    assert img.size[0] > 0
    assert out.exists()


def test_trophy_not_awarded_while_treasure_remains(game_module, three_room_world, minimal_player):
    """Game trophy helper must still see treasures after normalize."""
    treasure = game_module.Treasure("Gold Coin", 10, runllm=False)
    three_room_world[2].items = [treasure]
    trophies = []
    console, _, _ = check_trophy(
        "treasure",
        game_module._room_still_has_treasures,
        "you got all treasures",
        three_room_world,
        trophies,
        minimal_player,
    )
    assert "Congratulations" not in console


def test_trophy_awarded_when_no_treasures_remain(game_module, three_room_world, minimal_player):
    for room in three_room_world.values():
        room.items = []
    trophies = []
    console, _, _ = check_trophy(
        "treasure",
        game_module._room_still_has_treasures,
        "you got all treasures",
        three_room_world,
        trophies,
        minimal_player,
    )
    assert "Congratulations" in console


def test_trophy_not_awarded_while_monsters_remain(game_module, three_room_world, minimal_player):
    goblin = game_module.Monster("Goblin", 50, 10, runllm_mon=False)
    three_room_world[3].monsters = [goblin]
    trophies = []
    console, _, _ = check_trophy(
        "monster",
        game_module._room_still_has_monsters,
        "you killed all monsters",
        three_room_world,
        trophies,
        minimal_player,
    )
    assert "Congratulations" not in console


def test_move_none_neighbor_does_not_corrupt_player(game_module, minimal_game_world):
    player = minimal_game_world["player"]
    room = minimal_game_world["r1"]
    room.connected_rooms["south"] = None
    player.current_room = room
    result = player.move("south")
    assert player.current_room is room
    assert "can't go" in result.lower()


def test_execute_action_take_with_string_entity(game_module, test_player):
    game_module.playerOne = test_player
    sword = game_module.Weapon("Iron Sword", 20, runllm=False)
    test_player.current_room.items.append(sword)
    entities = {k: [] for k in [
        "location", "room_items", "monsters", "npcs", "npc_items",
        "my_items", "my_magic", "weapons", "my_armor", "worn_armor", "directions", "values",
    ]}
    entities["room_items"] = ["Iron Sword"]
    result = game_module.execute_action(["take"], entities, "take sword", confidence=1.0)
    console = result[0] if isinstance(result, tuple) else result
    assert "Iron Sword" in console or "took" in console.lower()


def test_load_normalizes_items_none(game_module, minimal_game_world, tmp_path, monkeypatch):
    rooms = minimal_game_world["rooms"]
    player = minimal_game_world["player"]
    rooms[1].items = None
    rooms[2].monsters = None
    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    monkeypatch.setattr(game_module, "game_dir", save_dir)
    with open(os.path.join(save_dir, "Adv_norm_test.pkl"), "wb") as f:
        pickle.dump({"rooms": rooms, "playerOne": player, "old_room": rooms[1],
                     "trophies": [], "cheat_mode": False, "magicword": "xyzzy",
                     "difficulty_g": "easy", "adventure_log": ""}, f)
    monkeypatch.setattr(game_module, "_migrate_legacy_videos", lambda r: None)
    monkeypatch.setattr(game_module, "_ensure_master_beast", lambda r: None)
    monkeypatch.setattr(game_module, "_patch_all_npcs", lambda r: None)
    loaded_rooms, loaded_player, _ = game_module.load_game_state("norm_test")
    assert isinstance(loaded_rooms[1].items, list)
    assert isinstance(loaded_rooms[2].monsters, list)


def test_talk_to_functions_returns_thirteen_outputs(game_module, minimal_game_world, tmp_path, monkeypatch):
    gm = game_module
    player = minimal_game_world["player"]
    rooms = minimal_game_world["rooms"]
    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    monkeypatch.setattr(gm, "game_dir", save_dir)
    with open(os.path.join(save_dir, "Adv_tuple_test.pkl"), "wb") as f:
        pickle.dump({"rooms": rooms, "playerOne": player, "old_room": rooms[1],
                     "trophies": [], "cheat_mode": False, "magicword": "xyzzy",
                     "difficulty_g": "easy", "adventure_log": ""}, f)
    monkeypatch.setattr(gm, "_migrate_legacy_videos", lambda r: None)
    monkeypatch.setattr(gm, "_ensure_master_beast", lambda r: None)
    monkeypatch.setattr(gm, "_patch_all_npcs", lambda r: None)
    monkeypatch.setattr(gm, "draw_map_png", lambda *a, **k: None)
    monkeypatch.setattr(gm, "check_trophy", lambda *a, **k: ("", [], []))
    outputs = gm.talk_to_functions(
        " ", "tuple_test", "Easy", "Load-Game", "", "", False, False, False,
        "test-model", "test-diffuser", "", False, play_llm_name="gemma-4-12B-it-8bit",
    )
    assert len(outputs) == 13


def test_load_game_long_description_uses_gr_update(game_module, minimal_game_world, tmp_path, monkeypatch):
    gm = game_module
    player = minimal_game_world["player"]
    rooms = minimal_game_world["rooms"]
    rooms[1].full_description = "A long expanded room description for testing."
    save_dir = str(tmp_path / "saves")
    os.makedirs(save_dir, exist_ok=True)
    monkeypatch.setattr(gm, "game_dir", save_dir)
    with open(os.path.join(save_dir, "Adv_longdesc.pkl"), "wb") as f:
        pickle.dump({"rooms": rooms, "playerOne": player, "old_room": rooms[1],
                     "trophies": [], "cheat_mode": False, "magicword": "xyzzy",
                     "difficulty_g": "easy", "adventure_log": ""}, f)
    monkeypatch.setattr(gm, "_migrate_legacy_videos", lambda r: None)
    monkeypatch.setattr(gm, "_ensure_master_beast", lambda r: None)
    monkeypatch.setattr(gm, "_patch_all_npcs", lambda r: None)
    monkeypatch.setattr(gm, "draw_map_png", lambda *a, **k: None)
    monkeypatch.setattr(gm, "check_trophy", lambda *a, **k: ("", [], []))
    outputs = gm.talk_to_functions(
        " ", "longdesc", "Easy", "Load-Game", "", "", False, False, False,
        "test-model", "test-diffuser", "", False, play_llm_name="gemma-4-12B-it-8bit",
    )
    long_desc = outputs[8]
    assert isinstance(long_desc, dict)
    assert long_desc.get("value")
    assert "Error generating" not in long_desc.get("value", "")


def test_sanitize_long_description_falls_back_on_ollama_error(game_module):
    short = "A narrow tunnel, slick with moisture."
    assert game_module._sanitize_long_description("Error generating response", short) == short
    assert game_module._sanitize_long_description("", short) == short
    assert game_module._sanitize_long_description("Expanded prose here.", short) == "Expanded prose here."


def test_new_game_ollama_down_returns_error(game_module, monkeypatch):
    gm = game_module
    monkeypatch.setattr(gm, "check_ollama_running", lambda: False)
    monkeypatch.setattr(gm, "_llm_model_available", lambda name: False)
    outputs = gm.talk_to_functions(
        " ", "", "Easy", "New-Game", "testgame", "9", False, False, False,
        "test-model", "test-diffuser", "", False, play_llm_name="gemma-4-12B-it-8bit",
    )
    assert "not available" in outputs[4]


def test_game_module_room_still_has_treasures(game_module, minimal_game_world):
    """Game trophy helpers treat items=None as empty, not 'all collected'."""
    room = minimal_game_world["r1"]
    room.items = None
    assert game_module._room_still_has_treasures(room) is False
    room.items = []
    assert game_module._room_still_has_treasures(room) is False


def test_game_module_room_still_has_monsters(game_module, minimal_game_world):
    room = minimal_game_world["r1"]
    room.monsters = None
    assert game_module._room_still_has_monsters(room) is False
    goblin = game_module.Monster("Goblin", 50, 10, runllm_mon=False)
    room.monsters = [goblin]
    assert game_module._room_still_has_monsters(room) is True


def test_ollama_has_model_matches_base_tag(game_module, monkeypatch):
    monkeypatch.setattr(game_module, "_ollama_installed_model_names", lambda: ["qwen3.6:27b", "gemma2:latest"])
    assert game_module._ollama_has_model("qwen3.6:27b") is True
    assert game_module._ollama_has_model("gemma2") is True
    assert game_module._ollama_has_model("gemma-4-12B-it-8bit") is False


def test_log_llm_inventory_writes_missing_model(game_module, minimal_game_world, tmp_path, monkeypatch):
    gm = game_module
    game_dir = str(tmp_path / "games")
    os.makedirs(game_dir, exist_ok=True)
    monkeypatch.setattr(gm, "_ollama_installed_model_names", lambda: ["qwen3.6:27b"])
    monkeypatch.setattr(gm, "check_ollama_running", lambda: True)
    monkeypatch.setattr(gm, "resolve_mlx_folder_path", lambda name: None)
    monkeypatch.setattr(gm, "log_game_event", gm.log_game_event)
    from engine.debug_log import start_debug_session
    start_debug_session(game_dir, "llm_inv_test")
    gm._log_llm_inventory(context="test", creation="qwen3.6:27b", play="missing-model", diffuser="none")
    from engine.debug_log import get_debug_log_path
    with open(get_debug_log_path(), encoding="utf-8") as f:
        text = f.read()
    assert "LLM_SETUP" in text
    assert "play_ok=False" in text
    assert "LLM_MISSING" in text


def test_log_llm_inventory_mlx_folder_ok(game_module, tmp_path, monkeypatch):
    gm = game_module
    game_dir = str(tmp_path / "games")
    os.makedirs(game_dir, exist_ok=True)
    monkeypatch.setattr(gm, "check_ollama_running", lambda: False)
    monkeypatch.setattr(gm, "resolve_mlx_folder_path", lambda name: "/fake/gemma" if name == "gemma-4-12B-it-8bit" else None)
    from engine.debug_log import start_debug_session, get_debug_log_path
    start_debug_session(game_dir, "mlx_ok_test")
    gm._log_llm_inventory(context="test", creation="gemma-4-12B-it-8bit", play="gemma-4-12B-it-8bit", diffuser="none")
    with open(get_debug_log_path(), encoding="utf-8") as f:
        text = f.read()
    assert "play_ok=True" in text
    assert "play_backend=mlx" in text
    assert "LLM_MISSING" not in text


def test_draw_and_save_skips_when_pipe_none(game_module, test_player):
    game_module.pipe = None
    result = test_player.draw_and_save("test prompt")
    assert result == ""

