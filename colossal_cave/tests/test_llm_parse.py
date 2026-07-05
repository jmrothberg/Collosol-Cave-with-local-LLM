"""Tests for engine.llm_commands parsing helpers."""

from engine.llm_commands import (
    apply_command_shortcuts,
    filter_entities_for_room,
    needs_clarification,
    parse_llm_json_response,
    DETERMINISTIC_ACTIONS,
)


def test_shortcut_north():
    assert apply_command_shortcuts("n") == "go north"


def test_shortcut_inventory():
    assert apply_command_shortcuts("i") == "inventory"


def test_shortcut_unknown_passthrough():
    assert apply_command_shortcuts("take sword") == "take sword"


def test_parse_valid_json():
    raw = '{"chosen_action":["take"],"entity_categories":{"room_items":["sword"]},"confidence":0.95}'
    result = parse_llm_json_response(raw)
    assert result["chosen_action"] == ["take"]
    assert result["entities"]["room_items"] == ["sword"]
    assert result["confidence"] == 0.95


def test_parse_typo_field_name():
    raw = '{"cchosen_action":["attack"],"entity_categories":{},"confidence":0.8}'
    result = parse_llm_json_response(raw)
    assert result["chosen_action"] == ["attack"]


def test_parse_invalid_json_low_confidence():
    result = parse_llm_json_response("not json at all")
    assert result["confidence"] == 0.3
    assert result["chosen_action"] == []


def test_parse_null_entity_categories():
    """LLM JSON mode can return entity_categories: null — must not crash."""
    raw = '{"chosen_action":["talk"],"entity_categories":null,"confidence":0.9}'
    result = parse_llm_json_response(raw)
    assert result["chosen_action"] == ["talk"]
    assert result["entities"]["room_items"] == []


def test_parse_scalar_entity_coerced_to_list():
    raw = '{"chosen_action":["take"],"entity_categories":{"room_items":"sword"},"confidence":0.9}'
    result = parse_llm_json_response(raw)
    assert result["entities"]["room_items"] == ["sword"]


def test_filter_entities_for_room():
    result = parse_llm_json_response(
        '{"chosen_action":["take"],"entity_categories":{"room_items":["sword","shield","all"]},"confidence":0.9}'
    )
    filtered = filter_entities_for_room(
        result, room_item_names=["sword"], monster_names=[], directions=["north"],
    )
    assert "sword" in filtered["entities"]["room_items"]
    assert "shield" not in filtered["entities"]["room_items"]
    assert "all" in filtered["entities"]["room_items"]


def test_needs_clarification_low_confidence_complex():
    assert needs_clarification(0.2, "attack") is True


def test_needs_clarification_low_confidence_movement():
    assert needs_clarification(0.2, "go") is False


def test_deterministic_actions_set():
    assert "inventory" in DETERMINISTIC_ACTIONS
    assert "map" in DETERMINISTIC_ACTIONS
