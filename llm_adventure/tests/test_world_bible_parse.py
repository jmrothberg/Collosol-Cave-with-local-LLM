"""Tests for JSON parse helpers."""

from engine.parse import extract_json_from_text, fix_common_json_errors, strip_llm_thinking_blocks


def test_strip_thinking_block():
    raw = "<think>secret</think>Hello world"
    assert strip_llm_thinking_blocks(raw) == "Hello world"


def test_fix_trailing_comma():
    fixed = fix_common_json_errors('{"state_updates": {"move_to": "Cave"},}')
    import json

    parsed = json.loads(fixed)
    assert parsed["state_updates"]["move_to"] == "Cave"


def test_extract_json_from_text():
    text = 'Narration here\n{"state_updates": {"add_note": "found key"}, "images": []}'
    objs = extract_json_from_text(text)
    assert len(objs) >= 1
    assert "state_updates" in objs[0]
