"""Tests for adventure_dataRA.json schema validation."""

from engine.data import load_adventure_data_file, validate_adventure_data


def test_load_canonical_adventure_data(adventure_data_path):
    data = load_adventure_data_file(adventure_data_path)
    valid, issues = validate_adventure_data(data)
    assert valid, issues


def test_room_count_at_least_25(adventure_data_path):
    data = load_adventure_data_file(adventure_data_path)
    assert len(data["room_descs_riddles"]) >= 25


def test_riddle_rooms_have_magic_items(adventure_data_path):
    data = load_adventure_data_file(adventure_data_path)
    for idx, room in enumerate(data["room_descs_riddles"]):
        if room.get("riddles"):
            assert room.get("magic_items"), f"room {idx} has riddle but no magic_items"


def test_invalid_data_missing_keys():
    valid, issues = validate_adventure_data({})
    assert not valid
    assert any("room_descs_riddles" in i for i in issues)
