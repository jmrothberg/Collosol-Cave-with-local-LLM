"""Tests for auto_repair_world_bible and synthetic chain."""

from engine.world_bible.repair import auto_repair_world_bible, synthesize_solution_chain
from engine.world_bible.validate import validate_world_bible_solvability


def test_auto_repair_idempotent(default_cave_bible):
    first = auto_repair_world_bible(default_cave_bible)
    second = auto_repair_world_bible(default_cave_bible)
    assert second == []


def test_synthetic_chain_when_missing():
    wb = {
        "locations": [
            {"name": "A", "description": "a", "exits": ["B"]},
            {"name": "B", "description": "b", "exits": ["A", "C"]},
            {"name": "C", "description": "c", "exits": ["B"]},
            {"name": "D", "description": "d", "exits": ["C"]},
            {"name": "E", "description": "e", "exits": ["D"]},
        ],
        "key_items": [
            {"name": "tool", "purpose": "useful"},
            {"name": "prize", "purpose": "win"},
            {"name": "note", "purpose": "hint"},
            {"name": "lens", "purpose": "see"},
        ],
        "win_condition": {"required_items": ["prize"], "required_location": "A", "description": "Win."},
        "item_locations": {"tool": "A: here", "prize": "C: there", "note": "B: shelf", "lens": "D: case"},
        "solution_chain": [],
    }
    repairs = synthesize_solution_chain(wb)
    assert repairs
    assert len(wb["solution_chain"]) >= 2


def test_auto_repair_adds_missing_item_locations(minimal_bible):
    del minimal_bible["item_locations"]["map"]
    repairs = auto_repair_world_bible(minimal_bible)
    assert any("map" in r for r in repairs)
    assert "map" in minimal_bible["item_locations"]
