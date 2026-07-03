"""Tests for validate_world_bible and validate_world_bible_solvability."""

from engine.world_bible.repair import auto_repair_world_bible
from engine.world_bible.validate import validate_world_bible, validate_world_bible_solvability


def test_validate_world_bible_ok(minimal_bible):
    ok, issues = validate_world_bible(minimal_bible)
    assert ok, issues


def test_validate_missing_field():
    ok, issues = validate_world_bible({})
    assert not ok
    assert any("objectives" in i for i in issues)


def test_solvability_unreachable_room(minimal_bible):
    ok, gaps, _report = validate_world_bible_solvability(minimal_bible)
    assert not ok
    assert any("unreachable" in g for g in gaps)


def test_solvability_after_auto_repair(minimal_bible):
    auto_repair_world_bible(minimal_bible)
    ok, gaps, report = validate_world_bible_solvability(minimal_bible)
    assert ok, f"gaps={gaps}\n{report}"


def test_default_cave_solvable(default_cave_bible):
    ok, gaps, report = validate_world_bible_solvability(default_cave_bible)
    assert ok, f"gaps={gaps}\n{report}"
