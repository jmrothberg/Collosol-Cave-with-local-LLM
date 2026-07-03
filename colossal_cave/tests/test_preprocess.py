"""Tests for engine.preprocess."""

from engine.preprocess import preprocess_command


def test_preprocess_attack_typo():
    assert preprocess_command("attck dragon") == "attack dragon"


def test_preprocess_take_typo():
    assert preprocess_command("tkae sword") == "take sword"


def test_preprocess_strips_whitespace():
    assert preprocess_command("  look  ") == "look"


def test_preprocess_preserves_unrelated_words():
    # "attac" is a substring of "attack" in the original typo table — use a neutral phrase
    assert preprocess_command("Look around") == "Look around"
