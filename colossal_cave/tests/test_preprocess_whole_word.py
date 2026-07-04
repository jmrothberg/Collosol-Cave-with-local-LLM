"""Tests for preprocess whole-word typo fix."""

from engine.preprocess import preprocess_command


def test_take_sword_not_corrupted():
    assert preprocess_command("take sword") == "take sword"


def test_tkae_typo_fixed():
    assert preprocess_command("tkae sword") == "take sword"


def test_attck_typo_fixed():
    assert preprocess_command("attck dragon") == "attack dragon"
