"""Tests for adventure log helpers."""

from engine.ux_helpers import (
    append_adventure_log,
    get_adventure_log_text,
    is_empty_game_response,
    reset_adventure_log,
)


def test_adventure_log_append():
    reset_adventure_log()
    append_adventure_log("go east", "You moved east.")
    text = get_adventure_log_text()
    assert "> go east" in text
    assert "You moved east" in text


def test_empty_response_detection():
    assert is_empty_game_response("") is True
    assert is_empty_game_response("Error generating response") is True
    assert is_empty_game_response("You moved north.") is False
