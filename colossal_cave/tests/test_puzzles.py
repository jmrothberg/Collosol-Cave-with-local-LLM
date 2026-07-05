"""Puzzle handlers — tool use and puzzle variety."""

from types import SimpleNamespace

from engine.puzzles import (
    PUZZLE_HANDLERS,
    handle_typed_answer,
    try_action_word_puzzle,
    try_feature_action,
)


def test_puzzle_handlers_registry():
    assert "item_solve" in PUZZLE_HANDLERS
    assert "typed_answer" in PUZZLE_HANDLERS


def test_action_word_puzzle_solves():
    class Magic:
        pass

    room = SimpleNamespace(
        riddles=["A riddle"],
        riddle_solved=False,
        riddle_action_words="play",
        riddle_magic_items=["Piano of Time"],
    )
    item = Magic()
    item.name = "Piano of Time"
    result = try_action_word_puzzle(room, "play", [item])
    assert result is not None
    assert room.riddle_solved


def test_typed_answer():
    room = SimpleNamespace(puzzle_answers=["silence", "quiet"])
    assert handle_typed_answer(room, "Silence") is True


def test_feature_action_requires_item():
    room = SimpleNamespace(
        features=[{
            "name": "piano",
            "verbs": ["play"],
            "requires_item": "Piano of Time",
            "success": "The piano plays.\n",
        }],
        riddles=["r"],
        riddle_solved=False,
    )
    result = try_feature_action(room, "play", None, [])
    assert "need" in result.lower()
