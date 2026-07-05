"""Tests for engine.directions 8-way compass."""

import math

from engine.directions import (
    OPPOSITES,
    snap_direction,
    normalize_direction_phrase,
    ranked_directions,
)


def test_snap_cardinal_when_straight():
    assert snap_direction(1, 0) == "east"
    assert snap_direction(0, 1) == "south"
    assert snap_direction(-1, 0) == "west"
    assert snap_direction(0, -1) == "north"


def test_snap_intercardinal_when_diagonal():
    assert snap_direction(1, 1) == "southeast"
    assert snap_direction(-1, 1) == "southwest"
    assert snap_direction(1, -1) == "northeast"
    assert snap_direction(-1, -1) == "northwest"


def test_opposites_8way():
    for a, b in OPPOSITES.items():
        assert OPPOSITES[b] == a


def test_normalize_southeast_variants():
    assert normalize_direction_phrase("south-east") == "southeast"
    assert normalize_direction_phrase("se") == "southeast"
    assert normalize_direction_phrase("south east") == "southeast"


def test_ranked_directions_prefers_cardinal_for_pure_east():
    assert ranked_directions(5, 0)[0] == "east"
