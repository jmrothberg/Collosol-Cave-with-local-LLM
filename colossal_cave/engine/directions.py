"""Shared 8-way compass: N, NE, E, SE, S, SW, W, NW."""

import math
from typing import Dict, List, Optional, Tuple

# Unit vectors (screen layout: x right, y down)
_EIGHT_WAY = (
    ("east", (1, 0)),
    ("southeast", (1, 1)),
    ("south", (0, 1)),
    ("southwest", (-1, 1)),
    ("west", (-1, 0)),
    ("northwest", (-1, -1)),
    ("north", (0, -1)),
    ("northeast", (1, -1)),
)

CARDINALS = frozenset({"north", "south", "east", "west"})
INTERCARDINALS = frozenset({"northeast", "northwest", "southeast", "southwest"})

OPPOSITES: Dict[str, str] = {
    "east": "west",
    "west": "east",
    "north": "south",
    "south": "north",
    "northeast": "southwest",
    "southwest": "northeast",
    "northwest": "southeast",
    "southeast": "northwest",
}

ABBREV: Dict[str, str] = {
    "north": "N",
    "south": "S",
    "east": "E",
    "west": "W",
    "northeast": "NE",
    "northwest": "NW",
    "southeast": "SE",
    "southwest": "SW",
}

# Cardinal shortcuts + intercardinal phrases for command parsing
_DIRECTION_WORDS = {
    "north": "north", "south": "south", "east": "east", "west": "west",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "up": "north", "down": "south",
    "northeast": "northeast", "ne": "northeast", "north east": "northeast",
    "northwest": "northwest", "nw": "northwest", "north west": "northwest",
    "southeast": "southeast", "se": "southeast", "south east": "southeast",
    "southwest": "southwest", "sw": "southwest", "south west": "southwest",
}

_CARDINAL_ANGLES = {
    "east": 0.0,
    "southeast": math.pi / 4,
    "south": math.pi / 2,
    "southwest": 3 * math.pi / 4,
    "west": math.pi,
    "northwest": -3 * math.pi / 4,
    "north": -math.pi / 2,
    "northeast": -math.pi / 4,
}

# Within 22.5° of a cardinal → use cardinal, not intercardinal
_CARDINAL_SNAP = math.pi / 8


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def layout_angle(dx: int, dy: int) -> float:
    """Bearing from room to neighbor in layout space."""
    if dx == 0 and dy == 0:
        return 0.0
    return math.atan2(dy, dx)


def ranked_directions(dx: int, dy: int) -> List[str]:
    """All 8 directions sorted best-match first for layout delta."""
    scored: List[Tuple[float, str]] = []
    for name, (vx, vy) in _EIGHT_WAY:
        length = math.hypot(vx, vy)
        score = (dx * vx + dy * vy) / length if length else 0.0
        scored.append((score, name))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in scored]


def snap_direction(dx: int, dy: int) -> str:
    """
    Pick direction label from layout delta.
    Clear cardinals → N/S/E/W; diagonal tunnels → NE/SE/SW/NW.
    """
    if dx == 0 and dy == 0:
        return "east"
    angle = layout_angle(dx, dy)
    # Prefer cardinal when tunnel aligns within ~22.5°
    best_card = None
    best_card_diff = math.pi
    for card in CARDINALS:
        diff = _angle_diff(angle, _CARDINAL_ANGLES[card])
        if diff < best_card_diff:
            best_card_diff = diff
            best_card = card
    if best_card and best_card_diff <= _CARDINAL_SNAP:
        return best_card
    # Otherwise nearest of all 8 (including intercardinals)
    best_name = "east"
    best_diff = math.pi * 2
    for name, target in _CARDINAL_ANGLES.items():
        diff = _angle_diff(angle, target)
        if diff < best_diff:
            best_diff = diff
            best_name = name
    return best_name


def edge_ambiguity(dx: int, dy: int) -> float:
    """Lower = clearer bearing (assign first)."""
    scores = []
    for name, (vx, vy) in _EIGHT_WAY:
        length = math.hypot(vx, vy)
        scores.append(dx * vx / length + dy * vy / length if length else 0.0)
    scores.sort(reverse=True)
    if len(scores) < 2:
        return 0.0
    return scores[0] - scores[1]


def normalize_direction_phrase(phrase: str) -> Optional[str]:
    """Normalize movement phrase to canonical direction name."""
    p = phrase.lower().strip().replace("-", " ")
    while "  " in p:
        p = p.replace("  ", " ")
    if p in _DIRECTION_WORDS:
        return _DIRECTION_WORDS[p]
    compact = p.replace(" ", "")
    return _DIRECTION_WORDS.get(compact)


def opposite(direction: str) -> Optional[str]:
    return OPPOSITES.get(direction)
