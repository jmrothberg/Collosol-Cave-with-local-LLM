"""Shared pytest fixtures for colossal_cave tests."""

import os
import pickle
import sys
from types import SimpleNamespace

import pytest

# Ensure colossal_cave/ is on sys.path for `engine` imports
_COLossal_CAVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLossal_CAVE_DIR not in sys.path:
    sys.path.insert(0, _COLossal_CAVE_DIR)


class _StubItem:
    """Minimal item stand-in for find_item_smart tests."""

    def __init__(self, name: str):
        self.name = name


class _StubRoom:
    """Minimal room for map/trophy tests."""

    def __init__(self, number: int, explored: bool = False, image=None):
        self.number = number
        self.explored = explored
        self.image = image
        self.connected_rooms = {}
        self.items = []
        self.monsters = []
        self.riddles = []
        self.riddle_solved = False
        self.riddle_magic_items = []
        self.hints = []
        self.npc = None

    def add_connection(self, direction: str, other: "_StubRoom") -> None:
        self.connected_rooms[direction] = other


@pytest.fixture
def stub_item():
    return _StubItem


@pytest.fixture
def three_room_world():
    """Three connected rooms for map/save tests."""
    r1 = _StubRoom(1, explored=True)
    r2 = _StubRoom(2, explored=True)
    r3 = _StubRoom(3, explored=False)
    r1.add_connection("east", r2)
    r2.add_connection("west", r1)
    r2.add_connection("east", r3)
    r3.add_connection("west", r2)
    return {1: r1, 2: r2, 3: r3}


@pytest.fixture
def adventure_data_path():
    return os.path.join(_COLossal_CAVE_DIR, "adventure_dataRA.json")


@pytest.fixture
def tmp_save_dir(tmp_path):
    save_dir = tmp_path / "saves"
    save_dir.mkdir(parents=True, exist_ok=True)
    return str(save_dir)


@pytest.fixture
def minimal_player(three_room_world):
    """Player-like object for trophy/save round-trip tests."""
    rooms = three_room_world
    player = SimpleNamespace()
    player.health = 100
    player.wealth = 0
    player.points = 0
    player.kills = 0
    player.items = []
    player.current_room = rooms[1]
    player.image = None

    def take(name):
        player.items.append(SimpleNamespace(name=name))
        return f"Took {name}.\n", []

    def drawing_self(_action=""):
        pass

    player.take = take
    player.drawing_self = drawing_self
    rooms[1].add_treasure = lambda *a, **k: None
    return player
