"""Shared pytest fixtures for colossal_cave tests."""

# CHANGE (tests): mock heavy diffusers import so game module loads in CI without GPU stack
import sys
from unittest.mock import MagicMock

if "diffusers" not in sys.modules:
    _mock_diff = MagicMock()
    _mock_diff.DiffusionPipeline = MagicMock
    _mock_diff.FluxPipeline = MagicMock
    _mock_diff.StableDiffusion3Pipeline = MagicMock
    _mock_diff.ZImagePipeline = MagicMock
    sys.modules["diffusers"] = _mock_diff

import os
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def pytest_configure(config):
    """Earliest hook — mock diffusers/ollama paths before any test module imports the game."""
    if "diffusers" not in sys.modules or not isinstance(sys.modules.get("diffusers"), MagicMock):
        _mock_diff = MagicMock()
        _mock_diff.DiffusionPipeline = MagicMock
        _mock_diff.FluxPipeline = MagicMock
        _mock_diff.StableDiffusion3Pipeline = MagicMock
        _mock_diff.ZImagePipeline = MagicMock
        sys.modules["diffusers"] = _mock_diff
    os.environ.setdefault("COLOSSAL_CAVE_SKIP_VECTOR_MODEL", "1")
    os.environ.setdefault("COLOSSAL_CAVE_HEADLESS", "1")

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
        self.description = f"Room {number}"

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


_MINIMAL_ROOM = {
    "description": "A test room.",
    "riddles": [],
    "hints": [],
    "magic_items": [],
    "action_words": [],
}

_RIDDLE_ROOM = {
    "description": "A riddle chamber.",
    "riddles": ["What has keys but no locks?"],
    "hints": ["Think music."],
    "magic_items": ["Piano Key"],
    "action_words": ["solve"],
}


@pytest.fixture(scope="module")
def game_module():
    """Import main game module and load adventure data."""
    import importlib
    mod_name = "Colossal_Cave_Aug_2_25"
    # CHANGE (tests): force clean import with diffusers mocked
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    importlib.import_module(mod_name)
    mod = sys.modules[mod_name]
    mod.load_adventure_data()
    return mod


@pytest.fixture
def test_room(game_module):
    return game_module.Room(_MINIMAL_ROOM.copy(), 99)


@pytest.fixture
def riddle_room(game_module):
    return game_module.Room(_RIDDLE_ROOM.copy(), 50)


@pytest.fixture
def test_player(game_module, test_room):
    """Player with equip_self skipped for deterministic tests."""
    with patch.object(game_module.Player, "equip_self"):
        player = game_module.Player(name="Hero", current_room=test_room)
    return player


@pytest.fixture
def minimal_game_world(game_module):
    """Three-room world with monster, riddle, and items for integration tests."""
    r1 = game_module.Room(_MINIMAL_ROOM.copy(), 1)
    r2 = game_module.Room(_MINIMAL_ROOM.copy(), 2)
    r3 = game_module.Room(_RIDDLE_ROOM.copy(), 3)
    r1.add_connection("east", r2)
    r2.add_connection("west", r1)
    r2.add_connection("east", r3)
    r3.add_connection("west", r2)

    sword = game_module.Weapon("Iron Sword", 20, runllm=False)
    r2.items.append(sword)

    goblin = game_module.Monster("Goblin", 50, 10, runllm_mon=False)
    r3.monsters.append(goblin)

    magic = game_module.Magic("Piano Key", 5, runllm=False)
    r1.items.append(magic)

    with patch.object(game_module.Player, "equip_self"):
        player = game_module.Player(name="Hero", current_room=r1)

    rooms = {1: r1, 2: r2, 3: r3}
    return {"rooms": rooms, "player": player, "r1": r1, "r2": r2, "r3": r3, "sword": sword, "goblin": goblin, "magic": magic}
