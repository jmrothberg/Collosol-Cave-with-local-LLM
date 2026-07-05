#!/usr/bin/env python3
"""
Preview organic bubble maps — saved game vs simulated new game.

Usage (from colossal_cave/):
  COLOSSAL_CAVE_SKIP_VECTOR_MODEL=1 python scripts/preview_organic_map.py

Outputs:
  map_preview_saved_bubbles.png   — your Load-Game save relayout
  map_preview_newgame_bubbles.png — simulated 25-room new cave topology
"""

import os
import pickle
import sys

# Mock diffusers before game import
from unittest.mock import MagicMock

if "diffusers" not in sys.modules:
    _mock = MagicMock()
    _mock.DiffusionPipeline = MagicMock
    _mock.FluxPipeline = MagicMock
    _mock.StableDiffusion3Pipeline = MagicMock
    _mock.ZImagePipeline = MagicMock
    sys.modules["diffusers"] = _mock

os.environ.setdefault("COLOSSAL_CAVE_SKIP_VECTOR_MODEL", "1")
os.environ.setdefault("COLOSSAL_CAVE_HEADLESS", "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import colossal_cave_July_5_26 as gm
from engine.graph_map import apply_cave_connections, relayout_rooms
from engine.map_utils import draw_map_png


    class _RemapUnpickler(pickle.Unpickler):
        """Saves from python colossal_cave_July_5_26.py store classes as __main__."""

        def find_class(self, module, name):
            if module in ("__main__", "Colossal_Cave_Aug_2_25"):
                return getattr(gm, name)
            return super().find_class(module, name)


def load_save(name="July_3_4_16_"):
    path = os.path.join(_ROOT, "Adventure_Game_Saved", f"Adv_{name}.pkl")
    with open(path, "rb") as f:
        loaded = _RemapUnpickler(f).load()
    if isinstance(loaded, dict):
        return loaded["rooms"], loaded["playerOne"]
    return loaded[0], loaded[1]


def preview_saved_game():
    rooms, player = load_save()
    relayout_rooms(rooms)
    out = os.path.join(_ROOT, "map_preview_saved_bubbles.png")
    draw_map_png(
        rooms,
        cheat_mode=True,
        output_path=out,
        current_room_number=player.current_room.number,
    )
    print(f"Saved game preview: {out} ({len(rooms)} rooms, player at {player.current_room.number})")


def preview_new_game(num_rooms=25, seed=42):
    """Simulate new-game cave topology with placeholder room images from a save if available."""
    try:
        src_rooms, _ = load_save()
        templates = list(src_rooms.values())
    except Exception:
        templates = []

    from types import SimpleNamespace

    rooms = {}
    for n in range(1, num_rooms + 1):
        r = SimpleNamespace(
            number=n,
            explored=n <= 5,
            connected_rooms={},
            map_x=None,
            map_y=None,
            image=templates[n % len(templates)].image if templates and templates[n % len(templates)].image else None,
        )
        rooms[n] = r

    apply_cave_connections(rooms, seed=seed)
    out = os.path.join(_ROOT, "map_preview_newgame_bubbles.png")
    draw_map_png(rooms, cheat_mode=True, output_path=out, current_room_number=1)
    edges = sum(len(r.connected_rooms) for r in rooms.values()) // 2
    print(f"New game preview: {out} ({num_rooms} rooms, {edges} tunnels, seed={seed})")


if __name__ == "__main__":
    preview_saved_game()
    preview_new_game()
