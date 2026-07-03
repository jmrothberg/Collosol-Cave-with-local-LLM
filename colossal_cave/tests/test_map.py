"""Tests for engine.map_utils.draw_map_png."""

import io

from PIL import Image

from engine.map_utils import draw_map_png


def test_draw_map_png_returns_image(three_room_world, tmp_path):
    out = tmp_path / "test_map.png"
    img = draw_map_png(three_room_world, cheat_mode=False, output_path=str(out))
    assert isinstance(img, Image.Image)
    assert out.exists()


def test_draw_map_png_valid_png_bytes(three_room_world, tmp_path):
    out = tmp_path / "test_map.png"
    draw_map_png(three_room_world, output_path=str(out))
    with open(out, "rb") as f:
        header = f.read(8)
    assert header[:4] == b"\x89PNG"


def test_cheat_mode_does_not_crash(three_room_world, tmp_path):
    out = tmp_path / "cheat_map.png"
    img = draw_map_png(three_room_world, cheat_mode=True, output_path=str(out))
    assert img.size[0] > 0
