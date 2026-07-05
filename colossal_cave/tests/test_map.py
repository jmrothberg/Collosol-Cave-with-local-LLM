"""Tests for engine.map_utils.draw_map_png."""

import io

from PIL import Image

from engine.map_utils import draw_map_png


def test_draw_map_png_returns_image(three_room_world, tmp_path):
    from engine.graph_map import relayout_rooms
    relayout_rooms(three_room_world)
    out = tmp_path / "test_map.png"
    img = draw_map_png(three_room_world, cheat_mode=False, output_path=str(out))
    assert isinstance(img, Image.Image)
    assert out.exists()


def test_draw_map_cave_background(three_room_world, tmp_path):
    from engine.graph_map import relayout_rooms
    from engine.map_utils import _fit_image_preserve_aspect, _jagged_chamber_polygon

    relayout_rooms(three_room_world)
    three_room_world[1].explored = True
    # Jagged polygons have 7-11 vertices, not smooth circles
    pts = _jagged_chamber_polygon(52, 1)
    assert len(pts) >= 7
    out = tmp_path / "cave_map.png"
    img = draw_map_png(three_room_world, cheat_mode=False, output_path=str(out))
    pixel = img.getpixel((5, 5))
    assert sum(pixel) < 80

    # Fit preserves aspect — wide image stays wide inside canvas
    wide = Image.new("RGB", (200, 100), color=(255, 0, 0))
    fitted = _fit_image_preserve_aspect(wide, 50)
    assert fitted.size == (50, 50)
    # Red content should not fill entire height (letterboxed)
    assert fitted.getpixel((25, 2))[3] == 0 or fitted.getpixel((25, 2))[0] == 0


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
