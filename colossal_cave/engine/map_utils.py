"""Map PNG generation — organic cave chambers and tunnel connections."""

import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw, ImageFont

from engine.graph_map import assign_organic_layout
from engine.directions import ABBREV as _DIR_ABBREV

# CHANGE (organic map): cave palette
_CAVE_BG = "#1a1510"
_TUNNEL_COLOR = "#6b5344"
_BUBBLE_BORDER = "#8b7355"
_BUBBLE_FILL = "#3d3428"
_CURRENT_RING = "#c9a227"
_LABEL_COLOR = "#f0e6d3"
_UNEXPLORED_DOT = "#4a4035"
_DIR_LABEL = "#d4c4a8"
_DIR_LABEL_UNEXPLORED = "#8a7a68"


def _room_coords(rooms: Dict[int, Any], number: int, room_size: int) -> Tuple[int, int]:
    """Pixel top-left for room chamber."""
    room = rooms[number]
    if getattr(room, "map_x", None) is not None and getattr(room, "map_y", None) is not None:
        min_x = min(getattr(r, "map_x", 0) for r in rooms.values())
        min_y = min(getattr(r, "map_y", 0) for r in rooms.values())
        return (
            (room.map_x - min_x) * room_size,
            (room.map_y - min_y) * room_size,
        )
    num_rooms_side = int(len(rooms) ** 0.5) or 1
    return (
        ((number - 1) % num_rooms_side) * room_size,
        ((number - 1) // num_rooms_side) * room_size,
    )


def _room_center(x: int, y: int, room_size: int, padding: int) -> Tuple[float, float]:
    return (x + room_size / 2 + padding, y + room_size / 2 + padding)


def _jagged_chamber_polygon(
    size: int, room_number: int, cx: float = None, cy: float = None, radius: float = None
) -> List[Tuple[float, float]]:
    """Jagged closed polygon — unique per room, clearly not a circle."""
    cx = cx if cx is not None else size / 2
    cy = cy if cy is not None else size / 2
    radius = radius if radius is not None else size * 0.44
    rng = random.Random(room_number * 7919 + 104729)
    n_vertices = rng.randint(8, 12)
    points = []
    for i in range(n_vertices):
        angle = (2 * math.pi * i / n_vertices) + rng.uniform(-0.55, 0.55)
        # Strong radius jitter — obvious flat sides and sharp corners
        r = radius * rng.uniform(0.52, 1.05)
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


def _jagged_chamber_mask(size: int, room_number: int) -> Image.Image:
    """Hard-edged jagged polygon mask (clip only — no blur)."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(_jagged_chamber_polygon(size, room_number), fill=255)
    return mask


def _fit_image_preserve_aspect(source: Image.Image, size: int) -> Image.Image:
    """Fit room art inside square canvas without stretching — letterbox if needed."""
    img = source.convert("RGBA") if source.mode != "RGBA" else source.copy()
    img.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ox = (size - img.width) // 2
    oy = (size - img.height) // 2
    canvas.paste(img, (ox, oy))
    return canvas


def _make_organic_thumbnail(source: Image.Image, size: int, room_number: int) -> Image.Image:
    """Clip room art into a jagged chamber — no distortion, only mask."""
    fitted = _fit_image_preserve_aspect(source, size)
    mask = _jagged_chamber_mask(size, room_number)
    chamber = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    chamber.paste(fitted, (0, 0), mask)
    return chamber


def _draw_jagged_outline(
    draw: ImageDraw.Draw,
    cx: float,
    cy: float,
    size: int,
    room_number: int,
    color: str,
    width: int = 2,
) -> None:
    """Draw jagged chamber border matching the clip polygon."""
    radius = size * 0.44
    points = _jagged_chamber_polygon(size, room_number, cx=cx, cy=cy, radius=radius)
    draw.polygon(points, outline=color, fill=None, width=width)


def _collect_edges(rooms: Dict[int, Any]) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for num, room in rooms.items():
        for nb in room.connected_rooms.values():
            if nb is None:
                continue
            edges.add(tuple(sorted((num, nb.number))))
    return edges


def draw_map_png(
    rooms: Dict[int, Any],
    cheat_mode: bool = False,
    output_path: str = "map.png",
    current_room_number: Optional[int] = None,
) -> Image.Image:
    """Draw explored room map as organic cave chambers with tunnel lines."""
    if rooms and getattr(next(iter(rooms.values())), "map_x", None) is None:
        assign_organic_layout(rooms)

    room_size = 50
    padding = 4
    chamber_size = room_size - 2

    max_x = max_y = 0
    for number in rooms:
        x, y = _room_coords(rooms, number, room_size)
        max_x = max(max_x, x + room_size)
        max_y = max(max_y, y + room_size)

    image_size = (max(max_x + padding * 2, room_size), max(max_y + padding * 2, room_size))
    image = Image.new("RGB", image_size, _CAVE_BG)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    centers: Dict[int, Tuple[float, float]] = {}
    for number in rooms:
        x, y = _room_coords(rooms, number, room_size)
        centers[number] = _room_center(x, y, room_size, padding)

    # Tunnels between connected rooms (center to center)
    for a, b in _collect_edges(rooms):
        ra, rb = rooms[a], rooms[b]
        touches_current = (
            current_room_number is not None
            and (a == current_room_number or b == current_room_number)
        )
        if not cheat_mode and not (ra.explored or rb.explored or touches_current):
            continue
        c1, c2 = centers[a], centers[b]
        draw.line([c1, c2], fill=_TUNNEL_COLOR, width=7)

    # Direction labels on tunnels from current room (all exits, even unexplored)
    if current_room_number and current_room_number in rooms:
        cur = rooms[current_room_number]
        for direction, nb in cur.connected_rooms.items():
            if nb is None or nb.number not in centers:
                continue
            c1, c2 = centers[current_room_number], centers[nb.number]
            mx, my = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
            abbrev = _DIR_ABBREV.get(direction, direction[:1].upper())
            label_color = _DIR_LABEL if (nb.explored or cheat_mode) else _DIR_LABEL_UNEXPLORED
            draw.text((mx - 4, my - 5), abbrev, fill=label_color, font=font)

    # Room chambers
    for number, room in rooms.items():
        x, y = _room_coords(rooms, number, room_size)
        x += padding
        y += padding
        cx, cy = centers[number]
        is_current = current_room_number is not None and number == current_room_number
        show = room.explored or cheat_mode or is_current

        if show:
            inset = (room_size - chamber_size) // 2
            bx, by = int(x + inset), int(y + inset)
            if room.image:
                chamber = _make_organic_thumbnail(room.image, chamber_size, number)
                image.paste(chamber, (bx, by), chamber)
            else:
                _draw_jagged_outline(draw, cx, cy, chamber_size, number, _BUBBLE_BORDER, width=2)
                draw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=_BUBBLE_FILL)
            tw, th = 8, 10
            draw.text((cx - tw / 2, cy - th / 2), str(number), fill=_LABEL_COLOR, font=font)
        elif cheat_mode:
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=_UNEXPLORED_DOT)

        if current_room_number and number == current_room_number and show:
            _draw_jagged_outline(
                draw, cx, cy, chamber_size + 4, number, _CURRENT_RING, width=3
            )

    image.save(output_path)
    return image
