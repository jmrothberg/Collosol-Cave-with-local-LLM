"""Organic cave map — graph topology, layout, and direction assignment."""

import math
import random
from collections import deque
from typing import Any, Dict, List, Set, Tuple

from engine.directions import (
    OPPOSITES,
    _EIGHT_WAY,
    edge_ambiguity,
    ranked_directions,
    snap_direction,
)


def generate_cave_graph(num_rooms: int, seed: int = None) -> List[Tuple[int, int]]:
    """
    Build organic cave edges: spanning tree from room 1 + optional loop edges.
    Returns list of undirected edges (a, b) with a < b.
    """
    if num_rooms <= 1:
        return []
    rng = random.Random(seed)
    rooms = list(range(1, num_rooms + 1))
    in_tree: Set[int] = {1}
    edges: Set[Tuple[int, int]] = set()

    # Random spanning tree — ensures connectivity with branches/dead ends
    while len(in_tree) < num_rooms:
        outside = [r for r in rooms if r not in in_tree]
        parent = rng.choice(list(in_tree))
        child = rng.choice(outside)
        edges.add(tuple(sorted((parent, child))))
        in_tree.add(child)

    # Add loop passages (~20% of tree size), cap degree at 4
    degree: Dict[int, int] = {r: 0 for r in rooms}
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    extra_attempts = max(1, len(edges) // 5)
    for _ in range(extra_attempts):
        a, b = sorted(rng.sample(rooms, 2))
        if a == b or (a, b) in edges:
            continue
        if degree[a] >= 4 or degree[b] >= 4:
            continue
        edges.add((a, b))
        degree[a] += 1
        degree[b] += 1

    return sorted(edges)


def layout_from_graph(rooms: Dict[int, Any], spacing: float = 0.85) -> None:
    """Place rooms on map_x/map_y by BFS from room 1 — tight spacing."""
    if not rooms:
        return
    start = min(rooms.keys())
    positions: Dict[int, Tuple[int, int]] = {start: (0, 0)}
    queue = deque([start])
    visited = {start}
    angle_step = 72

    while queue:
        cur = queue.popleft()
        cx, cy = positions[cur]
        neighbors = list(rooms[cur].connected_rooms.items())
        slot = 0
        for _direction, nb in neighbors:
            if nb is None:
                continue
            nnum = nb.number
            if nnum in visited:
                continue
            angle = math.radians(slot * angle_step - 90)
            slot += 1
            nx = cx + round(math.cos(angle) * spacing)
            ny = cy + round(math.sin(angle) * spacing)
            # Nudge away from occupied cells
            attempts = 0
            while (nx, ny) in positions.values() and attempts < 8:
                angle += math.radians(30)
                nx = cx + round(math.cos(angle) * spacing)
                ny = cy + round(math.sin(angle) * spacing)
                attempts += 1
            positions[nnum] = (nx, ny)
            visited.add(nnum)
            queue.append(nnum)

    for num, room in rooms.items():
        if num not in positions:
            positions[num] = (num % 6, num // 6)
        room.map_x, room.map_y = positions[num]


def _snap_direction(dx: int, dy: int) -> str:
    """Pick 8-way direction closest to layout delta."""
    return snap_direction(dx, dy)


def _connect_rooms(ra: Any, rb: Any, direction: str) -> None:
    """Bidirectional edge without requiring Room.add_connection."""
    ra.connected_rooms[direction] = rb
    opp = OPPOSITES.get(direction, "west")
    if opp not in rb.connected_rooms:
        rb.connected_rooms[opp] = ra


def _collect_undirected_edges(rooms: Dict[int, Any]) -> Set[Tuple[int, int]]:
    """All unique room-to-room edges from connected_rooms."""
    edges: Set[Tuple[int, int]] = set()
    for num, room in rooms.items():
        for nb in room.connected_rooms.values():
            if nb is None:
                continue
            edges.add(tuple(sorted((num, nb.number))))
    return edges


def count_undirected_edges(rooms: Dict[int, Any]) -> int:
    return len(_collect_undirected_edges(rooms))


def directions_are_bidirectional(rooms: Dict[int, Any]) -> bool:
    """True when every exit has a matching opposite on the neighbor room."""
    for num, room in rooms.items():
        for direction, nb in room.connected_rooms.items():
            if nb is None:
                continue
            if str(direction).startswith("_"):
                continue
            opp = OPPOSITES.get(direction)
            if not opp or nb.connected_rooms.get(opp) is not room:
                return False
    return True


def _edge_ambiguity(dx: int, dy: int) -> float:
    return edge_ambiguity(dx, dy)


def _ranked_directions(dx: int, dy: int) -> List[str]:
    return ranked_directions(dx, dy)


def _assign_edge_directions(ra: Any, rb: Any) -> bool:
    """Assign reciprocal 8-way direction for one edge. Returns True if assigned."""
    dx = (rb.map_x or 0) - (ra.map_x or 0)
    dy = (rb.map_y or 0) - (ra.map_y or 0)
    used_a = set(ra.connected_rooms.keys())
    used_b = set(rb.connected_rooms.keys())
    for dir_ab in _ranked_directions(dx, dy):
        dir_ba = OPPOSITES[dir_ab]
        if dir_ab not in used_a and dir_ba not in used_b:
            ra.connected_rooms[dir_ab] = rb
            rb.connected_rooms[dir_ba] = ra
            return True
    for dir_ab, _ in _EIGHT_WAY:
        dir_ba = OPPOSITES[dir_ab]
        if dir_ab not in used_a and dir_ba not in used_b:
            ra.connected_rooms[dir_ab] = rb
            rb.connected_rooms[dir_ba] = ra
            return True
    return False


def assign_directions_from_layout(rooms: Dict[int, Any]) -> None:
    """Set connected_rooms direction keys from map positions (after layout)."""
    edges = _collect_undirected_edges(rooms)
    edge_count = len(edges)
    if not edge_count:
        return

    # Sort least-ambiguous edges first so clear N/S/E/W pairs claim slots before diagonals
    edge_order: List[Tuple[float, int, int]] = []
    for a, b in edges:
        ra, rb = rooms[a], rooms[b]
        dx = (rb.map_x or 0) - (ra.map_x or 0)
        dy = (rb.map_y or 0) - (ra.map_y or 0)
        edge_order.append((_edge_ambiguity(dx, dy), a, b))
    edge_order.sort(key=lambda t: (t[0], t[1], t[2]))

    for room in rooms.values():
        room.connected_rooms = {}

    for _ambiguity, a, b in edge_order:
        ra, rb = rooms[a], rooms[b]
        if not _assign_edge_directions(ra, rb):
            # Last resort: force any free reciprocal pair (never drop topology)
            used_a = set(ra.connected_rooms.keys())
            used_b = set(rb.connected_rooms.keys())
            for dir_ab, _ in _EIGHT_WAY:
                dir_ba = OPPOSITES[dir_ab]
                if dir_ab not in used_a and dir_ba not in used_b:
                    ra.connected_rooms[dir_ab] = rb
                    rb.connected_rooms[dir_ba] = ra
                    break

    assert count_undirected_edges(rooms) == edge_count, (
        f"edge drop during assign_directions: had {edge_count}, now {count_undirected_edges(rooms)}"
    )


def apply_cave_connections(rooms: Dict[int, Any], seed: int = None) -> None:
    """Generate cave graph edges, layout rooms, assign cardinal directions."""
    num = len(rooms)
    edges = generate_cave_graph(num, seed=seed)
    for room in rooms.values():
        room.connected_rooms = {}
    # Wire undirected edges for layout (temporary keys)
    for a, b in edges:
        rooms[a].connected_rooms[f"_e{b}"] = rooms[b]
        rooms[b].connected_rooms[f"_e{a}"] = rooms[a]
    layout_from_graph(rooms)
    for room in rooms.values():
        room.connected_rooms = {}
    for a, b in edges:
        rooms[a].connected_rooms[f"_e{b}"] = rooms[b]
        rooms[b].connected_rooms[f"_e{a}"] = rooms[a]
    assign_directions_from_layout(rooms)


def relayout_rooms(rooms: Dict[int, Any], preserve_coords: bool = False) -> None:
    """Recompute map positions and sync direction keys to match visual layout."""
    sanitize_connected_rooms(rooms)
    has_coords = bool(rooms) and all(
        getattr(r, "map_x", None) is not None and getattr(r, "map_y", None) is not None
        for r in rooms.values()
    )
    if preserve_coords and has_coords:
        # CHANGE (load): keep positions, relabel to 8-way from geometry
        assign_directions_from_layout(rooms)
        sanitize_connected_rooms(rooms)
        return
    if preserve_coords and directions_are_bidirectional(rooms):
        # CHANGE (load): place rooms for map, then relabel keys to match screen N/E/S/W
        layout_from_graph(rooms)
        assign_directions_from_layout(rooms)
        sanitize_connected_rooms(rooms)
        return
    layout_from_graph(rooms)
    assign_directions_from_layout(rooms)
    sanitize_connected_rooms(rooms)


def safe_connected_rooms(room: Any) -> Dict[str, Any]:
    """Return room.connected_rooms as a dict — never None (prevents .items crashes)."""
    conns = getattr(room, "connected_rooms", None)
    return conns if isinstance(conns, dict) else {}


def sanitize_connected_rooms(rooms: Dict[int, Any]) -> None:
    """Drop None neighbors from connected_rooms (prevents .number / .items crashes)."""
    for room in rooms.values():
        conns = getattr(room, "connected_rooms", None)
        if not isinstance(conns, dict):
            room.connected_rooms = {}
            continue
        room.connected_rooms = {d: nb for d, nb in conns.items() if nb is not None}


def spiral_layout(num_rooms: int) -> Dict[int, Tuple[int, int]]:
    """Legacy fallback — outward spiral coords."""
    positions = {}
    if num_rooms <= 0:
        return positions
    x, y = 0, 0
    dx, dy = 1, 0
    segment_len = 1
    seg_progress = 0
    seg_count = 0
    for n in range(1, num_rooms + 1):
        positions[n] = (x, y)
        x += dx
        y += dy
        seg_progress += 1
        if seg_progress == segment_len:
            seg_progress = 0
            seg_count += 1
            dx, dy = -dy, dx
            if seg_count % 2 == 0:
                segment_len += 1
    return positions


def assign_organic_layout(rooms: Dict[int, Any]) -> None:
    """Set map coords — prefer graph layout when connections exist."""
    if not rooms:
        return
    has_edges = any(r.connected_rooms for r in rooms.values())
    if has_edges:
        relayout_rooms(rooms)
    else:
        positions = spiral_layout(len(rooms))
        for number, room in rooms.items():
            if number in positions:
                room.map_x, room.map_y = positions[number]
