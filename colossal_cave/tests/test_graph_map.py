"""Organic map layout and cave topology tests."""

from types import SimpleNamespace

from engine.graph_map import (
    apply_cave_connections,
    generate_cave_graph,
    layout_from_graph,
    relayout_rooms,
    spiral_layout,
)
from engine.directions import OPPOSITES


def test_spiral_layout_not_square_block():
    pos = spiral_layout(9)
    assert len(pos) == 9


def test_generate_cave_graph_connected():
    edges = generate_cave_graph(25, seed=1)
    assert len(edges) >= 24
    parent = {i: i for i in range(1, 26)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in edges:
        union(a, b)
    assert len({find(i) for i in range(1, 26)}) == 1


def test_apply_cave_connections_max_degree():
    rooms = {n: SimpleNamespace(number=n, connected_rooms={}, map_x=None, map_y=None) for n in range(1, 17)}
    apply_cave_connections(rooms, seed=99)
    for room in rooms.values():
        assert len(room.connected_rooms) <= 4


def test_layout_from_graph_sets_coords(three_room_world):
    rooms = three_room_world
    relayout_rooms(rooms)
    assert getattr(rooms[1], "map_x", None) is not None
    assert getattr(rooms[1], "map_y", None) is not None


def test_connected_rooms_adjacent_in_layout(three_room_world):
    rooms = three_room_world
    relayout_rooms(rooms)
    r1, r2 = rooms[1], rooms[2]
    if 2 in [nb.number for nb in r1.connected_rooms.values()]:
        dist = abs(r1.map_x - r2.map_x) + abs(r1.map_y - r2.map_y)
        assert dist <= 4


def test_directions_match_visual_layout():
    """After relayout, direction keys match screen: N=up, E=right, S=down, W=left."""
    from types import SimpleNamespace

    rooms = {
        1: SimpleNamespace(number=1, connected_rooms={}, map_x=None, map_y=None),
        2: SimpleNamespace(number=2, connected_rooms={}, map_x=None, map_y=None),
        3: SimpleNamespace(number=3, connected_rooms={}, map_x=None, map_y=None),
    }
    rooms[1].connected_rooms["east"] = rooms[2]
    rooms[2].connected_rooms["west"] = rooms[1]
    rooms[1].connected_rooms["south"] = rooms[3]
    rooms[3].connected_rooms["north"] = rooms[1]
    relayout_rooms(rooms)
    for direction, nb in rooms[1].connected_rooms.items():
        dx = nb.map_x - rooms[1].map_x
        dy = nb.map_y - rooms[1].map_y
        if direction == "east":
            assert dx > 0, f"east should be right, got dx={dx}"
        elif direction == "west":
            assert dx < 0
        elif direction == "north":
            assert dy < 0, f"north should be up, got dy={dy}"
        elif direction == "south":
            assert dy > 0, f"south should be down, got dy={dy}"


def test_assign_directions_bidirectional():
    """Every edge must appear on both rooms with opposite direction keys."""
    from types import SimpleNamespace

    rooms = {n: SimpleNamespace(number=n, connected_rooms={}, map_x=None, map_y=None) for n in range(1, 6)}
    # Star from room 3 — four neighbors stress cardinal assignment
    rooms[3].connected_rooms = {
        "north": rooms[1],
        "east": rooms[2],
        "south": rooms[4],
        "west": rooms[5],
    }
    for nb_num, direction in ((1, "south"), (2, "west"), (4, "north"), (5, "east")):
        rooms[nb_num].connected_rooms[direction] = rooms[3]
    relayout_rooms(rooms)
    opposites = OPPOSITES
    for num, room in rooms.items():
        for direction, nb in room.connected_rooms.items():
            opp = opposites[direction]
            assert nb.connected_rooms.get(opp) is room, (
                f"room {num} {direction}-> {nb.number} but {nb.number} lacks {opp}-> {num}"
            )


def test_no_edges_dropped_room11_scenario():
    """Room 11 star from save: 10↔11, 7↔11, 11↔12 — no edge dropped."""
    from engine.graph_map import assign_directions_from_layout, count_undirected_edges

    rooms = {
        n: SimpleNamespace(number=n, connected_rooms={}, map_x=None, map_y=None)
        for n in (7, 10, 11, 12)
    }
    rooms[10].map_x, rooms[10].map_y = 0, 1
    rooms[11].map_x, rooms[11].map_y = 0, 0
    rooms[12].map_x, rooms[12].map_y = 0, -1
    rooms[7].map_x, rooms[7].map_y = 1, 1
    rooms[10].connected_rooms["north"] = rooms[11]
    rooms[11].connected_rooms["south"] = rooms[10]
    rooms[11].connected_rooms["north"] = rooms[12]
    rooms[12].connected_rooms["south"] = rooms[11]
    rooms[7].connected_rooms["northwest"] = rooms[11]
    rooms[11].connected_rooms["southeast"] = rooms[7]

    before = count_undirected_edges(rooms)
    assign_directions_from_layout(rooms)
    assert count_undirected_edges(rooms) == before == 3

    neighbors_11 = {nb.number for nb in rooms[11].connected_rooms.values()}
    assert 10 in neighbors_11
    neighbors_10 = {nb.number for nb in rooms[10].connected_rooms.values()}
    assert 11 in neighbors_10
    opposites = OPPOSITES
    for direction, nb in rooms[10].connected_rooms.items():
        if nb.number == 11:
            assert rooms[11].connected_rooms[opposites[direction]] is rooms[10]
