"""Room graph helpers shared by validation and repair."""

from typing import Any, Dict, List, Set, Tuple


def build_room_graph(world_bible: Dict[str, Any]) -> Tuple[List[str], Dict[str, Set[str]], Set[str], List[str]]:
    """Build bidirectional adjacency from locations[].exits.

    Returns (loc_names, sym_graph, reachable_from_start, gaps).
    """
    gaps: List[str] = []
    locs = world_bible.get("locations") or []
    loc_names: List[str] = []
    graph: Dict[str, List[str]] = {}
    for loc in locs:
        if not isinstance(loc, dict):
            continue
        nm = str(loc.get("name") or "").strip()
        if not nm:
            continue
        loc_names.append(nm)
        graph[nm] = [str(e).strip() for e in (loc.get("exits") or []) if isinstance(e, str)]
    loc_set = set(loc_names)
    if not loc_set:
        return loc_names, {}, set(), ["no usable locations in world bible"]

    sym: Dict[str, Set[str]] = {n: set() for n in loc_set}
    for a, neigh in graph.items():
        for b in neigh:
            if b in loc_set:
                sym[a].add(b)
                sym[b].add(a)
            else:
                gaps.append(f"location '{a}' lists exit to unknown room '{b}'")

    start = loc_names[0]
    seen: Set[str] = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in sym.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    unreachable = sorted(loc_set - seen)
    if unreachable:
        gaps.append(f"unreachable rooms from start '{start}': {', '.join(unreachable)}")
    return loc_names, sym, seen, gaps


def collect_key_item_names(world_bible: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for ki in world_bible.get("key_items") or []:
        if isinstance(ki, dict):
            n = str(ki.get("name") or "").strip()
            if n:
                names.append(n)
    return names
