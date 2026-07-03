"""Seed GameState from World Bible at adventure start."""

import re
from typing import List, Optional, TYPE_CHECKING

from engine.bible_store import get_world_bible

if TYPE_CHECKING:
    from engine.state import StateManager


def seed_game_state_from_world_bible(state_mgr: "StateManager") -> List[str]:
    debug: List[str] = []
    wb = get_world_bible()
    if not wb:
        return debug
    locations = wb.get("locations") or []
    if not locations:
        return debug
    room_names: List[str] = []
    for loc in locations:
        if isinstance(loc, dict) and loc.get("name"):
            room_names.append(str(loc["name"]).strip())
        elif isinstance(loc, str) and loc.strip():
            room_names.append(loc.strip())
    if not room_names:
        return debug

    for i, loc in enumerate(locations):
        if not isinstance(loc, dict):
            continue
        nm = str(loc.get("name") or "").strip()
        if not nm:
            continue
        desc = (loc.get("description") or "").strip()
        exits = loc.get("exits") if isinstance(loc.get("exits"), list) else []
        if nm not in state_mgr.state.known_map:
            state_mgr.state.known_map[nm] = {"exits": [], "notes": (desc[:1200] if desc else "")}
        else:
            if desc and not (state_mgr.state.known_map[nm].get("notes") or "").strip():
                state_mgr.state.known_map[nm]["notes"] = desc[:1200]
        for e in exits:
            if isinstance(e, str) and e.strip():
                state_mgr.connect_rooms(nm, e.strip())
        if not exits and i + 1 < len(locations):
            nxt = locations[i + 1]
            if isinstance(nxt, dict) and str(nxt.get("name") or "").strip():
                state_mgr.connect_rooms(nm, str(nxt["name"]).strip())

    start = room_names[0]
    state_mgr.move_to(start)
    debug.append(f"[seed] location={start}, rooms={len(room_names)}")

    il = wb.get("item_locations") or {}
    if isinstance(il, dict):
        for item_name, loc_desc in il.items():
            if not item_name or not isinstance(loc_desc, str):
                continue
            ld = loc_desc.lower().strip()
            placed_room: Optional[str] = None
            for rn in room_names:
                if rn.lower() in ld or ld in rn.lower():
                    placed_room = rn
                    break
            if placed_room is None and start.lower() in ld:
                placed_room = start
            if placed_room:
                state_mgr.place_item_in_room(str(item_name), room=placed_room)
                debug.append(f"[seed] {item_name} @ {placed_room}")
    return debug
