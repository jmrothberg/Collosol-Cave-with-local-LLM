"""Win condition normalization and runtime win checks."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.state import StateManager


def parse_win_condition(world_bible: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize world_bible['win_condition'] into structured form (in place) and return it."""
    wc = world_bible.get("win_condition")

    if isinstance(wc, dict):
        items = wc.get("required_items") or []
        if isinstance(items, str):
            items = [items]
        items = [str(x).strip() for x in items if str(x).strip()]
        loc = wc.get("required_location")
        loc = str(loc).strip() if loc else None
        desc = str(wc.get("description") or "").strip()
        if not desc:
            bits: List[str] = []
            if items:
                bits.append(f"have {', '.join(items)}")
            if loc:
                bits.append(f"reach {loc}")
            desc = "Win when you " + " and ".join(bits) + "." if bits else "Complete the adventure."
        normalized = {"required_items": items, "required_location": loc, "description": desc}
        world_bible["win_condition"] = normalized
        return normalized

    raw = str(wc or "").strip()
    items: List[str] = []
    loc: Optional[str] = None
    if raw:
        wc_lower = raw.lower()
        for ki in world_bible.get("key_items", []) or []:
            if not isinstance(ki, dict):
                continue
            name = str(ki.get("name") or "").strip()
            if name and name.lower() in wc_lower:
                items.append(name)
        best_len = 0
        for entry in world_bible.get("locations", []) or []:
            if not isinstance(entry, dict):
                continue
            nm = str(entry.get("name") or "").strip()
            if nm and nm.lower() in wc_lower and len(nm) > best_len:
                loc = nm
                best_len = len(nm)
    normalized = {
        "required_items": items,
        "required_location": loc,
        "description": raw or "Complete the adventure.",
    }
    world_bible["win_condition"] = normalized
    return normalized


def check_win_condition(state_mgr: "StateManager") -> Optional[str]:
    """Return victory message when win condition met, else None."""
    from engine.bible_store import get_world_bible

    wb = get_world_bible()
    if not wb:
        return None

    state = state_mgr.state

    if state.game_flags.get("game_complete") or state.game_flags.get("game_won"):
        struct = parse_win_condition(wb)
        return (
            f"**VICTORY!** {struct.get('description') or 'You have completed the adventure!'}\n"
            f"Congratulations, {state.player_name or 'adventurer'}!"
        )

    struct = parse_win_condition(wb)
    items_needed = struct.get("required_items") or []
    loc_needed = struct.get("required_location")
    if not items_needed and not loc_needed:
        return None

    items_ok = all(item in state.inventory for item in items_needed) if items_needed else True
    loc_ok = (loc_needed is None) or (
        bool(state.location) and state.location.strip().lower() == loc_needed.strip().lower()
    )
    if items_ok and loc_ok:
        return (
            f"**VICTORY!** {struct.get('description') or 'You have completed the adventure!'}\n"
            f"Congratulations, {state.player_name or 'adventurer'}!"
        )
    return None
