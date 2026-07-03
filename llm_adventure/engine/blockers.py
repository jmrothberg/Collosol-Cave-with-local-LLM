"""Fast-path blocker checks — respect world-bible gates without LLM calls."""

from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.state import GameState


def _defeated_flag(name: str) -> str:
    safe = name.lower().replace(" ", "_")
    return f"{safe}_defeated"


def _flag_is_true(state: "GameState", flag: str) -> bool:
    val = state.game_flags.get(flag)
    return val is True or val == "true" or val == 1


def is_dark_without_light(state: "GameState", wb: Optional[Dict[str, Any]], location: str) -> bool:
    """True when room requires light and player has no lit source."""
    if _flag_is_true(state, "lit") or _flag_is_true(state, "lantern_lit"):
        return False
    loc_l = (location or "").lower()
    if wb:
        for loc in wb.get("locations") or []:
            if not isinstance(loc, dict):
                continue
            if str(loc.get("name") or "").lower() != loc_l:
                continue
            desc = str(loc.get("description") or "").lower()
            if "dark" in desc or "pitch black" in desc or loc.get("requires_light"):
                return True
        for mech in wb.get("mechanics") or []:
            if not isinstance(mech, dict):
                continue
            eff = str(mech.get("effect") or "").lower()
            act = str(mech.get("action") or "").lower()
            mloc = str(mech.get("location") or "").lower()
            if mloc == loc_l and ("dark" in eff or "light" in act):
                if not _flag_is_true(state, "lit"):
                    return True
    if _flag_is_true(state, "dark") and not _flag_is_true(state, "lit"):
        return True
    return False


def monster_blocks_room(state: "GameState", wb: Dict[str, Any], target_room: str) -> Optional[str]:
    """Return in-character block message if an undefeated monster guards target_room."""
    target_l = target_room.lower()
    for m in wb.get("monsters") or []:
        if not isinstance(m, dict):
            continue
        mloc = str(m.get("location") or "").strip()
        if mloc.lower() != target_l:
            continue
        mname = str(m.get("name") or "something dangerous")
        if _flag_is_true(state, _defeated_flag(mname)):
            continue
        diff = m.get("difficulty", "dangerous")
        return (
            f"You cannot enter **{target_room}** yet — **{mname}** ({diff}) blocks the way. "
            f"Try using the storyteller for combat or clever tricks."
        )
    return None


def chain_blocker_at_room(state: "GameState", wb: Dict[str, Any], target_room: str) -> Optional[str]:
    """Block movement if solution_chain says a blocker at target is not cleared."""
    target_l = target_room.lower()
    inv_lower = {i.lower() for i in state.inventory}
    for step in wb.get("solution_chain") or []:
        if not isinstance(step, dict):
            continue
        sl = str(step.get("location") or "").strip()
        if sl.lower() != target_l:
            continue
        req = step.get("requires_item")
        if isinstance(req, str) and req.strip() and req.strip().lower() not in inv_lower:
            blk = str(step.get("blocker") or "something blocks the way")
            return (
                f"You cannot enter **{target_room}** yet — {blk}. "
                f"You may need **{req}** first (ask the storyteller if stuck)."
            )
        blk = str(step.get("blocker") or "").strip()
        if blk:
            blk_flag = blk.lower().replace(" ", "_") + "_cleared"
            if not _flag_is_true(state, blk_flag) and not _flag_is_true(state, "lit"):
                # Only block on named blockers when no generic lit flag helps
                for token in ("locked", "sealed", "guarded", "riddle", "dark"):
                    if token in blk.lower():
                        return (
                            f"You cannot enter **{target_room}** yet — {blk}. "
                            f"Describe what you do to the storyteller."
                        )
    return None


def item_requires_flag(state: "GameState", wb: Dict[str, Any], item_name: str) -> Optional[str]:
    """Return message if item cannot be taken until a game_flag is set."""
    for mech in wb.get("mechanics") or []:
        if not isinstance(mech, dict):
            continue
        act = str(mech.get("action") or "").lower()
        if item_name.lower() not in act:
            continue
        eff = str(mech.get("effect") or "").lower()
        if "requires" in eff or "locked" in eff or "opened" in eff:
            for flag in state.game_flags:
                if flag.endswith("_opened") or flag.endswith("_unlocked"):
                    if _flag_is_true(state, flag):
                        return None
            return (
                f"**{item_name}** cannot be taken yet — "
                f"{mech.get('action', 'something')} must happen first. Tell the storyteller what you try."
            )
    il = wb.get("item_locations") or {}
    desc = str(il.get(item_name, ""))
    if "locked" in desc.lower() or "guarded" in desc.lower():
        if not any(_flag_is_true(state, f) for f in state.game_flags if "open" in f or "unlock" in f):
            return (
                f"**{item_name}** is not available to take yet. "
                f"Interact with it through the storyteller first."
            )
    return None


def movement_blocked(state: "GameState", wb: Optional[Dict[str, Any]], target_room: str) -> Optional[str]:
    """Combined movement gate check. None = allowed."""
    if not wb:
        return None
    msg = monster_blocks_room(state, wb, target_room)
    if msg:
        return msg
    msg = chain_blocker_at_room(state, wb, target_room)
    if msg:
        return msg
    return None
