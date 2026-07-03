"""Fast-path player commands (no LLM)."""

from typing import List, Optional, Tuple, TYPE_CHECKING

from engine.blockers import (
    is_dark_without_light,
    item_requires_flag,
    movement_blocked,
)
from engine.bible_store import get_world_bible
from engine.item_matching import match_exit, match_item, split_item_list

if TYPE_CHECKING:
    from engine.state import StateManager


def post_turn_bookkeeping(
    state_mgr: "StateManager",
    fastpath_summary: Optional[str] = None,
) -> None:
    """Increment turns, advance chain steps, append fast-path history."""
    try:
        state_mgr.state.session_turns += 1
        if fastpath_summary:
            hist = state_mgr.state.recent_history
            hist.append({"role": "engine", "content": fastpath_summary})
            if len(hist) > 10:
                state_mgr.state.recent_history = hist[-10:]
            ctx = state_mgr.state.story_context or ""
            state_mgr.state.story_context = (ctx + " " + fastpath_summary).strip()[:2000]

        wb = get_world_bible()
        if not wb:
            return
        chain = wb.get("solution_chain") or []
        cur_loc = (state_mgr.state.location or "").lower()
        inventory_lower = {i.lower() for i in state_mgr.state.inventory}
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue
            sl = str(step.get("location") or "").strip()
            req = step.get("requires_item")
            holds = (
                not isinstance(req, str)
                or not req.strip()
                or req.strip().lower() in inventory_lower
            )
            if sl and cur_loc == sl.lower() and holds:
                if i not in state_mgr.state.chain_steps_completed:
                    state_mgr.state.chain_steps_completed.append(i)
    except Exception:
        pass


def handle_look_command(state_mgr: "StateManager") -> Optional[str]:
    state = state_mgr.state
    wb = get_world_bible()

    if is_dark_without_light(state, wb, state.location or ""):
        return "It is pitch dark. You cannot see anything."

    parts = []
    if wb:
        for loc in wb.get("locations", []):
            if isinstance(loc, dict) and loc.get("name", "").lower() == (state.location or "").lower():
                desc = loc.get("description", "")
                if desc:
                    parts.append(f"**{state.location}**: {desc}")
                break
    if not parts:
        parts.append(f"You are in **{state.location}**.")

    room_items = state.room_items.get(state.location, [])
    if room_items:
        parts.append(f"You see: {', '.join(room_items)}")

    exits = state.known_map.get(state.location, {}).get("exits", [])
    if exits:
        parts.append(f"Exits: {', '.join(exits)}")

    parts.append(f"Health: {state.health}")
    if state.inventory:
        parts.append(f"Carrying: {', '.join(state.inventory)}")
    else:
        parts.append("Carrying: nothing")
    return "\n".join(parts)


def try_local_command(
    state_mgr: "StateManager", user_text: str
) -> Optional[Tuple[str, List[str]]]:
    if not user_text:
        return None
    raw = user_text.strip()
    text = raw.lower()
    if not text:
        return None
    debug: List[str] = []
    state = state_mgr.state
    wb = get_world_bible()

    parts = text.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("help", "?"):
        debug.append("[fastpath] help")
        return (
            "**Engine commands (no LLM call, no token cost):**\n"
            "- `look` / `l` — describe your surroundings\n"
            "- `examine X` / `x X` — inspect an item, NPC, or monster\n"
            "- `inventory` / `inv` / `i` — list what you carry\n"
            "- `take X` / `get X` / `pick up X` — pick up an item from the room\n"
            "- `drop X` — set an item down in the room\n"
            "- `go X` / `n/s/e/w/up/down` — travel to an exit\n"
            "- `map` / `m` — show the known map\n"
            "- `wait` — pass time\n"
            "- anything else (combat, talk, riddles) goes to the storyteller LLM.",
            debug,
        )

    if cmd in ("inventory", "inv", "i"):
        debug.append("[fastpath] inventory")
        inv = state.inventory or []
        return (f"You are carrying: {', '.join(inv)}." if inv else "Your hands are empty."), debug

    if cmd in ("map", "m"):
        debug.append("[fastpath] map")
        return state_mgr.describe_map() or "You have not explored anywhere yet.", debug

    if cmd == "wait":
        debug.append("[fastpath] wait")
        return "You pause for a moment, listening to the silence.", debug

    if text in ("look", "l", "look around", "examine room"):
        debug.append("[fastpath] look")
        return (handle_look_command(state_mgr) or f"You stand in **{state.location}**."), debug

    examine_target: Optional[str] = None
    if cmd in ("examine", "x", "inspect"):
        examine_target = arg
    elif text.startswith("look at "):
        examine_target = text[len("look at ") :].strip()
    if examine_target is not None:
        if not examine_target:
            return "Examine what?", debug
        inv_match = match_item(examine_target, list(state.inventory))
        if inv_match:
            purpose = ""
            if wb:
                for ki in wb.get("key_items", []) or []:
                    if isinstance(ki, dict) and ki.get("name", "").lower() == inv_match.lower():
                        purpose = str(ki.get("purpose") or "")
                        break
            debug.append(f"[fastpath] examine item: {inv_match}")
            msg = f"You examine the **{inv_match}**."
            if purpose:
                msg += f" It seems useful for: {purpose[:120]}."
            return msg, debug
        room_items = list(state.room_items.get(state.location, []))
        room_match = match_item(examine_target, room_items)
        if room_match:
            debug.append(f"[fastpath] examine room item: {room_match}")
            return f"You look closely at **{room_match}** here in **{state.location}**.", debug
        if wb:
            for npc in wb.get("npcs", []) or []:
                if isinstance(npc, dict) and match_item(examine_target, [str(npc.get("name", ""))]):
                    debug.append(f"[fastpath] examine npc: {npc.get('name')}")
                    return (
                        f"**{npc.get('name')}** — {npc.get('personality', 'a mysterious figure')}.",
                        debug,
                    )
            for mon in wb.get("monsters", []) or []:
                if isinstance(mon, dict) and match_item(examine_target, [str(mon.get("name", ""))]):
                    debug.append(f"[fastpath] examine monster: {mon.get('name')}")
                    return (
                        f"**{mon.get('name')}** ({mon.get('difficulty', 'dangerous')}) "
                        f"looks formidable.",
                        debug,
                    )
        return None

    take_target: Optional[str] = None
    if cmd in ("take", "get", "grab"):
        take_target = arg
    elif text.startswith("pick up "):
        take_target = text[len("pick up ") :].strip()
    if take_target is not None:
        if not take_target:
            return "Take what?", debug
        room_items = list(state.room_items.get(state.location, []))
        whole_match = match_item(take_target, room_items)
        pieces = [take_target] if whole_match else split_item_list(take_target)
        taken: List[str] = []
        misses: List[str] = []
        for piece in pieces:
            match = match_item(piece, room_items)
            if match and match not in taken:
                if wb:
                    block = item_requires_flag(state, wb, match)
                    if block:
                        return block, debug
                state_mgr.remove_item_from_room(match)
                state_mgr.add_item(match)
                taken.append(match)
                debug.append(f"[fastpath] tool.room_take: {match}")
                room_items.remove(match)
            elif not match:
                misses.append(piece)
        if not taken and misses:
            still_here = state.room_items.get(state.location, [])
            return (
                f"You don't see '{', '.join(misses)}' here. "
                + (f"You see: {', '.join(still_here)}." if still_here else "The room is empty.")
            ), debug
        bits = []
        if len(taken) == 1:
            bits.append(f"You pick up the **{taken[0]}**.")
        elif taken:
            bits.append(f"You pick up: **{', '.join(taken)}**.")
        if misses:
            bits.append(f"(You don't see '{', '.join(misses)}' here.)")
        return " ".join(bits), debug

    drop_target: Optional[str] = None
    if cmd in ("drop", "leave"):
        drop_target = arg
    elif text.startswith("put down "):
        drop_target = text[len("put down ") :].strip()
    if drop_target is not None:
        if not drop_target:
            return "Drop what?", debug
        inventory = list(state.inventory)
        pieces = split_item_list(drop_target)
        dropped: List[str] = []
        misses: List[str] = []
        for piece in pieces:
            match = match_item(piece, inventory)
            if match and match not in dropped:
                state_mgr.remove_item(match)
                state_mgr.place_item_in_room(match)
                dropped.append(match)
                debug.append(f"[fastpath] tool.drop: {match}")
                inventory.remove(match)
            elif not match:
                misses.append(piece)
        if not dropped and misses:
            return (
                f"You aren't carrying '{', '.join(misses)}'. "
                + (f"You have: {', '.join(state.inventory)}." if state.inventory else "Your hands are empty.")
            ), debug
        bits = []
        if dropped:
            bits.append(f"You set down in **{state.location}**: **{', '.join(dropped)}**.")
        if misses:
            bits.append(f"(You weren't carrying '{', '.join(misses)}'.)")
        return " ".join(bits), debug

    exits = state.known_map.get(state.location, {}).get("exits", []) or []
    move_target: Optional[str] = None
    if cmd in ("go", "move", "enter", "head", "travel") and arg:
        move_target = match_exit(arg, exits)
    elif cmd in ("north", "south", "east", "west", "up", "down", "n", "s", "e", "w", "u", "d"):
        move_target = match_exit(cmd, exits)
    elif exits and text in {x.lower() for x in exits}:
        move_target = next(x for x in exits if x.lower() == text)

    if move_target:
        if wb:
            block = movement_blocked(state, wb, move_target)
            if block:
                debug.append(f"[fastpath] movement blocked: {move_target}")
                return block, debug
        state_mgr.move_to(move_target)
        debug.append(f"[fastpath] tool.move_to: {move_target}")
        room_desc = ""
        if wb:
            for loc in wb.get("locations", []) or []:
                if isinstance(loc, dict) and loc.get("name", "").lower() == move_target.lower():
                    room_desc = loc.get("description", "")
                    break
        room_items = state.room_items.get(move_target, [])
        new_exits = state.known_map.get(move_target, {}).get("exits", [])
        bits = [f"You enter **{move_target}**."]
        if room_desc and not is_dark_without_light(state, wb, move_target):
            bits.append(room_desc)
        elif is_dark_without_light(state, wb, move_target):
            bits.append("It is pitch dark here.")
        if room_items and not is_dark_without_light(state, wb, move_target):
            bits.append(f"You see: {', '.join(room_items)}.")
        if new_exits:
            bits.append(f"Exits: {', '.join(new_exits)}.")
        return " ".join(bits), debug

    return None
