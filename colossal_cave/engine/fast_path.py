"""Deterministic command fast-path — skip Ollama for common verbs."""

import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from engine.items import find_item_smart
from engine.llm_commands import COMMAND_SHORTCUTS, DETERMINISTIC_ACTIONS, apply_command_shortcuts
from engine.preprocess import preprocess_command


from engine.directions import (
    CARDINALS,
    INTERCARDINALS,
    _CARDINAL_ANGLES,
    layout_angle,
    normalize_direction_phrase,
)

# CHANGE (speed): category keywords for take/drop all <type>
_CATEGORY_ALIASES = {
    "weapon": "weapon", "weapons": "weapon", "sword": "weapon", "swords": "weapon",
    "armor": "armor", "armors": "armor", "armour": "armor",
    "magic": "magic", "magical": "magic", "potion": "magic", "potions": "magic",
    "treasure": "treasure", "treasures": "treasure", "gold": "treasure",
}

_ATTACK_PREFIXES = ("attack ", "fight ", "kill ", "hit ")


def normalize_direction(text: str) -> Optional[str]:
    """Map a direction token to canonical 8-way name."""
    return normalize_direction_phrase(text)


def _normalize_direction_phrase(phrase: str) -> Optional[str]:
    return normalize_direction_phrase(phrase)


def extract_direction_phrase(command: str) -> Optional[str]:
    """Pull direction text from go/walk/move or bare direction input."""
    cmd = command.lower().strip()
    cmd = apply_command_shortcuts(cmd)
    if cmd.startswith("go "):
        parts = cmd.split(None, 1)
        return parts[1].strip() if len(parts) == 2 else None
    for prefix in ("walk ", "move ", "travel ", "enter "):
        if cmd.startswith(prefix):
            return cmd[len(prefix):].strip()
    if _normalize_direction_phrase(cmd):
        return cmd.strip()
    return None


def is_movement_phrase(command: str) -> bool:
    """True if command looks like a movement direction."""
    phrase = extract_direction_phrase(command)
    return phrase is not None and _normalize_direction_phrase(phrase) is not None


def _exit_bearing(room: Any, neighbor: Any) -> float:
    """Layout angle from room to neighbor (screen coords: y down)."""
    dx = (getattr(neighbor, "map_x", 0) or 0) - (getattr(room, "map_x", 0) or 0)
    dy = (getattr(neighbor, "map_y", 0) or 0) - (getattr(room, "map_y", 0) or 0)
    return layout_angle(dx, dy)


def _nearest_exit_for_angle(room: Any, target_angle: float) -> Optional[str]:
    """Pick connected_rooms key whose tunnel bearing is closest to target_angle."""
    conns = getattr(room, "connected_rooms", {}) or {}
    best_dir = None
    best_diff = math.pi * 2
    for direction, nb in conns.items():
        if nb is None:
            continue
        angle = _exit_bearing(room, nb)
        diff = abs((angle - target_angle + math.pi) % (2 * math.pi) - math.pi)
        if diff < best_diff:
            best_diff = diff
            best_dir = direction
    return best_dir


def resolve_movement_direction(command: str, room: Any) -> Optional[str]:
    """
    Resolve movement to a connected_rooms key (8-way: N, NE, E, SE, S, SW, W, NW).
    Uses exact key when present; otherwise nearest exit by bearing.
    """
    phrase = extract_direction_phrase(command)
    if not phrase:
        return None
    normalized = normalize_direction_phrase(phrase)
    if not normalized:
        return None

    conns = getattr(room, "connected_rooms", {}) or {}
    if normalized in conns:
        return normalized

    target_angle = _CARDINAL_ANGLES.get(normalized)
    if target_angle is not None:
        nearest = _nearest_exit_for_angle(room, target_angle)
        if nearest:
            return nearest

    return normalized


def movement_monsters_in_direction(current_room: Any, direction: str) -> List[Any]:
    """Return live monsters in the destination room for a direction (debug/ambient use)."""
    if direction not in getattr(current_room, "connected_rooms", {}):
        return []
    dest = current_room.connected_rooms[direction]
    return list(getattr(dest, "monsters", None) or [])


def movement_blocked_by_monster(current_room: Any, direction: str) -> Optional[str]:
    """Legacy helper — reports monsters ahead without blocking (used in tests/logging)."""
    monsters = movement_monsters_in_direction(current_room, direction)
    if monsters:
        names = ", ".join(m.name for m in monsters)
        dest_num = current_room.connected_rooms[direction].number
        return f"Monsters in room {dest_num} ({direction}): {names}"
    return None


def parse_movement_command(command: str) -> Optional[str]:
    """Extract an 8-way direction from go/walk/move or bare direction input."""
    phrase = extract_direction_phrase(command)
    if not phrase:
        return None
    return normalize_direction_phrase(phrase)


def parse_category_keyword(text: str) -> Optional[str]:
    """Return category key (weapon/armor/magic/treasure) or None."""
    text = text.lower().strip()
    if text in ("all", "everything"):
        return None
    for word in text.split():
        if word in _CATEGORY_ALIASES:
            return _CATEGORY_ALIASES[word]
    return None


def _is_take_all_phrase(remainder: str) -> bool:
    """True for take all / take all the items / take everything in the room (not category)."""
    r = remainder.lower().strip()
    if r in ("all", "everything"):
        return True
    if not r.startswith("all "):
        return False
    tail = r[4:].strip()
    if tail in ("items", "the items", "item", "the item", "things", "the things", "stuff"):
        return True
    if tail.startswith("the ") and tail[4:].strip() in ("items", "item", "things", "stuff"):
        return True
    # "all the items in the room", "all items in this room", etc.
    if "in the room" in tail or "in this room" in tail:
        if any(w in tail for w in ("item", "things", "stuff", "everything")):
            return True
    return False


def _split_item_names(text: str) -> List[str]:
    """Split 'dirk, crown and flute' into separate item names."""
    parts = re.split(r",|\s+and\s+", text)
    return [p.strip() for p in parts if p.strip()]


def parse_take_items(command: str) -> Optional[Union[List[str], Tuple[str, str]]]:
    """
    Extract item name(s) from take/get/grab/collect/gather/pick up commands.
    Returns list of names, or ('category', 'weapon') for take all weapons.
    """
    cmd = command.lower().strip()
    for prefix in ("take ", "get ", "grab ", "pick up ", "collect ", "gather "):
        if cmd.startswith(prefix):
            remainder = cmd[len(prefix):].strip()
            if remainder.startswith("all ") or remainder == "all":
                cat = parse_category_keyword(remainder)
                if cat:
                    return ("category", cat)
                if _is_take_all_phrase(remainder):
                    return ["all"]
            if remainder in ("all", "everything"):
                return [remainder]
            # Multi-item: "get the dirk, crown and flute"
            names = _split_item_names(remainder)
            if len(names) > 1:
                return names
            return [remainder]
    return None


def parse_drop_items(command: str) -> Optional[Union[List[str], Tuple[str, str]]]:
    """Extract item name(s) from drop/leave/put down commands."""
    cmd = command.lower().strip()
    for prefix in ("drop ", "leave ", "put down "):
        if cmd.startswith(prefix):
            remainder = cmd[len(prefix):].strip()
            if remainder.startswith("all ") or remainder == "all":
                cat = parse_category_keyword(remainder)
                if cat:
                    return ("category", cat)
            if remainder in ("all", "everything"):
                return [remainder]
            return [remainder]
    return None


def parse_wear_armor(command: str) -> Optional[str]:
    """Extract armor name from wear/put on; empty string means wear any armor."""
    cmd = command.lower().strip()
    for prefix in ("wear ", "put on "):
        if cmd.startswith(prefix):
            name = cmd[len(prefix):].strip()
            # CHANGE (speed): generic "the armor" / "armor" → fuzzy pick from inventory
            if name in ("armor", "the armor", "an armor", "some armor", ""):
                return ""
            return name
    return None


def parse_remove_armor(command: str) -> Optional[str]:
    """Extract armor name from remove/take off."""
    cmd = command.lower().strip()
    for prefix in ("remove ", "take off "):
        if cmd.startswith(prefix):
            name = cmd[len(prefix):].strip()
            if name in ("armor", "the armor", ""):
                return ""
            return name
    return None


def parse_use_item(command: str) -> Optional[str]:
    """Extract item name from use command."""
    cmd = command.lower().strip()
    if cmd.startswith("use "):
        return cmd[4:].strip()
    return None


def parse_solve_items(command: str) -> Optional[Tuple[str, Optional[str]]]:
    """Extract magic item name(s) from solve commands."""
    cmd = command.lower().strip()
    if not cmd.startswith("solve"):
        return None
    rest = cmd[5:].strip()
    if rest in ("", "riddle", "the riddle", "puzzle", "the puzzle"):
        return None
    if rest.startswith("riddle with "):
        rest = rest[12:].strip()
    elif rest.startswith("the riddle with "):
        rest = rest[16:].strip()
    if " and " in rest:
        parts = rest.split(" and ", 1)
        return (parts[0].strip(), parts[1].strip())
    return (rest, None)


def parse_use_on_feature(command: str) -> Optional[Tuple[str, str]]:
    """Extract item and feature from 'use key on door'."""
    cmd = command.lower().strip()
    match = re.match(r"^use\s+(.+?)\s+on\s+(.+)$", cmd)
    if match:
        return (match.group(1).strip(), match.group(2).strip())
    return None


def parse_feature_verb(command: str) -> Optional[Tuple[str, Optional[str]]]:
    """Extract verb and optional target from play/listen/push/pull/light/open/close."""
    cmd = command.lower().strip()
    prefixes = (
        ("listen to ", "listen"),
        ("play ", "play"),
        ("push ", "push"),
        ("pull ", "pull"),
        ("light ", "light"),
        ("open ", "open"),
        ("close ", "close"),
    )
    for prefix, verb in prefixes:
        if cmd.startswith(prefix):
            target = cmd[len(prefix):].strip()
            return (verb, target or None)
    if cmd.startswith("listen "):
        return ("listen", cmd[7:].strip() or None)
    return None


def parse_answer(command: str) -> Optional[str]:
    """Extract typed answer from 'answer <text>'."""
    cmd = command.lower().strip()
    if cmd.startswith("answer "):
        return cmd[7:].strip()
    return None


def parse_attack_command(command: str) -> Optional[Tuple[str, Optional[str]]]:
    """Extract monster and optional weapon from attack/fight/kill/hit."""
    cmd = command.lower().strip()
    for prefix in _ATTACK_PREFIXES:
        if cmd.startswith(prefix):
            parts = re.split(r"\s+with\s+", cmd[len(prefix):], maxsplit=1)
            monster_name = parts[0].strip()
            weapon_name = parts[1].strip() if len(parts) > 1 else None
            if monster_name:
                return (monster_name, weapon_name)
    return None


def parse_cheat_command(command: str, magicword: str = "xyzzy") -> Optional[Tuple[str, Any]]:
    """
    Parse cheat commands. Returns (action, arg) or None.
    action: xyzzy | health | magicword
    """
    cmd = command.lower().strip()
    if cmd == "xyzzy" or cmd == magicword.lower():
        return ("xyzzy", None)
    health_match = re.match(r"health\s+(\d+)", cmd)
    if health_match:
        return ("health", int(health_match.group(1)))
    mw = magicword.lower()
    room_match = re.match(rf"(?:{re.escape(mw)}|magicword|magic word)\s+(\d+)", cmd)
    if room_match:
        return ("magicword", int(room_match.group(1)))
    return None


def bare_solve_magic_items(room: Any, player_items: List[Any]) -> Optional[Tuple[str, Optional[str]]]:
    """If inventory unambiguously matches riddle requirements, return (item1, item2)."""
    if not room or not getattr(room, "riddles", None) or getattr(room, "riddle_solved", False):
        return None
    required = list(getattr(room, "riddle_magic_items", []) or [])
    if not required:
        return None
    magic_names = [i.name for i in player_items if i.__class__.__name__ == "Magic"]
    if not all(item in magic_names for item in required):
        return None
    if len(required) == 1:
        return (required[0], None)
    return (required[0], required[1])


def resolve_armor_name(name: str, items: List[Any], worn: bool = False) -> Optional[str]:
    """Fuzzy-resolve armor name from inventory or worn list."""
    pool = items
    armor_items = [i for i in pool if i.__class__.__name__ == "Armor"]
    if not armor_items:
        return None
    if not name:
        return armor_items[0].name
    found = find_item_smart(name, armor_items)
    return found.name if found else None


def try_local_command(
    command: str,
    player: Any,
    room: Any,
    helpers: Dict[str, Callable],
    magicword: str = "xyzzy",
) -> Optional[Any]:
    """
    Handle common commands without Ollama.
    Returns action result, or None to fall through to llm_process_command.
    """
    if not command or not command.strip():
        return None

    cmd = preprocess_command(command)
    cmd_lower = cmd.lower().strip()
    cmd_expanded = apply_command_shortcuts(cmd_lower)

    # --- Cheat (xyzzy / health / magicword room) ---
    cheat = parse_cheat_command(cmd_lower, magicword=magicword)
    if cheat:
        action, arg = cheat
        if action == "xyzzy":
            fn = helpers.get("cheat_xyzzy")
            if fn:
                return fn()
        elif action == "health":
            fn = helpers.get("health_up")
            if fn:
                return fn(arg)
        elif action == "magicword":
            fn = helpers.get("magic_connections")
            if fn:
                return fn(arg)

    # --- Movement ---
    direction = resolve_movement_direction(cmd_expanded, room)
    if direction is not None:
        move_fn = helpers.get("move")
        if move_fn:
            return move_fn(direction)

    # --- Take (including category) ---
    take_parsed = parse_take_items(cmd_lower)
    if take_parsed:
        take_fn = helpers.get("take")
        if take_fn:
            if isinstance(take_parsed, tuple):
                return take_fn(take_parsed)
            return take_fn(take_parsed)

    # --- Drop / leave (including category and put down) ---
    drop_parsed = parse_drop_items(cmd_lower)
    if drop_parsed:
        leave_fn = helpers.get("leave")
        if leave_fn:
            if isinstance(drop_parsed, tuple):
                return leave_fn(drop_parsed)
            return leave_fn(drop_parsed)

    # --- Wear armor ---
    wear_name = parse_wear_armor(cmd_lower)
    if wear_name is not None:
        wear_fn = helpers.get("wear")
        if wear_fn:
            resolved = resolve_armor_name(wear_name, getattr(player, "items", []))
            if resolved is not None:
                return wear_fn(resolved)
            return "You don't have any armor to wear.\n"

    # --- Remove armor ---
    remove_name = parse_remove_armor(cmd_lower)
    if remove_name is not None:
        remove_fn = helpers.get("remove")
        if remove_fn:
            worn = getattr(player, "armors", [])
            resolved = resolve_armor_name(remove_name, worn, worn=True)
            if resolved is not None:
                return remove_fn(resolved)
            return "You don't have that armor on.\n"

    # --- Use item on room feature (tool use) ---
    use_on = parse_use_on_feature(cmd_lower)
    if use_on:
        feat_fn = helpers.get("feature_action")
        if feat_fn:
            result = feat_fn("use", use_on[1], use_on[0])
            if result:
                return result

    # --- Feature verbs: play, listen, push, etc. ---
    feat_verb = parse_feature_verb(cmd_lower)
    if feat_verb:
        action_fn = helpers.get("action_word")
        if action_fn:
            result = action_fn(feat_verb[0])
            if result:
                return result
        feat_fn = helpers.get("feature_action")
        if feat_fn:
            result = feat_fn(feat_verb[0], feat_verb[1], None)
            if result:
                return result

    # --- Typed answer puzzles ---
    answer_text = parse_answer(cmd_lower)
    if answer_text:
        answer_fn = helpers.get("answer_puzzle")
        if answer_fn:
            return answer_fn(answer_text)

    # --- Use item ---
    use_name = parse_use_item(cmd_lower)
    if use_name:
        use_fn = helpers.get("use")
        if use_fn:
            return use_fn(use_name)

    # --- Solve with named item(s) ---
    solve_named = parse_solve_items(cmd_lower)
    if solve_named:
        solve_fn = helpers.get("solve_puzzle")
        if solve_fn:
            return solve_fn(solve_named[0], solve_named[1])

    # --- Look / examine (no diffusion regen on fast-path) ---
    if cmd_lower in ("look", "look around", "l") or cmd_lower.startswith(
        ("examine ", "examin ", "study ", "look at ", "inspect ", "check ")
    ):
        study_fn = helpers.get("study")
        if study_fn:
            if cmd_lower in ("look", "look around", "l"):
                return study_fn("around", cmd, redraw=False)
            for prefix in ("look at ", "examine ", "examin ", "study ", "inspect ", "check "):
                if cmd_lower.startswith(prefix):
                    target = cmd_lower[len(prefix):].strip()
                    return study_fn(target, cmd, redraw=False)

    # --- Bare solve when match is unambiguous ---
    if cmd_lower in ("solve", "solve riddle", "solve the riddle"):
        solve_fn = helpers.get("solve_puzzle")
        match = bare_solve_magic_items(room, getattr(player, "items", []))
        if solve_fn and match:
            return solve_fn(match[0], match[1])

    # --- Deterministic info commands ---
    if cmd_expanded == "inventory" or cmd_lower in ("inv", "i", "inventory"):
        inv_fn = helpers.get("inventory")
        return inv_fn() if inv_fn else player.inventory_living()

    if cmd_lower in DETERMINISTIC_ACTIONS or cmd_expanded in DETERMINISTIC_ACTIONS:
        action = cmd_lower if cmd_lower in DETERMINISTIC_ACTIONS else cmd_expanded
        helper = helpers.get(action)
        if helper:
            return helper(command)

    # --- Attack (attack/fight/kill/hit) ---
    attack_parsed = parse_attack_command(cmd_lower)
    if attack_parsed:
        attack_fn = helpers.get("attack")
        if attack_fn:
            return attack_fn(attack_parsed[0], attack_parsed[1])

    return None
