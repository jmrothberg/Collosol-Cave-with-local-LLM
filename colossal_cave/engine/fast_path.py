"""Deterministic command fast-path — skip Ollama for common verbs."""

import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.items import find_item_smart
from engine.llm_commands import COMMAND_SHORTCUTS, DETERMINISTIC_ACTIONS, apply_command_shortcuts
from engine.preprocess import preprocess_command


_DIRECTION_WORDS = {
    "north": "north", "south": "south", "east": "east", "west": "west",
    "n": "north", "s": "south", "e": "east", "w": "west",
    "up": "north", "down": "south",
}


def normalize_direction(text: str) -> Optional[str]:
    """Map a direction token to north/south/east/west."""
    token = text.lower().strip()
    return _DIRECTION_WORDS.get(token)


def movement_blocked_by_monster(current_room: Any, direction: str) -> Optional[str]:
    """Return a warning string if destination has live monsters, else None."""
    if direction not in getattr(current_room, "connected_rooms", {}):
        return None
    dest = current_room.connected_rooms[direction]
    monsters = getattr(dest, "monsters", None) or []
    if monsters:
        names = ", ".join(m.name for m in monsters)
        return (
            f"You hesitate — {names} lurk to the {direction}. "
            f"Try: attack {monsters[0].name}\n"
        )
    return None


def parse_movement_command(command: str) -> Optional[str]:
    """Extract a direction from go/walk/move or bare cardinal input."""
    cmd = command.lower().strip()
    cmd = apply_command_shortcuts(cmd)
    if cmd.startswith("go "):
        parts = cmd.split(None, 1)
        if len(parts) == 2:
            return normalize_direction(parts[1].strip())
    for prefix in ("walk ", "move ", "travel ", "enter "):
        if cmd.startswith(prefix):
            return normalize_direction(cmd[len(prefix):].strip())
    return normalize_direction(cmd)


def parse_take_items(command: str) -> Optional[List[str]]:
    """Extract item name(s) from take/get/grab/collect commands."""
    cmd = command.lower().strip()
    for prefix in ("take ", "get ", "grab ", "pick up ", "collect "):
        if cmd.startswith(prefix):
            return [cmd[len(prefix):].strip()]
    return None


def parse_drop_items(command: str) -> Optional[List[str]]:
    """Extract item name(s) from drop/leave commands."""
    cmd = command.lower().strip()
    for prefix in ("drop ", "leave "):
        if cmd.startswith(prefix):
            return [cmd[len(prefix):].strip()]
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


def try_local_command(
    command: str,
    player: Any,
    room: Any,
    helpers: Dict[str, Callable],
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

    # --- Movement ---
    direction = parse_movement_command(cmd_expanded)
    if direction is not None:
        block_msg = movement_blocked_by_monster(room, direction)
        if block_msg:
            return block_msg
        move_fn = helpers.get("move")
        if move_fn:
            return move_fn(direction)

    # --- Take ---
    take_names = parse_take_items(cmd_lower)
    if take_names:
        take_fn = helpers.get("take")
        if take_fn:
            return take_fn(take_names)

    # --- Drop / leave ---
    drop_names = parse_drop_items(cmd_lower)
    if drop_names:
        leave_fn = helpers.get("leave")
        if leave_fn:
            return leave_fn(drop_names)

    # --- Look / examine (no diffusion regen on fast-path) ---
    if cmd_lower in ("look", "look around", "l") or cmd_lower.startswith(("examine ", "study ", "look at ")):
        study_fn = helpers.get("study")
        if study_fn:
            if cmd_lower in ("look", "look around", "l"):
                return study_fn("around", cmd, redraw=False)
            target = cmd_lower.split(" ", 2)[-1] if cmd_lower.startswith("look at ") else cmd_lower.split(" ", 1)[-1]
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

    # --- Attack (simple form: attack <monster>) ---
    if cmd_lower.startswith("attack ") or cmd_lower.startswith("fight "):
        parts = re.split(r"\s+with\s+", cmd_lower, maxsplit=1)
        monster_part = parts[0].split(" ", 1)
        if len(monster_part) == 2:
            monster_name = monster_part[1].strip()
            weapon_name = parts[1].strip() if len(parts) > 1 else None
            attack_fn = helpers.get("attack")
            if attack_fn:
                return attack_fn(monster_name, weapon_name)

    return None
