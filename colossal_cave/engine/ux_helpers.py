"""Player-facing UX helpers — atmosphere, help, adventure log, status."""

from typing import Any, Dict, List, Optional


# CHANGE (UX): module-level adventure log buffer (reset on new/load game)
_adventure_log_lines: List[str] = []
_session_turns: int = 0


def reset_adventure_log() -> None:
    """Clear the chronological adventure log and turn counter."""
    global _adventure_log_lines, _session_turns
    _adventure_log_lines = []
    _session_turns = 0


def get_adventure_log_text() -> str:
    """Return the full adventure log as Markdown."""
    if not _adventure_log_lines:
        return "*Your adventure begins…*"
    return "\n\n".join(_adventure_log_lines)


def get_session_turns() -> int:
    """Return how many play turns have been logged."""
    return _session_turns


def append_adventure_log(player_command: str, response_text: str) -> None:
    """Append one turn to the adventure log (player input + game response)."""
    global _session_turns
    _session_turns += 1
    command_line = f"> {player_command.strip()}" if player_command.strip() else "> (turn)"
    body = (response_text or "").strip()
    if not body:
        body = "*(no response — try: look · inventory · go <direction>)*"
    _adventure_log_lines.append(f"{command_line}\n\n{body}")


def append_adventure_log_system(message: str) -> None:
    """Append a system/narrative line (ambient cue, riddle hint) without a player command."""
    if message and message.strip():
        _adventure_log_lines.append(message.strip())


def is_empty_game_response(result: Any) -> bool:
    """True when the game produced no usable text for the player."""
    if result is None:
        return True
    if isinstance(result, tuple):
        text = result[0] if result else ""
    else:
        text = str(result)
    cleaned = text.strip()
    return not cleaned or cleaned == "Error generating response"


def adjacent_cues(room: Any, difficulty: str = "easy") -> str:
    """One short ambient line from neighboring rooms (pure Python, no LLM)."""
    if difficulty == "hard":
        return ""
    if not room or not getattr(room, "connected_rooms", None):
        return ""

    lines = []
    for direction, neighbor in room.connected_rooms.items():
        if neighbor is None:
            continue
        if getattr(neighbor, "monsters", None):
            if difficulty == "cheat":
                names = ", ".join(m.name for m in neighbor.monsters)
                lines.append(f"*You sense {names} to the {direction}.*")
            else:
                lines.append(f"*A low growl drifts from the {direction}.*")
        elif getattr(neighbor, "riddles", None) and not getattr(neighbor, "riddle_solved", True):
            lines.append(f"*A faint chime hums to the {direction}.*")
        elif difficulty == "cheat":
            treasures = [i for i in getattr(neighbor, "items", []) if getattr(i, "type", "") == "treasure"
                         or i.__class__.__name__ == "Treasure"]
            if treasures:
                lines.append(f"*Something gleams to the {direction} ({treasures[0].name}).*")
            elif getattr(neighbor, "items", None):
                lines.append(f"*Something gleams from the {direction}.*")
        elif any(getattr(i, "type", "") == "treasure" or i.__class__.__name__ == "Treasure"
                 for i in getattr(neighbor, "items", [])):
            lines.append(f"*Something gleams from the {direction}.*")
        if getattr(neighbor, "npc", None) and not getattr(neighbor.npc, "is_master_beast", False):
            lines.append(f"*You hear voices to the {direction}.*")

    if not lines:
        return ""
    if difficulty == "medium":
        return lines[0]
    return " ".join(lines[:2])


def format_room_exits(room: Any, difficulty: str = "easy") -> str:
    """Clear exit list for the room you are standing in (north/east/south/west)."""
    if not room or not getattr(room, "connected_rooms", None):
        return ""
    parts = []
    for direction, neighbor in sorted(room.connected_rooms.items()):
        if neighbor is None:
            continue
        note = f"go {direction} (room {neighbor.number})"
        if getattr(neighbor, "monsters", None):
            if difficulty == "cheat":
                note += f" — {', '.join(m.name for m in neighbor.monsters)}"
            elif difficulty in ("easy", "medium"):
                note += " — monsters"
        elif getattr(neighbor, "riddles", None) and not getattr(neighbor, "riddle_solved", False):
            note += " — riddle"
        parts.append(note)
    if not parts:
        return ""
    return "Exits from here: " + "; ".join(parts) + "\n"


def exit_abbrev(room: Any) -> str:
    """Short compass summary for HUD, e.g. N·SE·W."""
    from engine.directions import ABBREV
    if not room or not getattr(room, "connected_rooms", None):
        return ""
    keys = [ABBREV.get(d, d[:2].upper()) for d in sorted(room.connected_rooms.keys())]
    return "·".join(keys) if keys else ""


def riddle_eligibility_text(room: Any, player_items: List[Any]) -> str:
    """Non-LLM riddle eligibility summary for auto-surface on room entry."""
    if not room or not getattr(room, "riddles", None) or getattr(room, "riddle_solved", False):
        return ""

    magic_names = [i.name for i in player_items if i.__class__.__name__ == "Magic"]
    required = list(getattr(room, "riddle_magic_items", []) or [])
    text = "\n💡 Riddle help: bring a magic item and type `solve <item name>`\n"
    if magic_names:
        text += f"📦 Your magic items: {', '.join(magic_names)}\n"
        if required and all(item in magic_names for item in required):
            solve_cmd = "solve " + " and ".join(required)
            text += f"✅ You have the right item(s)! Type: {solve_cmd}\n"
        else:
            text += "❌ You don't have the right magic item yet.\n"
    else:
        text += "⚠️ You need magic items to solve this riddle.\n"
    return text


def progress_summary(player: Any, rooms: Dict[int, Any]) -> str:
    """Short stuck-detector lines for situational help (no LLM)."""
    lines = []
    item_count = len(getattr(player, "items", []) or [])
    if item_count < 2:
        unexplored = [r for r in rooms.values() if not getattr(r, "explored", False)]
        if unexplored:
            lines.append(f"You haven't visited {len(unexplored)} rooms yet — keep exploring!")
    magic_count = len([i for i in getattr(player, "items", []) if i.__class__.__name__ == "Magic"])
    unsolved = [r for r in rooms.values() if getattr(r, "riddles", None) and not getattr(r, "riddle_solved", False)]
    if unsolved:
        lines.append(f"{len(unsolved)} riddles remain; you have {magic_count} magic item(s).")
    if 1 in rooms and getattr(rooms[1], "npc", None):
        lines.append("MB is in Room 1 — ask: MB what should I do?")
    return "\n".join(lines)


def build_status_header(
    player: Any,
    room: Any,
    ambient: str = "",
    difficulty: str = "easy",
) -> str:
    """One-line Markdown status for the HUD above the adventure log."""
    if difficulty == "hard":
        return f"❤️ {int(player.health)}/{getattr(player, 'max_health', 300)} · 📍 Room {room.number}"

    weapons = len([i for i in getattr(player, "items", []) if i.__class__.__name__ == "Weapon"])
    magic = len([i for i in getattr(player, "items", []) if i.__class__.__name__ == "Magic"])
    desc = getattr(room, "description", "")[:30]
    warn = ""
    if ambient and difficulty != "hard":
        warn = f" · ⚠️ {ambient[:40]}"
    exits = exit_abbrev(room)
    exit_line = f" · Exits: {exits}" if exits else ""
    return (
        f"❤️ {int(player.health)}/300 · 🪙 {getattr(player, 'wealth', 0)} · "
        f"📍 Room {room.number} ({desc}) · ⚔️ {weapons} · ✨ {magic}{warn}{exit_line} · "
        f"turn {get_session_turns()}"
    )


def format_session_postmortem(
    player: Any,
    rooms: Dict[int, Any],
    trophies: List[str],
    turns: int,
    outcome: str,
) -> str:
    """End-of-run stats block for win or death."""
    visited = len([r for r in rooms.values() if getattr(r, "explored", False)])
    total = len(rooms)
    riddles_solved = len([r for r in rooms.values() if getattr(r, "riddle_solved", False)])
    lines = [
        "\n" + "=" * 40,
        f"📊 ADVENTURE SUMMARY — {outcome}",
        f"Rooms explored: {visited}/{total}",
        f"Monsters killed: {getattr(player, 'kills', 0)}",
        f"Riddles solved: {riddles_solved}",
        f"Peak wealth: {getattr(player, 'wealth', 0)}",
        f"Trophies: {', '.join(trophies) if trophies else 'none'}",
        f"Turns played: {turns}",
        "=" * 40 + "\n",
    ]
    return "\n".join(lines)
