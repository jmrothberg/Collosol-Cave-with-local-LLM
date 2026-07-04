"""Persistent per-game play debug log — movement, combat, routing, room state."""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

_debug_log_path: Optional[str] = None
_debug_game_name: str = ""
_turn_counter: int = 0
_command_meta: Dict[str, Any] = {}


def _debug_log_dir(game_dir: str) -> str:
    path = os.path.join(game_dir, "Debug_Logs")
    os.makedirs(path, exist_ok=True)
    return path


def start_debug_session(game_dir: str, game_name: str, resumed: bool = False) -> str:
    """Open/append a debug log file for one saved game."""
    global _debug_log_path, _debug_game_name, _turn_counter
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (game_name or "unknown"))[:80]
    _debug_log_path = os.path.join(_debug_log_dir(game_dir), f"{safe}_play_debug.log")
    _debug_game_name = game_name or safe
    _turn_counter = 0
    tag = "RESUMED" if resumed else "NEW"
    _append(
        f"\n{'=' * 72}\n"
        f"SESSION {tag} @ {datetime.now().isoformat(timespec='seconds')} game={_debug_game_name}\n"
        f"log_file={_debug_log_path}\n"
        f"{'=' * 72}\n"
    )
    return _debug_log_path


def get_debug_log_path() -> Optional[str]:
    return _debug_log_path


def set_command_meta(**kwargs: Any) -> None:
    """Record how the last command was routed (fast-path, llm, npc, etc.)."""
    global _command_meta
    _command_meta = dict(kwargs)


def consume_command_meta() -> Dict[str, Any]:
    """Read and clear routing metadata for the turn just processed."""
    global _command_meta
    meta = dict(_command_meta)
    _command_meta = {}
    return meta


def _append(text: str) -> None:
    if not _debug_log_path:
        return
    with open(_debug_log_path, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _room_snapshot(room: Any, label: str) -> str:
    if not room:
        return f"{label}: (none)"
    lines = [
        f"{label}: room={getattr(room, 'number', '?')} "
        f"explored={getattr(room, 'explored', '?')} "
        f"riddle_solved={getattr(room, 'riddle_solved', '?')}",
        f"  desc={getattr(room, 'description', '')[:80]!r}",
    ]
    monsters_here = getattr(room, "monsters", None) or []
    lines.append(
        "  monsters_here=["
        + (", ".join(m.name for m in monsters_here) if monsters_here else "none")
        + "]"
    )
    npc = getattr(room, "npc", None)
    lines.append(f"  npc={npc.name if npc else 'none'}")
    conns = getattr(room, "connected_rooms", {}) or {}
    for direction, neighbor in sorted(conns.items()):
        nb_monsters = getattr(neighbor, "monsters", None) or []
        mnames = ", ".join(m.name for m in nb_monsters) if nb_monsters else "none"
        lines.append(f"  exit {direction} -> room {neighbor.number} monsters=[{mnames}]")
    return "\n".join(lines)


def log_play_turn(
    command: str,
    preprocessed: str,
    path: str,
    room_before: Any,
    room_after: Any,
    player: Any,
    result: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one play turn with full room/movement context for debugging."""
    global _turn_counter
    if not _debug_log_path:
        return
    _turn_counter += 1
    if isinstance(result, tuple):
        result_text = str(result[0] or "")
    else:
        result_text = str(result or "")
    extra = extra or {}
    lines = [
        f"\n--- turn {_turn_counter} @ {datetime.now().isoformat(timespec='seconds')} ---",
        f"command_raw={command!r}",
        f"command_preprocessed={preprocessed!r}",
        f"route={path}",
        f"player health={int(getattr(player, 'health', 0))} "
        f"wealth={getattr(player, 'wealth', 0)} "
        f"items={len(getattr(player, 'items', []) or [])}",
        _room_snapshot(room_before, "room_before"),
        _room_snapshot(room_after, "room_after"),
        f"response={result_text[:2000]!r}",
    ]
    if extra:
        for key, value in extra.items():
            lines.append(f"{key}={value!r}")
    _append("\n".join(lines))


def log_game_event(event: str, detail: str = "") -> None:
    """Log load/save/errors outside normal play turns."""
    if not _debug_log_path:
        return
    _append(f"[{datetime.now().isoformat(timespec='seconds')}] {event} {detail}".strip())
