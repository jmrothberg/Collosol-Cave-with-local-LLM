"""GameSession — holds runtime state instead of scattered globals."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GameSession:
    """Central game state container."""

    rooms: Dict[int, Any] = field(default_factory=dict)
    player: Any = None
    old_room: Optional[int] = None
    cheat_mode: bool = False
    magicword: str = ""
    trophies: List[str] = field(default_factory=list)
    active_game: bool = False
    art_style_g: str = ""
    difficulty_g: str = "Easy"

    # Content catalogs (populated from adventure_dataRA.json)
    room_descs_riddles: List[Any] = field(default_factory=list)
    monsters: List[str] = field(default_factory=list)
    weapons: List[str] = field(default_factory=list)
    armors: List[str] = field(default_factory=list)
    treasures: List[str] = field(default_factory=list)
    magic_items: List[str] = field(default_factory=list)
    npcs: List[str] = field(default_factory=list)
    npc_descs: List[str] = field(default_factory=list)


# Module-level session used during migration from globals
_current_session: Optional[GameSession] = None


def get_session() -> GameSession:
    """Return the active session, creating one if needed."""
    global _current_session
    if _current_session is None:
        _current_session = GameSession()
    return _current_session


def set_session(session: GameSession) -> None:
    """Replace the active session."""
    global _current_session
    _current_session = session


def sync_session_from_globals(
    rooms, player_one, old_room, cheat_mode, magicword, trophies, active_game, art_style_g,
) -> GameSession:
    """Build/update session from legacy module globals."""
    session = get_session()
    session.rooms = rooms
    session.player = player_one
    session.old_room = old_room
    session.cheat_mode = cheat_mode
    session.magicword = magicword
    session.trophies = trophies
    session.active_game = active_game
    session.art_style_g = art_style_g
    return session


def sync_globals_from_session(session: GameSession):
    """Return tuple of values to assign back to legacy globals."""
    return (
        session.rooms,
        session.player,
        session.old_room,
        session.cheat_mode,
        session.magicword,
        session.trophies,
        session.active_game,
        session.art_style_g,
    )
