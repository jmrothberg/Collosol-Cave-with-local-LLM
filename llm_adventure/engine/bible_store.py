"""Global World Bible storage (static game design document)."""

from typing import Any, Dict, Optional

WORLD_BIBLE: Optional[Dict[str, Any]] = None


def get_world_bible() -> Optional[Dict[str, Any]]:
    return WORLD_BIBLE


def set_world_bible(wb: Optional[Dict[str, Any]]) -> None:
    global WORLD_BIBLE
    WORLD_BIBLE = wb
