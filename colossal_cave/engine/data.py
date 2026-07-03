"""Adventure data loading and JSON schema validation."""

import json
import os
from typing import Any, Dict, List, Tuple


REQUIRED_TOP_LEVEL_KEYS = (
    "room_descs_riddles",
    "monsters",
    "weapons",
    "armors",
    "treasures",
    "magic_items",
    "npcs",
    "npc_descs",
)


def load_adventure_data_file(path: str) -> Dict[str, Any]:
    """Load adventure JSON from disk. Raises on missing/invalid file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_adventure_data(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate adventure_dataRA.json schema and riddle consistency."""
    issues: List[str] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        if key not in data:
            issues.append(f"missing top-level key: {key}")
        elif not isinstance(data[key], list):
            issues.append(f"{key} must be a list")

    rooms = data.get("room_descs_riddles", [])
    if isinstance(rooms, list) and len(rooms) < 25:
        issues.append(f"expected at least 25 room descriptions, got {len(rooms)}")

    for idx, room in enumerate(rooms if isinstance(rooms, list) else []):
        if not isinstance(room, dict):
            issues.append(f"room_descs_riddles[{idx}] is not an object")
            continue
        riddles = room.get("riddles") or []
        magic = room.get("magic_items") or []
        action_words = room.get("action_words")
        if riddles:
            if not magic:
                issues.append(f"room {idx}: riddle without magic_items")
            if action_words is None or action_words == "":
                issues.append(f"room {idx}: riddle without action_words")

    return (len(issues) == 0, issues)


def default_adventure_data_path(base_dir: str) -> str:
    """Return canonical path to adventure_dataRA.json relative to colossal_cave/."""
    return os.path.join(base_dir, "adventure_dataRA.json")
