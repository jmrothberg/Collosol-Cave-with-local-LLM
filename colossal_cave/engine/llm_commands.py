"""LLM command parsing helpers — shortcuts and JSON response parsing."""

import json
from typing import Any, Dict, List, Optional, Tuple


COMMAND_SHORTCUTS = {
    'n': 'go north', 's': 'go south', 'e': 'go east', 'w': 'go west',
    'i': 'inventory', 'l': 'look around', 'h': 'help', '?': 'help',
}

ALL_ACTIONS = (
    "go", "attack", "trade", "talk", "teach", "use", "wear", "remove", "take",
    "leave", "solve", "describe_location", "describe_room_item", "describe_my_item",
    "health", "magic_word", "help", "cheat", "hint", "magic", "riddle_status",
    "question", "inventory", "status", "map",
)

DETERMINISTIC_ACTIONS = frozenset({
    "help", "magic", "riddle_status", "inventory", "status", "map", "cheat", "hint",
})


def apply_command_shortcuts(command: str) -> str:
    """Expand single-letter movement/inventory shortcuts."""
    cmd_lower = command.lower().strip()
    return COMMAND_SHORTCUTS.get(cmd_lower, command)


def parse_llm_json_response(response_text: str) -> Dict[str, Any]:
    """Parse Ollama JSON-mode response into action + entities + confidence."""
    entity_keys = [
        "location", "room_items", "monsters", "npcs", "npc_items", "my_items",
        "my_magic", "weapons", "my_armor", "worn_armor", "directions", "values",
    ]
    result: Dict[str, Any] = {
        "chosen_action": [],
        "entities": {k: [] for k in entity_keys},
        "confidence": 0.5,
    }

    try:
        cleaned = response_text.replace('"cchosen_action"', '"chosen_action"')
        cleaned = cleaned.replace('"atttack"', '"attack"')
        parsed = json.loads(cleaned)

        result["chosen_action"] = parsed.get("chosen_action", [])
        if not isinstance(result["chosen_action"], list):
            result["chosen_action"] = [result["chosen_action"]] if result["chosen_action"] else []

        # CHANGE (LLM): models sometimes return entity_categories: null
        entity_categories = parsed.get("entity_categories") or {}
        if not isinstance(entity_categories, dict):
            entity_categories = {}
        for category, entities in entity_categories.items():
            if category in result["entities"]:
                if entities is None:
                    entities = []
                elif not isinstance(entities, list):
                    entities = [entities]
                result["entities"][category] = entities

        result["confidence"] = parsed.get("confidence", 0.8)
    except json.JSONDecodeError:
        result["confidence"] = 0.3
        result["chosen_action"] = []

    return result


def filter_entities_for_room(
    result: Dict[str, Any],
    room_item_names: List[str],
    monster_names: List[str],
    directions: List[str],
) -> Dict[str, Any]:
    """Validate/filter LLM entity lists against current room state."""
    special_keywords = ["all", "everything"]
    if room_item_names:
        result["entities"]["room_items"] = [
            e for e in result["entities"]["room_items"]
            if e in room_item_names or e in special_keywords
        ]
    if monster_names:
        result["entities"]["monsters"] = [
            e for e in result["entities"]["monsters"] if e in monster_names
        ]
    if directions:
        result["entities"]["directions"] = [
            e for e in result["entities"]["directions"] if e in directions
        ]
    return result


def needs_clarification(confidence: float, chosen_action: Optional[str]) -> bool:
    """True when low-confidence complex commands should ask for clarification."""
    if confidence >= 0.4 or not chosen_action:
        return False
    return chosen_action not in ('go', 'take', 'leave')
