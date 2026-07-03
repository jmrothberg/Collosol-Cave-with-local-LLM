"""Item name matching — exact, substring, and fuzzy fallback."""

from typing import Any, List, Optional

try:
    from fuzzywuzzy import fuzz
except ImportError:
    fuzz = None  # type: ignore


def find_item_smart(item_name: Optional[str], items: List[Any]) -> Optional[Any]:
    """Find item using intelligent partial matching prioritizing substring matches."""
    if not item_name or not items:
        return None

    item_name_lower = item_name.lower()

    for item in items:
        if item.name.lower() == item_name_lower:
            return item

    for item in items:
        if item_name_lower in item.name.lower():
            return item

    for item in items:
        if item.name.lower() in item_name_lower:
            return item

    if fuzz is None:
        return None

    best_match = None
    best_score = 0
    for item in items:
        score = fuzz.ratio(item.name.lower(), item_name_lower)
        if score > best_score and score > 60:
            best_score = score
            best_match = item

    return best_match
