"""Puzzle handlers — tool use, action words, varied puzzle types (CHANGE: tool-use + puzzle-variety)."""

from typing import Any, Callable, Dict, List, Optional, Tuple

from engine.items import find_item_smart


def normalize_action_word(text: str) -> str:
    return text.lower().strip().split()[0] if text else ""


def try_action_word_puzzle(room: Any, verb: str, player_items: List[Any]) -> Optional[str]:
    """
    Enforce riddle_action_words at runtime.
    Returns success message or None if not applicable.
    """
    if not room or getattr(room, "riddle_solved", False) or not getattr(room, "riddles", None):
        return None
    action_words = getattr(room, "riddle_action_words", None)
    if not action_words:
        return None
    if isinstance(action_words, str):
        expected = [action_words]
    elif isinstance(action_words, (list, tuple)):
        expected = list(action_words)
    else:
        return None
    verb_norm = normalize_action_word(verb)
    if verb_norm not in [normalize_action_word(a) for a in expected]:
        return None
    required = list(getattr(room, "riddle_magic_items", []) or [])
    magic_names = [i.name for i in player_items if i.__class__.__name__ == "Magic"]
    if required and not all(r in magic_names for r in required):
        return (
            f"You {verb_norm}, but you still need: {', '.join(required)}.\n"
        )
    room.riddle_solved = True
    return f"You {verb_norm} — the riddle is solved!\n"


def try_feature_action(
    room: Any,
    verb: str,
    item_name: Optional[str],
    player_items: List[Any],
    target_name: Optional[str] = None,
) -> Optional[str]:
    """Use verb/item on a room feature (tool use)."""
    features = getattr(room, "features", None) or []
    if not features:
        return None
    verb_norm = normalize_action_word(verb)
    target_norm = (target_name or "").lower().strip()
    for feat in features:
        if not isinstance(feat, dict):
            continue
        feat_name = (feat.get("name") or "").lower()
        if target_norm and feat_name and target_norm not in feat_name and feat_name not in target_norm:
            continue
        feat_verbs = feat.get("verbs", [])
        if verb_norm not in [normalize_action_word(v) for v in feat_verbs]:
            continue
        required_item = feat.get("requires_item")
        if required_item:
            names = [i.name for i in player_items]
            has_required = required_item in names
            if item_name:
                found = find_item_smart(item_name, player_items)
                if found and found.name == required_item:
                    has_required = True
            if not has_required:
                return f"You need {required_item} to {verb_norm} the {feat.get('name', 'object')}.\n"
        if feat.get("solves_riddle") and getattr(room, "riddles", None):
            room.riddle_solved = True
            return feat.get("success", f"You {verb_norm} the {feat.get('name', 'object')}.\n")
        return feat.get("success", f"You {verb_norm} the {feat.get('name', 'object')}.\n")
    return None


def handle_item_solve(room: Any, item_names: List[str], player_items: List[Any]) -> bool:
    """Classic solve-by-magic-item-name (puzzle_type=item_solve)."""
    required = list(getattr(room, "riddle_magic_items", []) or [])
    return set(item_names) == set(required) and all(
        any(i.name == n for i in player_items if i.__class__.__name__ == "Magic") for n in required
    )


def handle_typed_answer(room: Any, answer: str) -> bool:
    """Typed-answer puzzles (puzzle_type=typed_answer)."""
    answers = getattr(room, "puzzle_answers", None) or []
    return answer.lower().strip() in [a.lower().strip() for a in answers]


PUZZLE_HANDLERS: Dict[str, Callable] = {
    "item_solve": handle_item_solve,
    "typed_answer": handle_typed_answer,
    "verb_on_feature": try_action_word_puzzle,
}
