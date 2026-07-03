"""Deterministic command dispatch helpers."""

from typing import Any, Callable, Dict, Optional


def normalize_entities(list_entities_by_type: Dict[str, list]) -> tuple:
    """Convert entity list dict to first/second element maps (matches main game)."""
    entities_by_type = {k: v[0] for k, v in list_entities_by_type.items() if v}
    entities_by_type_2nd = {
        k: v[1] if len(v) > 1 else None
        for k, v in list_entities_by_type.items() if v
    }
    return entities_by_type, entities_by_type_2nd


def dispatch_action(
    chosen_action: Optional[str],
    action_map: Dict[str, Callable[[], Any]],
    command: str,
    confidence: float,
    clarification_fn: Callable[[str, str, dict], str],
    error_suggestions_fn: Callable[[str, Optional[str], dict], str],
    error_recovery_fn: Callable[[str, str, str], str],
    entities_by_type: dict,
    list_entities_by_type: dict,
) -> Any:
    """
    Execute a parsed command via action_map with confidence-based fallbacks.
    Mirrors execute_action() control flow without importing game classes.
    """
    from .llm_commands import needs_clarification

    if chosen_action and len(chosen_action) > 1:
        pass
    action = chosen_action[0] if chosen_action else None

    entities_by_type_norm, entities_by_type_2nd = normalize_entities(list_entities_by_type)

    if needs_clarification(confidence, action):
        return clarification_fn(command, action, entities_by_type_norm)

    if action in action_map:
        try:
            return action_map[action]()
        except Exception as e:
            return error_recovery_fn(command, action, str(e))

    return error_suggestions_fn(command, action, entities_by_type_norm)
