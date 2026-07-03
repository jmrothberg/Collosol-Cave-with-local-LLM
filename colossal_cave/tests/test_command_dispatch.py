"""Tests for engine.commands dispatch helpers."""

from engine.commands import dispatch_action, normalize_entities


def test_normalize_entities():
    raw = {"room_items": ["sword", "shield"], "monsters": ["dragon"]}
    first, second = normalize_entities(raw)
    assert first["room_items"] == "sword"
    assert second["room_items"] == "shield"
    assert first["monsters"] == "dragon"
    assert second["monsters"] is None


def test_dispatch_action_runs_handler():
    called = []

    def handler():
        called.append(True)
        return "ok"

    result = dispatch_action(
        chosen_action=["map"],
        action_map={"map": handler},
        command="map",
        confidence=1.0,
        clarification_fn=lambda *a: "clarify",
        error_suggestions_fn=lambda *a: "suggest",
        error_recovery_fn=lambda *a: "recover",
        entities_by_type={},
        list_entities_by_type={},
    )
    assert result == "ok"
    assert called


def test_dispatch_low_confidence_clarification():
    result = dispatch_action(
        chosen_action=["attack"],
        action_map={},
        command="attack dragon",
        confidence=0.1,
        clarification_fn=lambda c, a, e: f"clarify:{a}",
        error_suggestions_fn=lambda *a: "suggest",
        error_recovery_fn=lambda *a: "recover",
        entities_by_type={},
        list_entities_by_type={},
    )
    assert result == "clarify:attack"
