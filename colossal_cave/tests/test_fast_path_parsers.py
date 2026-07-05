"""Unit tests for fast_path parser helpers."""

from engine.fast_path import (
    parse_attack_command,
    parse_category_keyword,
    parse_cheat_command,
    parse_drop_items,
    parse_remove_armor,
    parse_solve_items,
    parse_take_items,
    parse_use_item,
    parse_wear_armor,
    resolve_movement_direction,
    is_movement_phrase,
)


def test_parse_wear_armor_specific():
    assert parse_wear_armor("wear plate armor") == "plate armor"


def test_parse_wear_armor_generic():
    assert parse_wear_armor("wear the armor") == ""


def test_parse_remove_armor():
    assert parse_remove_armor("take off splint armor") == "splint armor"


def test_parse_use_item():
    assert parse_use_item("use healing potion") == "healing potion"


def test_parse_solve_named():
    assert parse_solve_items("solve Piano of Time") == ("piano of time", None)


def test_parse_solve_two_items():
    assert parse_solve_items("solve Echo Stone and Whispering Wind") == (
        "echo stone", "whispering wind",
    )


def test_parse_attack_kill():
    assert parse_attack_command("kill goblin with sword") == ("goblin", "sword")


def test_parse_attack_hit():
    assert parse_attack_command("hit dragon") == ("dragon", None)


def test_parse_cheat_xyzzy():
    assert parse_cheat_command("xyzzy") == ("xyzzy", None)


def test_parse_cheat_health():
    assert parse_cheat_command("health 100") == ("health", 100)


def test_parse_take_category():
    assert parse_take_items("take all weapons") == ("category", "weapon")


def test_parse_take_all_only():
    assert parse_take_items("take all") == ["all"]


def test_parse_take_all_the_items():
    assert parse_take_items("take all the items") == ["all"]


def test_is_movement_phrase_diagonal():
    assert is_movement_phrase("go southeast")
    assert is_movement_phrase("go south-east")


def test_resolve_diagonal_uses_stored_key():
    from types import SimpleNamespace

    room = SimpleNamespace(number=2, map_x=0, map_y=0, connected_rooms={})
    r14 = SimpleNamespace(number=14, map_x=1, map_y=1)
    room.connected_rooms = {"southeast": r14}
    assert resolve_movement_direction("go southeast", room) == "southeast"
    assert resolve_movement_direction("go se", room) == "southeast"
    assert resolve_movement_direction("go south-east", room) == "southeast"


def test_resolve_cardinal_falls_back_to_nearest():
    from types import SimpleNamespace

    room = SimpleNamespace(number=11, map_x=0, map_y=0, connected_rooms={})
    r7 = SimpleNamespace(number=7, map_x=1, map_y=1)
    room.connected_rooms = {"southeast": r7}
    # User says "west" but only southeast exists — nearest by angle
    assert resolve_movement_direction("go west", room) == "southeast"


def test_parse_drop_put_down():
    assert parse_drop_items("put down warhammer") == ["warhammer"]


def test_parse_category_keyword():
    assert parse_category_keyword("all magic items") == "magic"
