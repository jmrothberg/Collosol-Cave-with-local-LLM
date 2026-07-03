"""Tests for item matching helpers."""

from engine.item_matching import match_exit, match_item, normalize_item_token, split_item_list


def test_normalize_article():
    assert normalize_item_token("the Runic Key") == normalize_item_token("runic key")


def test_split_item_list_multi_and():
    # split_item_list splits on ' and '; try_local_command matches whole item names first
    assert split_item_list("key, lantern and flint") == ["key", "lantern", "flint"]


def test_match_item_snake_case():
    assert match_item("the key", ["runic_key", "lantern"]) == "runic_key"


def test_match_item_partial():
    assert match_item("lantern", ["elven_lantern"]) == "elven_lantern"


def test_match_exit_cardinal():
    exits = ["North Tunnel", "South Cave"]
    assert match_exit("n", exits) == "North Tunnel"
