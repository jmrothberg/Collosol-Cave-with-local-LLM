"""Tests for engine.items.find_item_smart."""

from engine.items import find_item_smart


def test_exact_match(stub_item):
    items = [stub_item("Healing Potion"), stub_item("Iron Sword")]
    assert find_item_smart("Healing Potion", items).name == "Healing Potion"


def test_substring_match(stub_item):
    items = [stub_item("Healing Potion")]
    assert find_item_smart("healing", items).name == "Healing Potion"


def test_reverse_substring(stub_item):
    items = [stub_item("Sword")]
    assert find_item_smart("rusty iron sword", items).name == "Sword"


def test_fuzzy_match(stub_item):
    items = [stub_item("Healing Potion")]
    assert find_item_smart("Healng Poton", items).name == "Healing Potion"


def test_no_match(stub_item):
    items = [stub_item("Healing Potion")]
    assert find_item_smart("xyz", items) is None


def test_empty_inputs():
    assert find_item_smart("", []) is None
    assert find_item_smart("sword", []) is None
