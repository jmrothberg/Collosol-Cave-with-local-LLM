"""Tests for engine.ux_helpers.progress_summary."""

from types import SimpleNamespace

from engine.ux_helpers import progress_summary
from tests.conftest import _StubRoom


def test_progress_summary_when_few_items():
    r1 = _StubRoom(1, explored=True)
    r2 = _StubRoom(2, explored=False)
    player = SimpleNamespace(items=[])
    text = progress_summary(player, {1: r1, 2: r2})
    assert "haven't visited" in text or "MB" in text
