"""Standalone script import smoke tests."""

import importlib
import os
import sys

import pytest

_COLOSSAL_CAVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLOSSAL_CAVE_DIR not in sys.path:
    sys.path.insert(0, _COLOSSAL_CAVE_DIR)


@pytest.mark.parametrize("module_name", [
    "complete_instruction",
    "engine.preprocess",
    "engine.data",
])
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_config_devices_imports_when_torch_present():
    """config.devices needs torch — covered in test_gpu_smoke on Mac/CUDA."""
    pytest.importorskip("torch")
    importlib.import_module("config.devices")


def test_complete_instructions_nonempty():
    from complete_instruction import get_complete_instructions
    text = get_complete_instructions()
    assert "Basic Gameplay" in text
    assert len(text) > 100
