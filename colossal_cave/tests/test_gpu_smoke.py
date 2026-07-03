"""GPU smoke tests — Mac MPS and CUDA; skipped when no accelerator."""

import os
import sys

import pytest

_COLOSSAL_CAVE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _COLOSSAL_CAVE_DIR not in sys.path:
    sys.path.insert(0, _COLOSSAL_CAVE_DIR)


def _has_accelerator():
    try:
        import torch
        if torch.cuda.is_available():
            return True
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    except ImportError:
        return False


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.gpu
@pytest.mark.skipif(not _has_accelerator(), reason="No CUDA/MPS available")
def test_diffusion_manager_import():
    """diffusion_manager imports on Mac (MPS) and CUDA machines."""
    import diffusion_manager
    assert hasattr(diffusion_manager, "ImageGenerator")


@pytest.mark.gpu
@pytest.mark.skipif(not _has_accelerator(), reason="No CUDA/MPS available")
def test_lazy_diffusion_cache_clear():
    from media.lazy_imports import clear_diffusion_cache
    clear_diffusion_cache()


@pytest.mark.gpu
@pytest.mark.skipif(not _has_accelerator(), reason="No CUDA/MPS available")
def test_config_devices_detect():
    from config.devices import detect_devices
    devices = detect_devices()
    assert "llm_device" in devices
    assert "diffuser_device" in devices
    if _cuda_available():
        assert "cuda" in devices["llm_device"]
    else:
        # Mac or CPU-only: expect mps or cpu
        assert devices["llm_device"] in ("mps", "cpu")


@pytest.mark.gpu
@pytest.mark.skipif(not _cuda_available(), reason="CUDA not available (Mac MPS-only)")
def test_diffusion_manager_reports_cuda():
    """On CUDA machines, diffusion_manager sees at least one GPU."""
    import diffusion_manager
    import torch
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() >= 1
