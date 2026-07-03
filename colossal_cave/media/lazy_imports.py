"""Lazy media imports — defer heavy torch/diffusers until image/video enabled."""

from typing import Any, Optional

_pipe = None
_pipe_name: Optional[str] = None


def get_diffusion_pipeline(model_name: str, device: str, dtype: Any) -> Any:
    """Load diffusers pipeline on first use."""
    global _pipe, _pipe_name
    if _pipe is not None and _pipe_name == model_name:
        return _pipe

    from diffusers import DiffusionPipeline  # lazy import

    _pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=dtype)
    _pipe = _pipe.to(device)
    _pipe_name = model_name
    return _pipe


def clear_diffusion_cache() -> None:
    """Release cached pipeline (for tests)."""
    global _pipe, _pipe_name
    _pipe = None
    _pipe_name = None
