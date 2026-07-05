"""Direct MLX inference for folders under ~/MLX_Models (no Ollama required)."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional, Tuple

OLLAMA_CHOICE_PREFIX = "ollama:"

# model_type values that need mlx-vlm (not plain mlx-lm)
VLM_MODEL_TYPES = frozenset(
    {
        "gemma4_unified",
        "gemma3n",
        "gemma3",
        "qwen2_vl",
        "qwen2_5_vl",
        "qwen3_vl",
        "qwen3_5",
        "llava",
        "llava_next",
        "idefics",
        "idefics2",
        "idefics3",
        "mllama",
        "pixtral",
        "paligemma",
        "phi3_v",
        "molmo",
        "aya_vision",
        "deepseek_vl",
        "deepseek_vl_v2",
        "smolvlm",
        "florence2",
    }
)

_CACHE: Dict[str, Dict[str, Any]] = {}


def mlx_models_dir() -> str:
    return os.environ.get("COLOSSAL_CAVE_MLX_MODELS_DIR", os.path.expanduser("~/MLX_Models"))


def llm_choice_is_ollama(model_name: str) -> bool:
    """True when dropdown value explicitly selects Ollama (ollama:<tag>)."""
    return bool(model_name) and model_name.startswith(OLLAMA_CHOICE_PREFIX)


def ollama_model_from_choice(model_name: str) -> str:
    if llm_choice_is_ollama(model_name):
        return model_name[len(OLLAMA_CHOICE_PREFIX) :]
    return model_name


def resolve_mlx_folder_path(model_name: str) -> Optional[str]:
    """Return absolute path to an MLX weights folder, or None."""
    if not model_name or llm_choice_is_ollama(model_name):
        return None
    name = model_name.strip()
    if os.path.isdir(name) and os.path.isfile(os.path.join(name, "config.json")):
        return os.path.abspath(name)
    mlx_dir = mlx_models_dir()
    path = os.path.join(mlx_dir, name)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json")):
        return os.path.abspath(path)
    return None


def model_backend(model_name: str) -> Optional[str]:
    """'mlx', 'ollama', or None if the name cannot be resolved."""
    if not model_name or model_name == "No Ollama models found":
        return None
    if llm_choice_is_ollama(model_name):
        return "ollama"
    if resolve_mlx_folder_path(model_name):
        return "mlx"
    return "ollama"


def _read_config(model_path: str) -> dict:
    with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _is_vlm_model(model_path: str) -> bool:
    cfg = _read_config(model_path)
    model_type = (cfg.get("model_type") or "").lower()
    if model_type in VLM_MODEL_TYPES:
        return True
    arch = " ".join(cfg.get("architectures") or []).lower()
    if any(k in model_type for k in ("vlm", "vision", "multimodal", "unified")):
        return True
    return any(k in arch for k in ("vlm", "vision", "forconditionalgeneration"))


def _clean_prompt_for_mlx(prompt: str, model_name: str) -> str:
    """Strip Llama chat tokens; keep plain text for MLX chat templates."""
    llama_models = ["llama3", "llama2", "llama-3", "llama-2"]
    uses_llama_format = any((model_name or "").lower().startswith(m) for m in llama_models)
    if uses_llama_format:
        return prompt.strip()
    text = prompt.replace("<|begin_of_text|>", "")
    text = text.replace("<|start_header_id|>system<|end_header_id|>", "System: ")
    text = text.replace("<|start_header_id|>user<|end_header_id|>", "\nUser: ")
    text = text.replace("<|start_header_id|>assistant<|end_header_id|>", "\nAssistant: ")
    text = text.replace("<|eot_id|>", "\n")
    return text.strip()


def _load_entry(model_path: str) -> Dict[str, Any]:
    if model_path in _CACHE:
        return _CACHE[model_path]
    if _is_vlm_model(model_path):
        from mlx_vlm import load as vlm_load
        from mlx_vlm.utils import load_config

        model, processor = vlm_load(model_path)
        config = load_config(model_path)
        entry = {"kind": "vlm", "model": model, "processor": processor, "config": config}
    else:
        from mlx_lm import load as lm_load

        model, tokenizer = lm_load(model_path)
        entry = {"kind": "lm", "model": model, "tokenizer": tokenizer}
    _CACHE[model_path] = entry
    return entry


def _generate_vlm(entry: Dict[str, Any], prompt: str, max_tokens: int, temperature: float) -> str:
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

    formatted = apply_chat_template(entry["processor"], entry["config"], prompt, num_images=0)
    kwargs: Dict[str, Any] = {"verbose": False, "max_tokens": max_tokens}
    if temperature is not None:
        kwargs["temp"] = temperature
    output = vlm_generate(entry["model"], entry["processor"], formatted, **kwargs)
    if hasattr(output, "text"):
        return str(output.text).strip()
    return str(output).strip()


def _generate_lm(entry: Dict[str, Any], prompt: str, max_tokens: int, temperature: float) -> str:
    from mlx_lm import generate as lm_generate
    from mlx_lm.sample_utils import make_sampler

    tokenizer = entry["tokenizer"]
    prompt_tokens = prompt
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            messages = [{"role": "user", "content": prompt}]
            prompt_tokens = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:
            prompt_tokens = prompt
    sampler = make_sampler(temp=temperature)
    return lm_generate(
        entry["model"],
        tokenizer,
        prompt=prompt_tokens,
        max_tokens=max_tokens,
        sampler=sampler,
        verbose=False,
    ).strip()


def mlx_generate_text(
    model_name: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Load model from ~/MLX_Models/<model_name> and generate text."""
    model_path = resolve_mlx_folder_path(model_name)
    if not model_path:
        raise FileNotFoundError(f"MLX folder not found for {model_name!r} under {mlx_models_dir()!r}")
    cleaned = _clean_prompt_for_mlx(prompt, model_name)
    if json_mode:
        cleaned = cleaned + "\n\nRespond with valid JSON only."
    entry = _load_entry(model_path)
    if entry["kind"] == "vlm":
        return _generate_vlm(entry, cleaned, max_tokens, temperature)
    return _generate_lm(entry, cleaned, max_tokens, temperature)


def mlx_model_label(model_name: str) -> str:
    """Human-readable backend hint for debug logs."""
    path = resolve_mlx_folder_path(model_name)
    if not path:
        return ""
    kind = "vlm" if _is_vlm_model(path) else "lm"
    return f"mlx:{kind}@{path}"
