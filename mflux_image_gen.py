"""
Mac-native image generator using MFLUX (FLUX on Apple Silicon via MLX).

Supports both FLUX.1 (via Flux1) and FLUX.2 (via Flux2Klein).
Drop-in replacement for the old diffusion_manager.ImageGenerator.
API: generate(prompt) -> Optional[str], cleanup()
"""

import gc
import os
import random
import re
from datetime import datetime
from typing import Optional

# Known FLUX.2 model name prefixes (use Flux2Klein for these)
_FLUX2_NAMES = {"flux2-klein", "klein", "flux2"}

DIFFUSION_MODELS_DIR = "/Users/jonathanrothberg/Diffusion_Models"


def _is_flux2(model_name: str) -> bool:
    """Check if a model name refers to a FLUX.2 variant."""
    lower = model_name.lower()
    return any(lower.startswith(prefix) for prefix in _FLUX2_NAMES)


class MfluxImageGenerator:
    """Image generator using MFLUX (FLUX.1 or FLUX.2 on Apple Silicon)."""

    def __init__(self, model_name="flux2-klein-9b", local_path=None, quantize=None):
        self.model_name = model_name
        self.local_path = local_path or self._find_local_model(model_name)
        self.quantize = quantize
        self.model_id = model_name  # compatibility with getattr checks
        self._flux = None  # lazy-loaded on first generate()
        self._is_flux2 = _is_flux2(model_name)

    def _find_local_model(self, name):
        """Check standard paths for local FLUX models."""
        candidates = [
            os.path.join(DIFFUSION_MODELS_DIR, name),
            os.path.join(DIFFUSION_MODELS_DIR, f"FLUX.1-{name}"),
            os.path.join(DIFFUSION_MODELS_DIR, f"FLUX2-{name}"),
            # MLX quantized variants often use different naming
            os.path.join(DIFFUSION_MODELS_DIR, "FLUX2-klein-9B-mlx-8bit"),
        ]
        for p in candidates:
            if os.path.isdir(p):
                return p
        return None  # will download from HF on first use

    def _lazy_init(self):
        if self._flux is not None:
            return

        from mflux.models.common.config.model_config import ModelConfig

        if self._is_flux2:
            from mflux.models.flux2.variants import Flux2Klein
            self._flux = Flux2Klein(
                model_config=ModelConfig.from_name(self.model_name),
                model_path=self.local_path,
                quantize=self.quantize,
            )
        else:
            from mflux.models.flux.variants.txt2img.flux import Flux1
            self._flux = Flux1(
                model_config=ModelConfig.from_name(self.model_name),
                model_path=self.local_path,
                quantize=self.quantize,
            )

    def generate(self, prompt: str) -> Optional[str]:
        """Generate image from prompt. Returns absolute file path to PNG, or None."""
        try:
            self._lazy_init()

            if self._is_flux2:
                result = self._flux.generate_image(
                    seed=random.randint(0, 2**32 - 1),
                    prompt=prompt,
                    num_inference_steps=4,
                    height=768,
                    width=768,
                    guidance=1.0,
                    scheduler="flow_match_euler_discrete",
                )
            else:
                result = self._flux.generate_image(
                    seed=random.randint(0, 2**32 - 1),
                    prompt=prompt,
                    num_inference_steps=20,
                    height=768,
                    width=768,
                    guidance=3.5,
                )

            os.makedirs("Generated_Art", exist_ok=True)
            safe = re.sub(r'[^a-zA-Z0-9_]', '_', prompt[:60])
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.abspath(f"Generated_Art/{safe}_{ts}.png")
            result.save(path=filepath)
            return filepath
        except Exception as e:
            print(f"[MFLUX] Image generation failed: {e}")
            return None

    def cleanup(self):
        """Release model memory."""
        self._flux = None
        gc.collect()
