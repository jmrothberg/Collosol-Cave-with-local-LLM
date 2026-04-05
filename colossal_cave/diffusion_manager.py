#!/usr/bin/env python3
"""
Diffusion Manager - Unified interface for all diffusion model operations
Combines local image generation, remote server client, and server functionality.
"""

import os
import sys
import shutil
from datetime import datetime
from typing import Optional, Dict, List, Any
import re

# CRITICAL: Set CUDA environment BEFORE any torch imports!
cuda_lib_path = '/usr/local/cuda/lib64'
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if cuda_lib_path not in current_ld_path:
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{current_ld_path}".rstrip(':')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set CUDA memory management environment variables for better memory handling
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Now import torch AFTER CUDA environment is set
import torch

# Force CUDA to re-check after environment setup
if hasattr(torch.cuda, '_lazy_init'):
    torch.cuda._lazy_init()

print(f"[diffusion_manager] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[diffusion_manager] CUDA device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        try:
            print(f"[diffusion_manager] GPU {i}: {torch.cuda.get_device_name(i)}")
        except:
            print(f"[diffusion_manager] GPU {i}: Error getting device name")

# Server functionality removed - only local generation now

# Import diffusers components
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline
try:
    from diffusers import ZImagePipeline
except ImportError:
    ZImagePipeline = None






class ImageGenerator:
    """Local image generator for diffusion models."""

    def __init__(self, model_id: Optional[str] = None, device_preference: Optional[str] = None, local_root: Optional[str] = None):
        self.model_id = model_id
        self.device = self._resolve_device(device_preference)
        self.local_root = local_root
        self._pipeline = None  # lazy init
        self._torch_dtype = self._default_torch_dtype()

    def cleanup(self) -> None:
        """Clean up the current pipeline and free memory"""
        if self._pipeline is not None:
            try:
                del self._pipeline
                self._pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("[image] Cleaned up previous diffusion pipeline")
            except Exception as e:
                print(f"[image] Error cleaning up pipeline: {e}")

    def _resolve_device(self, device_pref: Optional[str]) -> str:
        # Check for explicit GPU override via environment variable
        gpu_override = os.environ.get("DIFFUSION_GPU")
        if gpu_override:
            try:
                gpu_id = int(gpu_override)
                if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
                    print(f"[image] Using GPU {gpu_id} (set by DIFFUSION_GPU environment variable)")
                    return f"cuda:{gpu_id}"
                else:
                    print(f"[image] DIFFUSION_GPU={gpu_override} not available, falling back to auto-selection")
            except ValueError:
                print(f"[image] Invalid DIFFUSION_GPU value: {gpu_override}, falling back to auto-selection")

        if device_pref:
            return device_pref
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            # For multi-GPU systems, try to find a less loaded GPU
            # Check if we have multiple GPUs (relevant for CUDA systems only, not MPS)
            gpu_count = torch.cuda.device_count()
            if gpu_count > 1:
                # Find the GPU with the most available memory
                best_gpu = 0
                max_free_memory = 0

                print(f"[image] Checking {gpu_count} GPUs for available memory:")
                for i in range(gpu_count):
                    try:
                        # Get both free and total memory for better calculation
                        free_bytes, total_bytes = torch.cuda.mem_get_info(i)
                        total_gb = total_bytes / (1024**3)
                        free_gb = free_bytes / (1024**3)
                        allocated_gb = torch.cuda.memory_allocated(i) / (1024**3)
                        reserved_gb = torch.cuda.memory_reserved(i) / (1024**3)
                        # More conservative estimate: account for PyTorch overhead and fragmentation
                        # FLUX needs ~16-20GB for inference, so be very conservative
                        effective_free_gb = free_gb - reserved_gb - 2.0  # 2GB safety buffer

                        print(f"[image] GPU {i}: {free_gb:.1f}GB free, {allocated_gb:.1f}GB allocated, {reserved_gb:.1f}GB reserved, {effective_free_gb:.1f}GB effective")

                        # Use effective free memory with safety buffer for FLUX
                        if effective_free_gb > max_free_memory and effective_free_gb >= 20.0:  # Minimum 20GB for FLUX
                            max_free_memory = effective_free_gb
                            best_gpu = i
                    except Exception as e:
                        print(f"[image] Error checking GPU {i}: {e}")
                        continue

                print(f"[image] Selected GPU {best_gpu} with {max_free_memory:.1f}GB available")
                return f"cuda:{best_gpu}"
            return "cuda"
        return "cpu"

    def _default_torch_dtype(self) -> torch.dtype:
        if self.device == "mps":
            return torch.float32
        if self.device.startswith("cuda"):
            return torch.float16
        return torch.float32

    def _local_model_path(self) -> Optional[str]:
        if not self.model_id:
            return None
        # Absolute path
        if os.path.isdir(self.model_id):
            return self.model_id
        # Optional explicit root
        if self.local_root:
            candidate = os.path.join(self.local_root, self.model_id)
            if os.path.isdir(candidate):
                return candidate
        # Environment override for diffusion models dir
        env_diff_dir = os.environ.get("DIFFUSION_MODELS_DIR")
        if env_diff_dir and os.path.isdir(env_diff_dir):
            for name in (self.model_id, self.model_id.split("/")[-1], self.model_id.replace("/", "_")):
                candidate = os.path.join(env_diff_dir, name)
                if os.path.isdir(candidate):
                    return candidate
        # Default path using same logic as diffusion_models_root()
        candidates = [
            "/Users/jonathanrothberg/Diffusion_Models",  # macOS
            "/home/jonathan/Models_Diffusers",          # Linux
            "/data/Diffusion_Models",                   # Alternative Linux
            "./Diffusion_Models",                       # Relative path
        ]

        default_diff_dir = None
        for candidate in candidates:
            if os.path.isdir(candidate):
                default_diff_dir = candidate
                break

        if not default_diff_dir:
            # Fallback
            default_model_root = "/Users/jonathanrothberg" if sys.platform == "darwin" else "/data"
            default_diff_dir = os.path.join(default_model_root, "Diffusion_Models")
        # Try multiple common folder namings
        candidates = [
            os.path.join(default_diff_dir, self.model_id),
            os.path.join(default_diff_dir, self.model_id.split("/")[-1]),
            os.path.join(default_diff_dir, self.model_id.replace("/", "_")),
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        return None

    def _apply_mps_compat_patches(self) -> None:
        # Mirror MPS safety patches used in Colossal Cave for FLUX
        torch.set_default_torch_dtype(torch.float32)
        original_from_numpy = torch.from_numpy
        def mps_safe_from_numpy(ndarray):
            tensor = original_from_numpy(ndarray)
            if tensor.torch_dtype == torch.float64:
                tensor = tensor.float()
            return tensor
        torch.from_numpy = mps_safe_from_numpy  # type: ignore
        original_arange = torch.arange
        def mps_safe_arange(*args, **kwargs):
            if 'torch_dtype' in kwargs and kwargs['torch_dtype'] == torch.float64:
                kwargs['torch_dtype'] = torch.float32
            return original_arange(*args, **kwargs)
        torch.arange = mps_safe_arange  # type: ignore

    def _lazy_init(self) -> None:
        if self._pipeline is not None or not self.model_id:
            return

        # Special case: Z-Image-Turbo - Load with correct parameters for Blackwell GPU
        if self.model_id == "Z-Image-Turbo":
            model_path = self._local_model_path()
            if not model_path:
                # Fall back to HuggingFace model ID like the working example
                model_path = "Tongyi-MAI/Z-Image-Turbo"
                print(f"[image] Local Z-Image model not found, using HuggingFace: {model_path}")

            print(f"Loading Z-Image pipeline from: {model_path}")
            try:
                if ZImagePipeline is None:
                    raise ImportError("ZImagePipeline not available in this diffusers version")
                # Load Z-Image with correct parameters for Blackwell GPU
                self._pipeline = ZImagePipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,  # Optimal for Blackwell GPU
                    low_cpu_mem_usage=False,     # No memory tricks needed
                )

                # Load directly to CUDA (full GPU utilization)
                if torch.cuda.is_available():
                    print("[image] Moving Z-Image to CUDA...")
                    self._pipeline.to("cuda")
                    print("[image] ✅ Z-Image loaded on CUDA!")
                else:
                    print("[image] ❌ No CUDA available - Z-Image requires CUDA")
                    self._pipeline = None
                    return

                # Skip attention optimization and compilation like the working test_cuda.py
                print("Z-Image pipeline loaded successfully!")
                return
            except Exception as load_error:
                print(f"[image] ❌ Z-Image loading failed: {load_error}")
                self._pipeline = None
                return

        model_local_path = self._local_model_path()
        try:
            if model_local_path:
                # Prefer local pipeline types matching model
                base_name_upper = os.path.basename(model_local_path).upper()
                print(f"[image] Loading local diffusion model from: {model_local_path}")
                if "SDXL" in base_name_upper:
                    self._pipeline = StableDiffusionXLPipeline.from_pretrained(
                        model_local_path,
                        torch_dtype=self._torch_dtype,
                        variant="fp16" if self.device.startswith("cuda") and self._torch_dtype == torch.float16 else None,
                        use_safetensors=True,
                    )
                    if self.device.startswith("cuda"):
                        self._pipeline = self._pipeline.to(self.device)
                    elif self.device == "mps":
                        self._pipeline = self._pipeline.to(self.device)
                    return

                if "FLUX" in base_name_upper or "FLUX" in (self.model_id or "").upper():
                    if self.device == "mps":
                        self._apply_mps_compat_patches()
                        self._pipeline = FluxPipeline.from_pretrained(
                            model_local_path,
                            torch_dtype=torch.float32,
                        )
                        self._pipeline = self._pipeline.to("mps")
                    elif self.device.startswith("cuda"):
                        torch.cuda.empty_cache()
                        gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0

                        # Check available GPU memory - if we have >= 20GB free, use GPU directly
                        try:
                            gpu_memory_free = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                            gpu_memory_used = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                            gpu_memory_available = gpu_memory_free - gpu_memory_used

                            # If we have >= 20GB available, load directly to GPU (FLUX needs ~12-16GB)
                            if gpu_memory_available >= 20.0:
                                print(f"[image] Local GPU {gpu_id}: {gpu_memory_available:.1f}GB available, loading FLUX directly to GPU")
                                self._pipeline = FluxPipeline.from_pretrained(
                                    model_local_path,
                                    torch_dtype=torch.bfloat16,
                                )
                                self._pipeline = self._pipeline.to(self.device)
                                print(f"[image] Local FLUX loaded directly to GPU {gpu_id}")
                            else:
                                print(f"[image] Local GPU {gpu_id}: {gpu_memory_available:.1f}GB available, using CPU offloading")
                                self._pipeline = FluxPipeline.from_pretrained(
                                    model_local_path,
                                    torch_dtype=torch.bfloat16,
                                )
                                if hasattr(self._pipeline, "enable_model_cpu_offload"):
                                    self._pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
                                    print(f"[image] Local FLUX using CPU offloading with GPU {gpu_id}")
                                else:
                                    self._pipeline = self._pipeline.to(self.device)
                        except Exception as e:
                            print(f"[image] Local memory check failed ({e}), using CPU offloading")
                            self._pipeline = FluxPipeline.from_pretrained(
                                model_local_path,
                                torch_dtype=torch.bfloat16,
                            )
                            if hasattr(self._pipeline, "enable_model_cpu_offload"):
                                self._pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
                                print(f"[image] Local FLUX using CPU offloading with GPU {gpu_id}")
                            else:
                                self._pipeline = self._pipeline.to(self.device)
                    else:
                        self._pipeline = FluxPipeline.from_pretrained(
                            model_local_path,
                            torch_dtype=torch.float32,
                        )
                else:
                    # Load generic diffusion pipeline
                    self._pipeline = DiffusionPipeline.from_pretrained(
                        model_local_path,
                        torch_dtype=self._torch_dtype,
                    )
                    if self.device.startswith("cuda"):
                        self._pipeline = self._pipeline.to(self.device)
                    elif self.device == "mps":
                        self._pipeline = self._pipeline.to(self.device)
                return

            name_upper = self.model_id.upper()
            if "FLUX" in name_upper:
                if self.device == "mps":
                    print("[image] Setting up FLUX for MPS with safety patches...")
                    self._apply_mps_compat_patches()
                    self._pipeline = FluxPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                    )
                    self._pipeline = self._pipeline.to("mps")
                elif self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0

                    # Check available GPU memory - if we have >= 20GB free, use GPU directly
                    try:
                        gpu_memory_free = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
                        gpu_memory_used = torch.cuda.memory_allocated(gpu_id) / (1024**3)
                        gpu_memory_available = gpu_memory_free - gpu_memory_used

                        # If we have >= 20GB available, load directly to GPU (FLUX needs ~12-16GB)
                        if gpu_memory_available >= 20.0:
                            print(f"[image] GPU {gpu_id}: {gpu_memory_available:.1f}GB available, loading FLUX directly to GPU")
                            self._pipeline = FluxPipeline.from_pretrained(
                                self.model_id,
                                torch_dtype=torch.bfloat16,
                            )
                            self._pipeline = self._pipeline.to(self.device)
                            print(f"[image] FLUX loaded directly to GPU {gpu_id} (sufficient memory available)")
                        else:
                            print(f"[image] GPU {gpu_id}: {gpu_memory_available:.1f}GB available, using CPU offloading")                            # Fallback to CPU offloading for memory management
                            self._pipeline = FluxPipeline.from_pretrained(
                                self.model_id,
                                torch_dtype=torch.bfloat16,
                            )
                            if hasattr(self._pipeline, "enable_model_cpu_offload"):
                                self._pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
                                print(f"[image] FLUX using CPU offloading with GPU {gpu_id}")
                            else:
                                self._pipeline = self._pipeline.to(self.device)
                    except Exception as e:
                        print(f"[image] Memory check failed ({e}), using CPU offloading")
                        # Fallback to CPU offloading
                        self._pipeline = FluxPipeline.from_pretrained(
                            self.model_id,
                            torch_dtype=torch.bfloat16,
                        )
                        if hasattr(self._pipeline, "enable_model_cpu_offload"):
                            self._pipeline.enable_model_cpu_offload(gpu_id=gpu_id)
                            print(f"[image] FLUX using CPU offloading with GPU {gpu_id}")
                        else:
                            self._pipeline = self._pipeline.to(self.device)
                else:
                    # CPU fallback
                    self._pipeline = FluxPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=torch.float32,
                    )
                return

            if "stable-diffusion-3" in self.model_id.lower():
                torch.cuda.empty_cache() if self.device == "cuda" else None
                self._pipeline = StableDiffusion3Pipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                )
                self._pipeline = self._pipeline.to(self.device if self.device in ("cuda", "mps") else "cpu")
                return

            # NO DOWNLOADS - only use local models
            print(f"[image] Model '{self.model_id}' not found locally. Only local models are supported.")
            self._pipeline = None
        except Exception as e:
            print(f"[image] Failed to initialize pipeline '{self.model_id}': {e}")
            # Try fallback to Z-Image-Turbo if available
            if self.model_id != "Z-Image-Turbo" and ZImagePipeline is not None and os.path.isdir('/data/Diffusion_Models/Z-Image-Turbo'):
                print(f"[image] Attempting fallback to Z-Image-Turbo")
                try:
                    self._pipeline = ZImagePipeline.from_pretrained(
                        '/data/Diffusion_Models/Z-Image-Turbo',
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=False,
                    )
                    self._pipeline.to("cuda")

                    # Enable optimized attention on fallback (skip on Blackwell GPUs)
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
                    if "GB10" in gpu_name or "Blackwell" in gpu_name:
                        print(f"[image] Skipping attention optimization on {gpu_name} fallback for stability")
                    else:
                        try:
                            import xformers
                            self._pipeline.transformer.set_attention_backend("xformers")
                            print("[image] xFormers attention enabled on fallback (ARM-compatible)")
                        except Exception as xformers_error:
                            try:
                                self._pipeline.transformer.set_attention_backend("flash")
                                print("[image] Flash attention enabled on fallback")
                            except Exception as flash_error:
                                print(f"[image] Optimized attention not available on fallback: xFormers({xformers_error}), Flash({flash_error})")

                    # Skip compilation on ARM where it may cause issues
                    try:
                        self._pipeline.transformer.compile()
                        print("[image] Model compiled for speed")
                    except Exception as compile_error:
                        print(f"[image] Model compilation failed (ARM limitation): {compile_error}")
                    print(f"[image] Successfully fell back to Z-Image-Turbo")
                    return
                except Exception as fallback_e:
                    print(f"[image] Fallback to Z-Image-Turbo also failed: {fallback_e}")
            self._pipeline = None

    def generate(self, prompt: str) -> Optional[str]:
        """Generate an image using cached pipeline."""
        if not self.model_id:
            print("[image] Image generation disabled (no model selected).")
            return None

        # Lazy initialize if needed
        self._lazy_init()
        if self._pipeline is None:
            return None

        try:
            from PIL import Image, ImageDraw, ImageFont
            import datetime

            # Generate image using cached pipeline
            image = self._pipeline(
                prompt=prompt,
                height=768,
                width=768,
                num_inference_steps=9,  # This actually results in 8 DiT forwards
                guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
                generator=torch.Generator("cuda").manual_seed(42),
            ).images[0]

            # Add date overlay (EXACT from working test)
            current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            draw = ImageDraw.Draw(image)

            # Try to use a nice font, fallback to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            except:
                font = ImageFont.load_default()

            # Add semi-transparent background for text
            text_bbox = draw.textbbox((0, 0), f"LLM Adventure - {current_date}", font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position in bottom-right corner with padding
            x = image.width - text_width - 20
            y = image.height - text_height - 20

            # Draw semi-transparent rectangle
            draw.rectangle([x-10, y-10, x+text_width+10, y+text_height+10],
                           fill=(0, 0, 0, 128))

            # Draw text
            draw.text((x, y), f"LLM Adventure - {current_date}", fill=(255, 255, 255), font=font)

            # Create Generated_Art directory if it doesn't exist
            output_dir = "Generated_Art"
            os.makedirs(output_dir, exist_ok=True)

            # Create filename: short_prompt_YYYYMMDD_HHMMSS.png
            clean_prompt = re.sub(r'[^\w\s-]', '', prompt)
            clean_prompt = re.sub(r'\s+', '_', clean_prompt)
            if len(clean_prompt) > 30:
                clean_prompt = clean_prompt[:30].rstrip('_')
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{clean_prompt}_{ts}.png"
            filepath = os.path.join(output_dir, filename)

            # Save image
            image.save(filepath)

            return filepath
        except Exception as e:
            print(f"[image] Generation error: {e}")
            return None

