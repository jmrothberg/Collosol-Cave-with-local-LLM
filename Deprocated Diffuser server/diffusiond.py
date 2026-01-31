#!/usr/bin/env python3
"""
DiffusionD - Ollama-like server for diffusion models
A centralized service for serving multiple diffusion models via REST API.

Usage:
    python diffusiond.py                    # Run server
    python diffusiond.py --host 0.0.0.0    # Bind to all interfaces
    python diffusiond.py --port 8080       # Use custom port
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import re
import torch
from pathlib import Path

# Set CUDA memory management environment variables for better memory handling
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Import diffusers components
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline, ZImagePipeline


class GenerateRequest(BaseModel):
    model: str
    prompt: str
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None


class ModelInfo(BaseModel):
    name: str
    size: str
    type: str
    loaded: bool = False


class DiffusionServer:
    def __init__(self, models_dir: str = None, gpu_id: int = None):
        self.models_dir = models_dir or self._find_models_dir()
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs: Dict[str, Dict] = {}
        self.model_last_used: Dict[str, float] = {}  # Track last usage time for LRU eviction
        self.device = self._detect_device(gpu_id)

        print(f"[diffusiond] Using device: {self.device}")
        print(f"[diffusiond] Models directory: {self.models_dir}")

        # Scan available models
        self._scan_models()

    def _find_models_dir(self) -> str:
        """Find the Diffusion_Models directory."""
        # Check environment variable first
        env_dir = os.environ.get("DIFFUSION_MODELS_DIR")
        if env_dir and os.path.isdir(env_dir):
            return env_dir

        # Check common locations
        candidates = [
            "/data/Diffusion_Models",
            "/Users/jonathanrothberg/Diffusion_Models",
            "./Diffusion_Models",
        ]

        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate

        # Default to ./Diffusion_Models
        return "./Diffusion_Models"

    def _detect_device(self, gpu_id: int = None) -> str:
        """Detect the best available device."""
        if torch.cuda.is_available():
            if gpu_id is not None:
                if gpu_id < torch.cuda.device_count():
                    return f"cuda:{gpu_id}"
                else:
                    print(f"[diffusiond] Warning: GPU {gpu_id} not available, falling back to cuda:0")
                    return "cuda:0"
            else:
                # Auto-select GPU with most available memory
                if torch.cuda.device_count() > 1:
                    max_free_memory = 0
                    best_gpu = 0
                    for i in range(torch.cuda.device_count()):
                        try:
                            torch.cuda.set_device(i)
                            free_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            used_memory = torch.cuda.memory_allocated(i) / (1024**3)
                            available_memory = free_memory - used_memory
                            if available_memory > max_free_memory:
                                max_free_memory = available_memory
                                best_gpu = i
                        except Exception as e:
                            print(f"[diffusiond] Error checking GPU {i}: {e}")
                            continue

                    selected_gpu = best_gpu
                    print(f"[diffusiond] Auto-selected GPU {selected_gpu} with most available memory ({max_free_memory:.1f}GB free)")
                    print(f"[diffusiond] GPU memory summary:")
                    for i in range(torch.cuda.device_count()):
                        try:
                            torch.cuda.set_device(i)
                            free_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                            used_memory = torch.cuda.memory_allocated(i) / (1024**3)
                            available_memory = free_memory - used_memory
                            print(f"[diffusiond]   GPU {i}: {available_memory:.1f}GB available ({free_memory:.1f}GB total - {used_memory:.1f}GB used)")
                        except Exception as e:
                            print(f"[diffusiond]   GPU {i}: Error checking memory - {e}")
                    return f"cuda:{selected_gpu}"
                else:
                    return "cuda:0"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _scan_models(self):
        """Scan the models directory for available diffusion models."""
        if not os.path.exists(self.models_dir):
            print(f"[diffusiond] Models directory not found: {self.models_dir}")
            return

        print(f"[diffusiond] Scanning models in: {self.models_dir}")

        for item in os.listdir(self.models_dir):
            model_path = os.path.join(self.models_dir, item)
            if not os.path.isdir(model_path):
                continue

            # Try to determine model type from name/directory structure
            model_type = self._detect_model_type(item, model_path)
            if model_type:
                self.model_configs[item] = {
                    'path': model_path,
                    'type': model_type,
                    'name': item
                }
                print(f"[diffusiond] Found {model_type} model: {item}")

    def _detect_model_type(self, name: str, path: str) -> Optional[str]:
        """Detect the type of diffusion model."""
        name_upper = name.upper()

        # Check for specific model types
        if 'FLUX' in name_upper:
            return 'flux'
        elif 'STABLE-DIFFUSION-XL' in name_upper or 'SDXL' in name_upper:
            return 'sdxl'
        elif 'STABLE-DIFFUSION-3' in name_upper or 'SD3' in name_upper:
            return 'sd3'
        elif 'Z-IMAGE' in name_upper:
            return 'zimage'
        elif 'QWEN' in name_upper:
            return 'qwen'

        # Check for common config files
        if os.path.exists(os.path.join(path, 'model_index.json')):
            try:
                with open(os.path.join(path, 'model_index.json'), 'r') as f:
                    config = json.load(f)
                    if '_class_name' in config and 'QwenImagePipeline' in config['_class_name']:
                        return 'qwen'
                    elif 'FluxTransformer2DModel' in str(config):
                        return 'flux'
                    elif '_class_name' in config and 'StableDiffusionXL' in config['_class_name']:
                        return 'sdxl'
                    elif '_class_name' in config and 'StableDiffusion3' in config['_class_name']:
                        return 'sd3'
            except:
                pass

        # Generic diffusion pipeline as fallback
        return 'generic'

    def _update_model_usage(self, model_name: str):
        """Update the last usage timestamp for a model."""
        self.model_last_used[model_name] = time.time()

    def _check_memory_before_load(self, model_name: str) -> bool:
        """No memory checking bullshit - just load the damn model."""
        return True


    def _monitor_memory_usage(self):
        """No memory monitoring bullshit."""
        pass

    def _load_model(self, model_name: str):
        """Load a model into memory."""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found")

        if model_name in self.loaded_models:
            self._update_model_usage(model_name)  # Update usage timestamp
            return  # Already loaded

        config = self.model_configs[model_name]
        model_path = config['path']
        model_type = config['type']

        print(f"[diffusiond] Loading {model_type} model: {model_name}")

        try:
            # Load based on model type
            if model_type == 'flux':
                if self.device == 'mps':
                    # Apply MPS compatibility patches
                    self._apply_mps_compat_patches()
                    pipeline = FluxPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                    )
                    pipeline = pipeline.to("mps")
                elif self.device == 'cuda':
                    # Load FLUX directly to GPU with bfloat16 - no CPU offloading
                    pipeline = FluxPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        low_cpu_mem_usage=True,  # Optimize for direct GPU loading
                    )
                    pipeline = pipeline.to(self.device)
                else:
                    pipeline = FluxPipeline.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                    )

            elif model_type == 'sdxl':
                dtype = torch.float16 if self.device == 'cuda' else torch.float32
                pipeline = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    variant="fp16" if self.device == 'cuda' else None,
                    use_safetensors=True,
                )
                pipeline = pipeline.to(self.device)

            elif model_type == 'sd3':
                dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
                pipeline = StableDiffusion3Pipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                )
                pipeline = pipeline.to(self.device)

            elif model_type == 'zimage':
                pipeline = ZImagePipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False,
                )
                pipeline = pipeline.to(self.device)
                # Enable optimizations
                pipeline.transformer.set_attention_backend("flash")
                pipeline.transformer.compile()

            elif model_type == 'qwen':
                # Qwen-Image works with standard DiffusionPipeline - exact same as diffusers example
                if self.device.startswith('cuda'):
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float32

                print(f"[diffusiond] Loading Qwen model exactly like diffusers example: {self.device}")

                # Load exactly like the working diffusers example - no extra bullshit
                pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
                pipeline = pipeline.to(self.device)

                print(f"[diffusiond] Qwen loaded to {self.device} with {torch_dtype}")

            else:  # generic
                dtype = torch.float16 if self.device == 'cuda' else torch.float32
                pipeline = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                )
                pipeline = pipeline.to(self.device)

            self.loaded_models[model_name] = pipeline
            print(f"[diffusiond] Successfully loaded model: {model_name}")

        except Exception as e:
            print(f"[diffusiond] Failed to load model {model_name}: {e}")
            raise

    def _apply_mps_compat_patches(self):
        """Apply MPS compatibility patches for FLUX."""
        torch.set_default_dtype(torch.float32)
        original_from_numpy = torch.from_numpy
        def mps_safe_from_numpy(ndarray):
            tensor = original_from_numpy(ndarray)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor
        torch.from_numpy = mps_safe_from_numpy

        original_arange = torch.arange
        def mps_safe_arange(*args, **kwargs):
            if 'dtype' in kwargs and kwargs['dtype'] == torch.float64:
                kwargs['dtype'] = torch.float32
            return original_arange(*args, **kwargs)
        torch.arange = mps_safe_arange

    def generate_image(self, model_name: str, prompt: str, width: int = 1024, height: int = 1024,
                      num_inference_steps: Optional[int] = None, guidance_scale: Optional[float] = None) -> str:
        """Generate an image and return the file path."""
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found")

        # Load model if not already loaded
        self._load_model(model_name)

        pipeline = self.loaded_models[model_name]
        model_type = self.model_configs[model_name]['type']

        # Update usage timestamp
        self._update_model_usage(model_name)

        # Set default parameters based on model type
        if num_inference_steps is None:
            if model_type == 'flux':
                num_inference_steps = 4
            elif model_type == 'zimage':
                num_inference_steps = 9
            elif model_type == 'qwen':
                num_inference_steps = 50  # Qwen official example uses 50 steps
            else:
                num_inference_steps = 10

        if guidance_scale is None:
            if model_type in ['flux', 'zimage']:
                guidance_scale = 0.0
            elif model_type == 'qwen':
                guidance_scale = 4.0  # Qwen uses true_cfg_scale=4.0
            else:
                guidance_scale = 7.5

        print(f"[diffusiond] Generating image with {model_name}: {prompt[:50]}...")

        try:
            # Generate based on model type
            if model_type == 'zimage':
                image = pipeline(
                    prompt=prompt,
                    height=min(height, 768),  # Z-Image has size constraints
                    width=min(width, 768),
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=torch.Generator(self.device).manual_seed(42),
                ).images[0]
            elif model_type == 'flux':
                image = pipeline(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]
            elif model_type == 'qwen':
                # Qwen uses true_cfg_scale instead of guidance_scale
                image = pipeline(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    true_cfg_scale=guidance_scale,
                ).images[0]
            else:
                image = pipeline(
                    prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                ).images[0]

            # Save the image
            output_dir = "Generated_Art"
            os.makedirs(output_dir, exist_ok=True)

            # Create filename
            clean_prompt = re.sub(r'[^\w\s-]', '', prompt)
            clean_prompt = re.sub(r'\s+', '_', clean_prompt)
            if len(clean_prompt) > 30:
                clean_prompt = clean_prompt[:30].rstrip('_')
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{clean_prompt}_{ts}.png"
            filepath = os.path.join(output_dir, filename)

            image.save(filepath)
            print(f"[diffusiond] Image saved: {filepath}")

            # Clean up GPU memory after generation
            if self.device == 'cuda':
                torch.cuda.empty_cache()

            return filepath

        except Exception as e:
            print(f"[diffusiond] Generation failed: {e}")
            # Clean up GPU memory even on failure
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            raise

    def list_models(self) -> List[Dict]:
        """List available models."""
        models = []
        for name, config in self.model_configs.items():
            models.append({
                'name': name,
                'type': config['type'],
                'loaded': name in self.loaded_models,
                'path': config['path']
            })
        return models

    def unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            if self.device.startswith('cuda'):
                torch.cuda.empty_cache()
            print(f"[diffusiond] Unloaded model: {model_name}")


# Global server instance
server = None

def get_server() -> DiffusionServer:
    global server
    if server is None:
        server = DiffusionServer()
    return server

# FastAPI app
app = FastAPI(title="DiffusionD", description="Ollama-like server for diffusion models")

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/models")
async def list_models():
    server = get_server()
    return {"models": server.list_models()}

@app.post("/generate")
async def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    server = get_server()


    try:
        filepath = server.generate_image(
            model_name=request.model,
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale
        )

        # Clean up old images in background (keep last 100)
        background_tasks.add_task(cleanup_old_images)

        return {"image_path": filepath}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload/{model_name}")
async def unload_model(model_name: str):
    server = get_server()
    server.unload_model(model_name)
    return {"status": "unloaded"}

def cleanup_old_images():
    """Clean up old generated images, keeping the most recent 100."""
    output_dir = "Generated_Art"
    if not os.path.exists(output_dir):
        return

    images = []
    for f in os.listdir(output_dir):
        if f.endswith('.png'):
            path = os.path.join(output_dir, f)
            mtime = os.path.getmtime(path)
            images.append((path, mtime))

    # Sort by modification time, newest first
    images.sort(key=lambda x: x[1], reverse=True)

    # Remove older images beyond the limit
    for path, _ in images[100:]:
        try:
            os.remove(path)
            print(f"[diffusiond] Cleaned up old image: {os.path.basename(path)}")
        except:
            pass

def main():
    parser = argparse.ArgumentParser(description="DiffusionD - Ollama-like diffusion model server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--models-dir", help="Path to Diffusion_Models directory")
    parser.add_argument("--gpu", type=int, help="GPU ID to use (0, 1, 2, 3, etc.). If not specified, auto-selects GPU with most available memory")

    args = parser.parse_args()

    # Initialize server
    global server
    server = DiffusionServer(args.models_dir, args.gpu)

    print("🎨 DiffusionD Server Starting...")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   GPU: {server.device}")
    print(f"   Models: {len(server.model_configs)} found")
    print("   API Endpoints:")
    print("     GET  /health     - Health check")
    print("     GET  /models     - List available models")
    print("     POST /generate   - Generate image")
    print("     POST /unload/{model} - Unload model from memory")
    print()

    # Start server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
