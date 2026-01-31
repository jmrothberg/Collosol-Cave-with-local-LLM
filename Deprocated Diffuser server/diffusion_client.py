#!/usr/bin/env python3
"""
Diffusion Client - Simple client for DiffusionD server
Provides the same interface as the original diffusion code but uses the server.
"""

import requests
import json
import time
from typing import Optional, List, Dict
import os


class DiffusionClient:
    """Client for DiffusionD server that mimics the original diffusion interface."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000, model: str = None):
        self.base_url = f"http://{host}:{port}"
        self.model = model
        self._check_server()

    def _check_server(self):
        """Check if the server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException:
            raise ConnectionError(f"DiffusionD server not available at {self.base_url}")

    def list_models(self) -> List[Dict]:
        """List available models on the server."""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()["models"]

    def generate(self, prompt: str, width: int = 1024, height: int = 1024,
                num_inference_steps: Optional[int] = None,
                guidance_scale: Optional[float] = None) -> Optional[str]:
        """
        Generate an image using the diffusion server.
        Returns the file path to the generated image, or None if failed.
        """
        if not self.model:
            print("[diffusion_client] No model specified")
            return None

        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "width": width,
                "height": height
            }

            if num_inference_steps is not None:
                data["num_inference_steps"] = num_inference_steps
            if guidance_scale is not None:
                data["guidance_scale"] = guidance_scale

            response = requests.post(
                f"{self.base_url}/generate",
                json=data,
                timeout=300  # 5 minutes timeout for generation
            )
            response.raise_for_status()

            result = response.json()
            return result["image_path"]

        except requests.exceptions.RequestException as e:
            print(f"[diffusion_client] Request failed: {e}")
            return None
        except Exception as e:
            print(f"[diffusion_client] Generation failed: {e}")
            return None

    def unload_model(self, model_name: str = None):
        """Unload a model from server memory."""
        model = model_name or self.model
        if not model:
            return

        try:
            response = requests.post(f"{self.base_url}/unload/{model}")
            response.raise_for_status()
            print(f"[diffusion_client] Unloaded model: {model}")
        except requests.exceptions.RequestException as e:
            print(f"[diffusion_client] Failed to unload model: {e}")


# Global client instance for backwards compatibility
_image_gen_client = None

def get_diffusion_client(host: str = "127.0.0.1", port: int = 8000, model: str = None) -> DiffusionClient:
    """Get or create a diffusion client instance."""
    global _image_gen_client
    if _image_gen_client is None:
        _image_gen_client = DiffusionClient(host=host, port=port, model=model)
    elif model and _image_gen_client.model != model:
        _image_gen_client.model = model
    return _image_gen_client


# Backwards compatibility: mimic the original ImageGenerator class
class ImageGenerator:
    """Drop-in replacement for the original ImageGenerator class."""

    def __init__(self, model_id: str = None, device: str = "auto", local_root: str = None):
        self.model_id = model_id
        self.device = device
        self.local_root = local_root

        # Try to connect to diffusion server
        try:
            self.client = DiffusionClient(model=model_id)
            self._available = True
            print(f"[image] Connected to DiffusionD server with model: {model_id}")
        except ConnectionError:
            print("[image] DiffusionD server not available, falling back to local generation")
            self._available = False
            self.client = None

    def generate(self, prompt: str) -> Optional[str]:
        """Generate an image. Returns file path or None."""
        if not self._available or not self.client:
            print("[image] Diffusion server not available")
            return None

        return self.client.generate(prompt)

    def _lazy_init(self):
        """Backwards compatibility - no-op since server handles this."""
        pass

    def cleanup(self):
        """Backwards compatibility - cleanup method expected by original code."""
        pass


# Utility functions
def list_available_models(host: str = "127.0.0.1", port: int = 8000) -> List[str]:
    """List available models on the diffusion server."""
    try:
        client = DiffusionClient(host=host, port=port)
        models = client.list_models()
        return [m["name"] for m in models]
    except Exception:
        return []


def wait_for_server(host: str = "127.0.0.1", port: int = 8000, timeout: int = 30) -> bool:
    """Wait for the diffusion server to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            client = DiffusionClient(host=host, port=port)
            return True
        except ConnectionError:
            time.sleep(1)
    return False


if __name__ == "__main__":
    # Test the client
    import sys

    if len(sys.argv) < 2:
        print("Usage: python diffusion_client.py <model_name> [prompt]")
        sys.exit(1)

    model = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "a beautiful landscape"

    try:
        client = DiffusionClient(model=model)
        print(f"Generating image with model '{model}': {prompt}")
        result = client.generate(prompt)
        if result:
            print(f"Image generated: {result}")
        else:
            print("Generation failed")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
