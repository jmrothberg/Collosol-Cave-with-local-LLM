# JMR's LLM Adventure Game Engine
# December 4, 2025 — Updated to MLX-LM + MFLUX (Feb 2026)
#
# This is a fully LLM-driven adventure game where the Large Language Model
# acts as the game master, storyteller, and world engine. The LLM has complete
# creative control while using structured JSON to maintain consistent game state.
#
# DEPENDENCIES:
# - MLX-LM (for LLM models on Apple Silicon): pip install mlx-lm
# - MFLUX (for FLUX image generation on Apple Silicon): pip install mflux
# - Gradio (for web UI): pip install gradio
# - Local FLUX models in ~/Diffusion_Models/ (e.g. FLUX2-klein-9B-mlx-8bit)
# - Optional Ollama (same machine): pip install ollama — LLM dropdown includes ollama:<tag> entries
#
# The LLM is instructed via system prompt to use these JSON tools, making this
# a powerful general-purpose adventure engine that can run ANY adventure the
# LLM can imagine, while maintaining proper game state through function calls.

import os
import argparse
import json
import pickle
import platform
import sys
import time
import zipfile
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
import re
import gradio as gr

from mlx_lm import load as mlx_load, generate as mlx_generate
from mlx_lm.sample_utils import make_sampler

# MLX model discovery
MLX_MODELS_DIR = "/Users/jonathanrothberg/MLX_Models"

# Ollama: dropdown values use this prefix so tags like "qwen3.6:27b-coding-mxfp8" never collide with MLX folder names.
OLLAMA_CHOICE_PREFIX = "ollama:"
# CHANGE: default Ollama tag for this workflow (must match `ollama list` after `ollama pull …`).
OLLAMA_PREFERRED_MODEL = "qwen3.6:27b-coding-mxfp8"

try:
    import ollama as ollama_pkg  # type: ignore
except ImportError:
    ollama_pkg = None

from mflux_image_gen import MfluxImageGenerator

"""
FLEXIBLE GAMEPLAY PHILOSOPHY:
- The LLM is the creative engine, not hard-coded functions
- World bible provides consistency (NPCs, monsters, riddles, objectives)
- LLM handles all dynamic gameplay:
  * NPC dialogue and personalities
  * Combat through descriptive narration and health changes
  * Riddle/puzzle presentation and solving
  * Emergent storytelling based on player choices
- As LLMs improve, the game automatically gets better
- No hard-coded combat system, dialogue trees, or puzzle mechanics
- The world bible is just a guide - LLM can expand creatively

MEMORY ARCHITECTURE FOR LLM-POWERED GAMEPLAY:
==============================================
The game uses TWO complementary memory structures that the LLM sees:

1. WORLD_BIBLE (Static Game Design)
   - NPCs, monsters, objectives, riddles
   - Story arc and win conditions
   - Theme and atmosphere
   - This is the "game design document" - rarely changes

2. GAME_STATE (Dynamic Game Memory)
   - Player status: health, inventory, location
   - World state: known_map, room_items, notes
   - Image cache: rooms_with_images, items_with_images
   - This changes constantly as the game progresses

HOW THE LLM USES THESE MEMORIES:
- Gets BOTH as context in every prompt
- Uses WORLD_BIBLE for consistency (e.g., "torch defeats troll")
- Uses GAME_STATE for current situation (e.g., "player has torch")
- Makes intelligent decisions based on complete picture
- As LLMs improve, they leverage this data more effectively

IMAGE CACHING & REUSE SYSTEM:
- Pregenerated images are tracked in rooms_with_images/items_with_images
- Dynamic images (generated during play) use the SAME tracking
- When returning to a location/item, system checks these lists
- If present -> reuse, if not -> generate and add to list
- This works across save/load because lists are persisted
"""

###############################################################################
# LLM-POWERED ADVENTURE GAME ENGINE WITH JSON STATE MANAGEMENT
#
# Core Concept:
# This is a FULLY LLM-DRIVEN adventure game where the Large Language Model
# acts as the game master, storyteller, and world engine. The LLM has complete
# creative control while using structured JSON to maintain consistent game state.
#
# How It Works:
# 1. The LLM generates narrative responses to player actions
# 2. The LLM emits JSON directives to update game state (hidden from player)
# 3. Helper functions process the JSON to maintain canonical game state
# 4. The system tracks inventory, map, health, and generates images automatically
#
# JSON-Based Function Calls (processed automatically):
# The LLM outputs JSON blocks that trigger state management functions:
# - state_updates.move_to: str                -> move player to a room
# - state_updates.connect: [[roomA, roomB]]   -> create bidirectional exits
# - state_updates.add_items: [str, ...]       -> add items to inventory
# - state_updates.remove_items: [str, ...]    -> remove items from inventory
# - state_updates.place_items: [str, ...]     -> place items in current room
# - state_updates.room_take: [str, ...]       -> transfer items room->inventory
# - state_updates.change_health: int          -> modify HP (clamped 0-100)
# - state_updates.add_note: str               -> append to quest log
# - state_updates.set_context: str            -> update narrative memory (story_context)
# - state_updates.set_flag: {name,value}      -> persist a puzzle/condition in game_flags
# - state_updates.mechanics: {action,effect}  -> record simple cause-effect rule
# - images: [str, ...]                        -> generate images for rooms/items
#
# Key Features:
# - UNLIMITED CREATIVITY: The LLM can create any story, any world, any items
# - CONSISTENT STATE: JSON directives ensure game state remains coherent
# - AUTOMATIC GRAPHICS: Images generated for new rooms and important items
# - CLEAN NARRATION: JSON is extracted and processed, never shown to player
# - PERSISTENT WORLD: Map, inventory, and notes tracked across the session
#
# The LLM is instructed via system prompt to use these JSON tools, making this
# a powerful general-purpose adventure engine that can run ANY adventure the
# LLM can imagine, while maintaining proper game state through function calls.
###############################################################################


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(_SCRIPT_DIR, "Adventure_Game_Saved")
ART_DIR = os.path.join(_SCRIPT_DIR, "Adventure_Art")
IN_GAME_IMG_DIR = os.path.join(SAVE_DIR, "in_game_images")

# Optional world bible (generated once with a heavier LLM) to guide fast runtime play
# Format: compact JSON with keys: objectives, acts, key_characters, locations,
# riddles_and_tasks, loot_progression, fail_states, dynamic_rules
WORLD_BIBLE: Optional[Dict[str, Any]] = None

# Default image theme used when no custom theme is specified
DEFAULT_IMAGE_THEME = (
    "Studio Ghibli background art, dark fantasy cavern, painterly matte painting, "
    "soft rim light, volumetric fog, by Studio Ghibli"
)

# Preset themes for the dropdown - each theme should be a complete description
# that guides both the world bible generation AND the image generation
PRESET_THEMES = {
    "Tolkien Cave Adventure": (
        "An adventure game starting at the entrance to a giant cave system, "
        "featuring Tolkien-inspired characters like wizards, hobbits, dwarves, and elves. "
        "Each character and object should play a meaningful part in completing the quest. "
        "Art style: detailed fantasy illustration with warm torchlight, ancient stone architecture, "
        "and atmospheric mist. Environments should feel lived-in and mysterious."
    ),
    "Star Wars Universe": (
        "An adventure game set in the Star Wars universe with iconic characters like Jedi, "
        "droids, and aliens. Features lightsabers, the Force, and galactic technology. "
        "Each character and item should be essential to completing the mission. "
        "Art style: cinematic sci-fi with neon glows, metallic surfaces, holographic displays, "
        "and dramatic lighting contrasts between light and dark side aesthetics."
    ),
    "Colossal Cave (Classic)": (
        "A classic text adventure inspired by the original Colossal Cave Adventure. "
        "Features underground caverns, twisty passages, treasure hunting, and puzzle solving. "
        "Characters include a helpful hermit, mysterious spirits, and cave-dwelling creatures. "
        "Art style: nostalgic fantasy illustration with glowing crystals, underground rivers, "
        "ancient ruins, and atmospheric cave lighting."
    ),
    "Zork-style Mystery": (
        "A mysterious adventure in the style of Zork, featuring a sprawling underground empire, "
        "ancient artifacts, and clever puzzles. Characters are enigmatic and locations are interconnected. "
        "Art style: dark fantasy with ornate architecture, magical glows, brass and copper machinery, "
        "and deep shadows hiding secrets."
    ),
    "Myst Island": (
        "A puzzle-focused adventure on a mysterious island with strange machines and hidden passages. "
        "Minimal characters but rich environmental storytelling. Each mechanism and clue connects to the solution. "
        "Art style: surreal landscapes, steampunk machinery, crystalline structures, "
        "ethereal lighting, and dreamlike atmosphere."
    ),
    "Pirate Treasure Hunt": (
        "A swashbuckling adventure seeking buried treasure on a tropical island. "
        "Features pirates, sea caves, ancient maps, and hidden troves. "
        "Characters include a helpful parrot, ghostly pirates, and island natives. "
        "Art style: vibrant tropical colors, weathered wood and rope, golden treasure glints, "
        "moonlit beaches, and mysterious jungle ruins."
    ),
    "Cyberpunk Heist": (
        "A high-tech adventure in a neon-lit dystopian city. "
        "Features hackers, corporate secrets, AI companions, and cybernetic enhancements. "
        "Each contact and gadget serves a purpose in the heist. "
        "Art style: neon-soaked cityscapes, holographic interfaces, rain-slicked streets, "
        "chrome and glass architecture, and dramatic urban lighting."
    ),
    "Custom (type below)": ""
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION FLAGS
# ═══════════════════════════════════════════════════════════════════════════════

# When False (default), use simplified prompts for medium-sized models (≤40B).
# When True, include timer_event, conditional_action, chain_reaction, mechanics.
ADVANCED_DIRECTIVES = False

# CHANGE (chunk D): Verbose tool logs are noisy. By default the UI only shows the per-turn
# [SUMMARY] line plus any [WARN]/[WARNING] lines from apply_llm_directives. Set the env var
# ADV_VERBOSE_DEBUG=1 to see every tool.* entry (room_take, place_item, etc.) — useful when
# debugging a specific bad turn.
ADV_VERBOSE_DEBUG = os.environ.get("ADV_VERBOSE_DEBUG", "").lower() in {"1", "true", "yes"}

# CHANGE (chunk O): engine-note injection into the next user prompt is OFF by default.
# 30B-class local models tend to narrate better when not constantly nagged with
# "fix this now" notes. The notes are still emitted into [SUMMARY] / debug for the
# user. Set ADV_INJECT_ENGINE_NOTES=1 to feed them to the model on the next turn
# (useful when actively diagnosing why a model keeps making the same mistake).
ADV_INJECT_ENGINE_NOTES = os.environ.get("ADV_INJECT_ENGINE_NOTES", "").lower() in {"1", "true", "yes"}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _get_world_bible() -> Optional[Dict[str, Any]]:
    """Safe accessor for the global WORLD_BIBLE. Returns None if unset or invalid."""
    global WORLD_BIBLE
    return WORLD_BIBLE if isinstance(WORLD_BIBLE, dict) else None


def _is_small_model(model_id: str) -> bool:
    """Heuristic: models roughly ≤40B are 'small' and get simplified prompts."""
    if not model_id:
        return True
    lower = model_id.lower()
    small_indicators = [
        '4b', '7b', '8b', '12b', '14b', '22b', '27b',
        '32b', '35b', 'qwen', 'gemma', 'phi', 'mistral-7',
    ]
    return any(ind in lower for ind in small_indicators)


# CHANGE (chunk D): minimum-but-rich debug helpers — produce one [SUMMARY] line per turn
# and let callers filter the rest down to warnings unless ADV_VERBOSE_DEBUG is on.
def _format_turn_summary(
    prev: Dict[str, Any],
    post: Dict[str, Any],
    warning_count: int,
    payload_count: int,
) -> str:
    """Single line capturing what changed this turn — keep it tight and act-on-able."""
    inv_prev = set(prev.get("inventory", []))
    inv_post = set(post.get("inventory", []))
    inv_added = sorted(inv_post - inv_prev)
    inv_removed = sorted(inv_prev - inv_post)

    flags_prev = set(prev.get("flags", []))
    flags_post = set(post.get("flags", []))
    flags_added = sorted(flags_post - flags_prev)

    loc_prev = prev.get("location") or "?"
    loc_post = post.get("location") or "?"
    loc_part = f"loc={loc_post}" if loc_prev == loc_post else f"loc={loc_prev}->{loc_post}"

    hp_prev = prev.get("health", 0)
    hp_post = post.get("health", 0)
    hp_part = f"hp={hp_post}" if hp_prev == hp_post else f"hp={hp_prev}->{hp_post}"

    inv_part = ""
    if inv_added:
        inv_part += f" inv+={','.join(inv_added)}"
    if inv_removed:
        inv_part += f" inv-={','.join(inv_removed)}"
    flag_part = f" flags+={','.join(flags_added)}" if flags_added else ""

    return (
        f"[SUMMARY] {loc_part} {hp_part}"
        f"{inv_part}{flag_part}"
        f" json={payload_count} warns={warning_count}"
    )


def _filter_tool_logs_for_display(tool_logs: List[str], verbose: bool) -> List[str]:
    """Return only summary + warning lines unless verbose is on (chunk D)."""
    if verbose:
        return list(tool_logs)
    keep_prefixes = ("[SUMMARY]", "[WARN]", "[WARNING]")
    return [ln for ln in tool_logs if any(ln.startswith(p) for p in keep_prefixes)]


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE GENERATION & CACHING
# ═══════════════════════════════════════════════════════════════════════════════

def ensure_directories_exist() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(IN_GAME_IMG_DIR, exist_ok=True)


def get_theme_suffix() -> str:
    """
    Get the current theme suffix for image generation.
    Returns WORLD_BIBLE["global_theme"] if set (via UI), otherwise default theme.
    """
    try:
        wb = _get_world_bible()
        theme_suffix = wb.get("global_theme") if wb else None
    except Exception:
        theme_suffix = None
    
    if not theme_suffix:
        theme_suffix = DEFAULT_IMAGE_THEME
    
    return theme_suffix


def execute_world_bible_generation(state_mgr: "StateManager", model_id: str, theme: Optional[str] = None, max_tokens: int = 4000) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Shared world bible generation logic.
    Returns (success, message, world_bible_data).
    """
    global WORLD_BIBLE
    
    # Handle auto model selection
    if model_id == "(auto)":
        heavy_model_id = select_llm_interactively()
    else:
        heavy_model_id = model_id

    if not heavy_model_id:
        return False, "No heavy model selected for world bible generation.", None

    # CHANGE: Ollama-backed world bible does not require MLX_MODELS_DIR; MLX-only path still does.
    if llm_choice_is_ollama(heavy_model_id):
        if ollama_pkg is None:
            return False, "Ollama backend selected but Python package missing: pip install ollama", None
    elif not check_mlx_available():
        return False, "MLX not available. Cannot generate world bible.", None

    print(f"Generating world bible using model: {heavy_model_id}")
    result = generate_world_bible(state_mgr, heavy_model_id, theme, max_tokens)
    
    if result:
        WORLD_BIBLE = result
        return True, "World bible generated successfully.", WORLD_BIBLE
    else:
        return False, "Failed to generate world bible.", None


# Image generation rules to prevent hallucinated content
IMAGE_RULES = (
    "ONLY show objects, characters, and elements explicitly described in the scene. "
    "Do NOT add people, creatures, or characters unless specifically mentioned. "
    "Empty rooms should appear empty. Solitary scenes should show no other beings. "
    "Focus on environmental details, lighting, and atmosphere rather than adding figures."
)


def generate_themed_image(image_gen: "MfluxImageGenerator", base_prompt: str, theme_suffix: Optional[str] = None) -> Optional[str]:
    """
    Generate an image with theme applied consistently.
    Returns the generated image path or None if generation fails.
    """
    if not image_gen:
        return None

    if not theme_suffix:
        theme_suffix = get_theme_suffix()

    # Build the final prompt with theme and anti-hallucination rules
    parts = [base_prompt]
    
    if theme_suffix and theme_suffix.lower() not in base_prompt.lower():
        parts.append(f"Art style: {theme_suffix}")
    
    # Always add the image rules to prevent extra characters
    parts.append(IMAGE_RULES)
    
    themed_prompt = ". ".join(parts)

    return image_gen.generate(themed_prompt)


def extract_narrative_elements(narrative: str, room_name: str = "room") -> str:
    """
    Extract only concrete objects, people, and environmental elements explicitly mentioned in the narrative.
    This prevents image generation from hallucinating content that's not actually in the story.
    """
    # Convert to lowercase for easier matching
    text = narrative.lower()

    # Define categories of elements to extract
    elements = []

    # Check if the narrative explicitly says we're alone or no one is here
    alone_phrases = [
        "no one", "nobody", "alone", "empty", "deserted", "abandoned",
        "solitary", "by yourself", "no visible", "no other", "just you"
    ]
    is_alone = any(phrase in text for phrase in alone_phrases)

    # Environmental descriptors (concrete, not abstract)
    env_words = [
        'stone', 'wall', 'floor', 'ceiling', 'door', 'console', 'pedestal', 'panel',
        'corridor', 'passage', 'chamber', 'room', 'tunnel', 'glow', 'light', 'shadow',
        'dust', 'metal', 'machine', 'machinery', 'humming', 'console', 'datapad',
        'draft', 'air', 'echo', 'vibration', 'seam', 'bulkhead', 'ventilation',
        'cave', 'cavern', 'rock', 'crystal', 'water', 'river', 'pool', 'torch',
        'lantern', 'candle', 'fire', 'flame', 'smoke', 'mist', 'fog'
    ]

    # Object descriptors
    object_words = [
        'console', 'pedestal', 'panel', 'datapad', 'button', 'lever', 'switch',
        'door', 'gate', 'portal', 'screen', 'display', 'terminal', 'control',
        'keypad', 'lock', 'mechanism', 'device', 'artifact', 'relic',
        'chest', 'box', 'crate', 'barrel', 'table', 'chair', 'throne',
        'altar', 'statue', 'pillar', 'column', 'arch', 'bridge'
    ]

    # Lighting/color descriptors (tied to concrete elements)
    lighting_words = [
        'blue glow', 'faint blue', 'dimly lit', 'flickering', 'red button',
        'humming glow', 'eerie shadows', 'cool draft', 'distant hum',
        'golden light', 'green glow', 'purple haze', 'orange flame'
    ]

    # Check for environmental elements
    for word in env_words:
        if word in text:
            elements.append(word)

    # Check for objects
    for word in object_words:
        if word in text:
            elements.append(word)

    # Check for specific lighting/color combinations
    for phrase in lighting_words:
        if phrase.replace(' ', '') in text.replace(' ', '') or phrase in text:
            elements.append(phrase)

    # Remove duplicates and join
    unique_elements = list(set(elements))

    if unique_elements:
        # Create a focused prompt with the room name and extracted elements
        room_part = f"{room_name}"
        elements_part = ", ".join(unique_elements[:8])  # Limit to avoid token bloat
        base = f"{room_part} with {elements_part}"
    else:
        # Fallback if no specific elements found
        base = f"{room_name}, adventure game environment"

    # Add explicit "empty, no people" if narrative says we're alone
    if is_alone:
        base += ", empty scene, no people, no characters, solitary environment"

    return base


def generate_room_image_if_needed(state_mgr: "StateManager", image_gen: Optional["MfluxImageGenerator"], room_name: str, debug_lines: Optional[List[str]] = None, narrative_context: Optional[str] = None) -> Optional[str]:
    """
    Generate a room image only if one doesn't already exist.
    Returns the image path (new or existing) or None.
    
    Args:
        narrative_context: Optional description from LLM narration to use for the image prompt
    """
    if not image_gen:
        return None
    
    # Check if room already has an image
    # If we have narrative context, always generate a new image to match the current state
    if state_mgr.state.has_room_image(room_name) and not narrative_context:
        state_mgr.state.images_reused += 1

        # Find and return the existing image path
        for img_data in state_mgr.state.last_images:
            if img_data.get("type") == "room" and img_data.get("subject") == room_name:
                existing_path = img_data.get("path")
                if existing_path and os.path.exists(existing_path):
                    if debug_lines:
                        debug_lines.append(f"Reuse existing image for room: {room_name} → {os.path.basename(existing_path)}")
                    return existing_path

        # Image tracked but file not found
        if debug_lines:
            debug_lines.append(f"WARNING: Room {room_name} marked as having image but file not found!")
        return None
    
    # Build image prompt from narrative context if available, otherwise use world bible or generic
    if narrative_context and len(narrative_context) > 20:
        # Extract only objects/people explicitly mentioned in the narrative
        # This prevents hallucinated content in images
        base_prompt = extract_narrative_elements(narrative_context, room_name)
        if debug_lines:
            debug_lines.append(f"Extracted narrative elements for image: {base_prompt[:50]}...")
    else:
        # Try to get description from world bible
        wb_desc = None
        try:
            wb = _get_world_bible()
            if wb:
                for loc in wb.get("locations", []):
                    if isinstance(loc, dict) and loc.get("name", "").lower() == room_name.lower():
                        wb_desc = loc.get("description")
                        break
        except Exception:
            pass

        if wb_desc:
            base_prompt = f"{room_name}: {wb_desc}"
            if debug_lines:
                debug_lines.append(f"Using world bible description for image: {base_prompt[:50]}...")
        else:
            base_prompt = f"atmospheric {room_name.lower()}, detailed environment, adventure game location"
    
    image_path = generate_themed_image(image_gen, base_prompt)
    
    if image_path:
        # Track the new image
        state_mgr.state.add_room_image(room_name, image_path, base_prompt)
        if debug_lines:
            debug_lines.append(f"Generated new image for room: {room_name}")
    
    return image_path


def generate_item_image_if_needed(state_mgr: "StateManager", image_gen: Optional["MfluxImageGenerator"], item_name: str, debug_lines: Optional[List[str]] = None) -> Optional[str]:
    """
    Generate an item image only if one doesn't already exist.
    Returns the image path (new or existing) or None.
    """
    if not image_gen:
        return None
    
    # Check if item already has an image
    if state_mgr.state.has_item_image(item_name):
        state_mgr.state.images_reused += 1
        
        # Find and return the existing image path
        for img_data in state_mgr.state.last_images:
            if img_data.get("type") == "item" and img_data.get("subject") == item_name:
                existing_path = img_data.get("path")
                if existing_path and os.path.exists(existing_path):
                    if debug_lines:
                        debug_lines.append(f"Reuse existing image for item: {item_name} → {os.path.basename(existing_path)}")
                    return existing_path
        
        # Image tracked but file not found
        if debug_lines:
            debug_lines.append(f"WARNING: Item {item_name} marked as having image but file not found!")
        return None
    
    # Generate new item image
    base_prompt = f"detailed close-up of {item_name}, fantasy adventure game item, magical artifact"
    image_path = generate_themed_image(image_gen, base_prompt)
    
    if image_path:
        # Track the new image
        state_mgr.state.add_item_image(item_name, image_path, base_prompt)
        if debug_lines:
            debug_lines.append(f"Generated new image for item: {item_name}")
    
    return image_path


def _world_bible_reference_for(subject_name: str, subject_type: str) -> Optional[Dict[str, Any]]:
    """
    CHANGE: Trivial helper to link images to WORLD_BIBLE objects by name.
    Returns a compact reference dict with section and index if found.
    Safe if WORLD_BIBLE is missing or structure varies.
    """
    wb = _get_world_bible() or {}
    name_l = (subject_name or "").strip().lower()
    if not name_l or not isinstance(wb, dict):
        return None
    # Locations (rooms)
    try:
        locations = wb.get("locations", []) or []
        for idx, loc in enumerate(locations):
            nm = (loc.get("name") if isinstance(loc, dict) else str(loc))
            if isinstance(nm, str) and nm.strip().lower() == name_l:
                return {"section": "locations", "index": idx, "name": nm}
    except Exception:
        pass
    # Key items
    try:
        key_items = wb.get("key_items", []) or []
        for idx, it in enumerate(key_items):
            nm = (it.get("name") if isinstance(it, dict) else str(it))
            if isinstance(nm, str) and nm.strip().lower() == name_l:
                return {"section": "key_items", "index": idx, "name": nm}
    except Exception:
        pass
    # NPCs, monsters, riddles (future-proof if we generate these images later)
    for sec in ("npcs", "monsters", "riddles"):
        try:
            arr = wb.get(sec, []) or []
            for idx, obj in enumerate(arr):
                nm = (obj.get("name") if isinstance(obj, dict) else str(obj))
                if isinstance(nm, str) and nm.strip().lower() == name_l:
                    return {"section": sec, "index": idx, "name": nm}
        except Exception:
            continue
    return None

def get_all_image_paths_from_state(state_mgr: "StateManager") -> List[str]:
    """
    Extract all image paths from the game state for gallery display.
    Returns a list of image file paths that exist on disk.
    """
    image_paths = []
    
    # Get all image paths from last_images
    for img_data in state_mgr.state.last_images:
        path = img_data.get("path")
        if path and os.path.exists(path):
            image_paths.append(path)
    
    return image_paths


def populate_gallery_from_state(state_mgr: "StateManager", ui_data: Dict[str, Any]) -> None:
    """
    Populate the UI gallery with all existing images from game state.
    This is called after loading games or pregeneration to show existing images.
    """
    existing_images = get_all_image_paths_from_state(state_mgr)
    # Deduplicate image paths while preserving order
    seen_paths = set()
    deduped = []
    for p in existing_images:
        if p not in seen_paths:
            seen_paths.add(p)
            deduped.append(p)
    ui_data["images"] = deduped
    if existing_images:
        print(f"[gallery] Populated with {len(deduped)} existing images (deduped)")


def get_current_room_image(state_mgr: "StateManager") -> Optional[str]:
    """Get the image for the current room. Simple and fast."""
    current_room = state_mgr.state.location
    
    # Find room image - exact match like inventory
    for img_data in state_mgr.state.last_images:
        if img_data.get("type") == "room" and img_data.get("subject") == current_room:
            path = img_data.get("path")
            if path and os.path.exists(path):
                return path
    
    return None


def get_inventory_item_image(state_mgr: "StateManager") -> Optional[str]:
    """Get image for most recent inventory item. Simple like room images."""
    if not state_mgr.state.inventory:
        return None
    
    # Check last few inventory items for images
    for item_name in reversed(state_mgr.state.inventory[-3:]):
        for img_data in state_mgr.state.last_images:
            if img_data.get("type") == "item" and img_data.get("subject") == item_name:
                path = img_data.get("path")
                if path and os.path.exists(path):
                    return path
    
    return None


def get_image_for_subject(state_mgr: "StateManager", subject_name: str) -> Optional[str]:
    """Return cached image path for a room or item subject name (case-insensitive)."""
    if not subject_name:
        return None
    try:
        target = subject_name.strip().lower()
        for img_data in state_mgr.state.last_images:
            if img_data.get("type") in ("room", "item"):
                if str(img_data.get("subject", "")).strip().lower() == target:
                    p = img_data.get("path")
                    if p and os.path.exists(p):
                        return p
    except Exception:
        return None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# GAME STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GameState:
    """
    GAME STATE MEMORY STRUCTURE - The Dynamic Game Memory
    =====================================================
    This is the LIVE, CHANGING memory that tracks everything happening in the game.
    The LLM gets this entire state as context, allowing it to make intelligent decisions.
    As LLMs improve, they can leverage this data more effectively for better gameplay.
    
    MEMORY ARCHITECTURE:
    1. WORLD_BIBLE (separate, static) = Game design/structure (NPCs, objectives, story arc)
    2. GAME_STATE (this class) = Everything that changes during play
    3. LLM sees both and makes decisions based on the complete picture
    """
    
    # === CORE PLAYER STATE (belongs in game state - changes during play) ===
    # The LLM uses these to understand player status and make appropriate responses
    player_name: str = "Adventurer"  # Who the player is
    location: str = "Start"          # Where player currently is (key into known_map)
    health: int = 100                # Combat/danger tracking - LLM decides consequences
    inventory: List[str] = field(default_factory=list)  # What player carries - LLM enforces item logic
    
    # === DYNAMIC WORLD STATE (belongs in game state - discovered/changed during play) ===
    # The LLM uses these to maintain world consistency and track exploration
    
    # Items present in rooms: room -> [items]
    # LLM checks this to know what's available in each location
    room_items: Dict[str, List[str]] = field(default_factory=dict)
    
    # Known map: room -> {"exits": [room, ...], "notes": str}
    # LLM uses this for navigation and room descriptions
    # This grows as player explores - LLM adds new rooms dynamically
    known_map: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Start": {"exits": [], "notes": "A nondescript starting place."}
    })
    
    # Free-form notes or quest log - LLM adds important events/clues here
    notes: List[str] = field(default_factory=list)
    
    # === CONVERSATION MEMORY (for NPC continuity) ===
    # Keep last 5 exchanges for context (player input + LLM response)
    # This allows NPCs to remember recent conversation without bloating context
    recent_history: List[Dict[str, str]] = field(default_factory=list)
    
    # === STORY CONTEXT (LLM's narrative memory) ===
    # The LLM can update this to track narrative threads, foreshadowing, and story intentions
    # As LLMs improve, they'll use this more intelligently for better storytelling
    story_context: str = ""
    
    # === GAME FLAGS (LLM-defined states) ===
    # Flexible flags the LLM can set for any purpose (puzzles, states, timers, etc.)
    game_flags: Dict[str, Any] = field(default_factory=dict)

    # === ENGINE NOTES (chunk J) ===
    # Warnings produced by the engine on the previous turn (e.g. room_take MISS,
    # add+remove auto-corrected). These are surfaced into the NEXT user prompt so
    # the LLM stops repeating the same mistake within one session. Cleared after use.
    engine_notes_for_next_turn: List[str] = field(default_factory=list)

    # === SESSION COUNTERS (chunk K post-mortem) ===
    # Aggregated tallies that drive the end-of-session report. Strictly grow.
    session_turns: int = 0
    warnings_history: List[str] = field(default_factory=list)
    chain_steps_completed: List[int] = field(default_factory=list)

    # === RUNTIME PROMPT TIGHTENING (chunk N) ===
    # We only dump full world-bible cues for a room on the FIRST turn the player is
    # there. After that the LLM has the description in its recent_history. Reset to
    # "" on death/restart so the new run gets a full opening.
    last_described_room: str = ""

    # === VISITED MAP (UI fix, May 2026) ===
    # known_map contains EVERY world-bible room (seeded at world-bible load time so
    # the engine can validate connectivity, place items, etc.). But the player-facing
    # "Known map" should only show rooms the player has actually been to. Track the
    # visited set here; move_to() appends. Order preserved for display.
    visited_rooms: List[str] = field(default_factory=list)
    
    # === IMAGE CACHING SYSTEM (belongs in game state - prevents regeneration) ===
    # CRITICAL FOR EFFICIENCY: Tracks which locations/items already have images
    # Works for BOTH pregenerated AND dynamically generated images
    
    # Complete history of all images (for reference/debugging)
    last_images: List[Dict[str, str]] = field(default_factory=list)
    
    # SIMPLE TRACKING LISTS - these are what prevent duplicate generation!
    # When we visit "Start" again, LLM checks: "Start" in rooms_with_images? Yes -> no new image
    rooms_with_images: List[str] = field(default_factory=list)  # ["Start", "Cave", "Forest"]
    items_with_images: List[str] = field(default_factory=list)  # ["Torch", "Sword", "Key"]
    
    # Future LLM enhancement: track what player has done recently
    recent_actions: List[str] = field(default_factory=list)
    
    # === GAME PERFORMANCE METRICS (belongs in game state - tracks optimization) ===
    # These help us measure how well the caching/reuse is working
    images_generated: int = 0  # Total images created
    images_reused: int = 0     # Times we avoided regeneration
    turns_played: int = 0      # Game length tracking
    
    """
    IMAGE REUSE VERIFICATION FLOW:
    ==============================
    1. PREGENERATION: Generate images for "Start", "Cave", "Forest"
       -> Calls add_room_image() for each
       -> rooms_with_images = ["Start", "Cave", "Forest"]
    
    2. GAMEPLAY START: Player in "Start"
       -> Check: has_room_image("Start") = True
       -> Result: NO new image, reuse pregenerated
    
    3. MOVE TO NEW ROOM: Player moves to "Dungeon" (not pregenerated)
       -> Check: has_room_image("Dungeon") = False
       -> Result: Generate new image, call add_room_image("Dungeon")
       -> rooms_with_images = ["Start", "Cave", "Forest", "Dungeon"]
    
    4. RETURN TO OLD ROOM: Player returns to "Start"
       -> Check: has_room_image("Start") = True
       -> Result: NO new image, reuse existing
    
    5. RETURN TO DYNAMIC ROOM: Player returns to "Dungeon"
       -> Check: has_room_image("Dungeon") = True
       -> Result: NO new image, reuse the one we generated in step 3
    
    This same flow works for items with has_item_image() and add_item_image()!
    """

    def add_room_image(self, room_name: str, image_path: str, prompt: str) -> None:
        """
        ROOM IMAGE TRACKING - Called when ANY room image is generated
        ============================================================
        Works for BOTH:
        1. Pregenerated images (created at game start)
        2. Dynamic images (created when entering new rooms)
        
        REUSE LOGIC:
        - First visit to "Cave" -> generates image, calls this method
        - Return to "Cave" later -> has_room_image("Cave") returns True -> NO regeneration
        - This works even across save/load because rooms_with_images is persisted
        """
        # Add world bible reference for debugging
        wb_ref = _world_bible_reference_for(room_name, "room")
        self.last_images.append({"prompt": prompt, "path": image_path, "type": "room", "subject": room_name, "wb_ref": wb_ref})
        if room_name not in self.rooms_with_images:
            self.rooms_with_images.append(room_name)
        self.images_generated += 1
    
    def add_item_image(self, item_name: str, image_path: str, prompt: str) -> None:
        """
        ITEM IMAGE TRACKING - Called when ANY item image is generated  
        ===========================================================
        Works for BOTH:
        1. Pregenerated item images
        2. Dynamic images (when player finds/examines items)
        
        REUSE LOGIC:
        - Find "Sword" first time -> generates image, calls this method
        - Drop and pick up "Sword" -> has_item_image("Sword") returns True -> NO regeneration
        - See "Sword" in different room -> still reuses same image
        """
        # Add world bible reference for debugging
        wb_ref = _world_bible_reference_for(item_name, "item")
        self.last_images.append({"prompt": prompt, "path": image_path, "type": "item", "subject": item_name, "wb_ref": wb_ref})
        if item_name not in self.items_with_images:
            self.items_with_images.append(item_name)
        self.images_generated += 1
    
    def has_room_image(self, room_name: str) -> bool:
        """
        CHECK FOR EXISTING ROOM IMAGE - Prevents regeneration
        =====================================================
        Called before generating any room image to check if we already have one.
        Returns True if this room has EVER had an image (pregenerated OR dynamic).
        """
        return room_name in self.rooms_with_images
    
    def has_item_image(self, item_name: str) -> bool:
        """
        CHECK FOR EXISTING ITEM IMAGE - Prevents regeneration
        =====================================================
        Called before generating any item image to check if we already have one.
        Returns True if this item has EVER had an image (pregenerated OR dynamic).
        """
        return item_name in self.items_with_images

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(json_str: str) -> "GameState":
        payload = json.loads(json_str)
        return GameState(**payload)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class StateManager:
    def __init__(self, state: Optional[GameState] = None) -> None:
        self.state = state or GameState()

    # -------------------- Helper functions the LLM can call conceptually --------------------
    def move_to(self, new_location: str) -> None:
        if not new_location:
            return
        if new_location not in self.state.known_map:
            self.state.known_map[new_location] = {"exits": [], "notes": ""}
        self.state.location = new_location
        # CHANGE (visited-map UI fix): record visit so describe_map() can hide rooms
        # the player has not yet been to. World-bible rooms are seeded into known_map
        # for engine logic but should not show up in the player-facing "Known" map.
        if new_location not in self.state.visited_rooms:
            self.state.visited_rooms.append(new_location)

    def connect_rooms(self, room_a: str, room_b: str) -> None:
        for room_x, room_y in ((room_a, room_b), (room_b, room_a)):
            if room_x not in self.state.known_map:
                self.state.known_map[room_x] = {"exits": [], "notes": ""}
            exits = self.state.known_map[room_x].setdefault("exits", [])
            if room_y not in exits:
                exits.append(room_y)

    def add_item(self, item: str) -> None:
        if item and item not in self.state.inventory:
            self.state.inventory.append(item)

    def remove_item(self, item: str) -> None:
        if item in self.state.inventory:
            self.state.inventory.remove(item)

    def change_health(self, delta: int) -> None:
        # Allow health beyond 100 for "supercharged" states, or negative for special conditions
        self.state.health = max(-10, min(200, self.state.health + int(delta)))

    def add_note(self, note: str) -> None:
        if note:
            self.state.notes.append(note)

    # -------------------- Room items --------------------
    def place_item_in_room(self, item: str, room: Optional[str] = None) -> None:
        if not item:
            return
        room_name = room or self.state.location
        items = self.state.room_items.setdefault(room_name, [])
        if item not in items:
            items.append(item)

    def remove_item_from_room(self, item: str, room: Optional[str] = None) -> bool:
        room_name = room or self.state.location
        items = self.state.room_items.get(room_name, [])
        if item in items:
            items.remove(item)
            return True
        return False

    def list_room_items(self, room: Optional[str] = None) -> List[str]:
        room_name = room or self.state.location
        return list(self.state.room_items.get(room_name, []))

    # -------------------- Enhanced State Query Helpers --------------------

    def get_shortest_path(self, start_room: Optional[str] = None, target_room: str = "") -> List[str]:
        """
        Find shortest path between rooms using BFS.
        Returns empty list if no path exists.
        """
        start = start_room or self.state.location
        if start == target_room:
            return [start]

        visited = set()
        queue = [(start, [start])]  # (current_room, path_to_here)
        visited.add(start)

        while queue:
            current, path = queue.pop(0)
            exits = self.state.known_map.get(current, {}).get("exits", [])

            for exit_room in exits:
                if exit_room not in visited:
                    new_path = path + [exit_room]
                    if exit_room == target_room:
                        return new_path
                    visited.add(exit_room)
                    queue.append((exit_room, new_path))

        return []  # No path found

    def get_room_distance(self, target_room: str, start_room: Optional[str] = None) -> int:
        """Get distance (rooms) to target room. Returns -1 if unreachable."""
        path = self.get_shortest_path(start_room, target_room)
        return len(path) - 1 if path else -1

    def find_items_by_type(self, item_type: str) -> Dict[str, List[str]]:
        """
        Find all items containing a type keyword in their name.
        Returns {"inventory": [...], "rooms": {"room_name": [...]}}
        """
        result = {"inventory": [], "rooms": {}}

        # Check inventory
        for item in self.state.inventory:
            if item_type.lower() in item.lower():
                result["inventory"].append(item)

        # Check room items
        for room, items in self.state.room_items.items():
            matching_items = [item for item in items if item_type.lower() in item.lower()]
            if matching_items:
                result["rooms"][room] = matching_items

        return result

    def get_connected_rooms(self, room: Optional[str] = None, depth: int = 1) -> Dict[str, List[str]]:
        """
        Get all rooms connected within specified depth.
        Returns {distance: [room_names]}
        """
        start = room or self.state.location
        result = {}
        visited = set()
        current_level = {start}

        for d in range(depth + 1):
            if d > 0:
                result[d] = list(current_level - visited)
            visited.update(current_level)

            next_level = set()
            for r in current_level:
                exits = self.state.known_map.get(r, {}).get("exits", [])
                next_level.update(exits)
            current_level = next_level - visited

        return result

    def get_item_relationships(self, item_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about an item's relationships.
        Returns location, nearby items, usage hints from flags, etc.
        """
        info = {
            "name": item_name,
            "location": None,
            "nearby_items": [],
            "related_flags": {},
            "usage_hints": []
        }

        # Find location
        if item_name in self.state.inventory:
            info["location"] = "inventory"
        else:
            for room, items in self.state.room_items.items():
                if item_name in items:
                    info["location"] = room
                    # Get nearby items in same room
                    info["nearby_items"] = [i for i in items if i != item_name]
                    break

        # Find related game flags
        for flag_name, flag_value in self.state.game_flags.items():
            if isinstance(flag_value, str) and item_name.lower() in flag_value.lower():
                info["related_flags"][flag_name] = flag_value
            elif isinstance(flag_value, dict) and str(flag_value.get("item", "")).lower() == item_name.lower():
                info["related_flags"][flag_name] = flag_value

        # Extract usage hints from story context
        if self.state.story_context:
            context_lower = self.state.story_context.lower()
            if item_name.lower() in context_lower:
                # Simple heuristic: extract sentences containing the item
                sentences = [s.strip() for s in self.state.story_context.split('.') if item_name.lower() in s.lower()]
                info["usage_hints"] = sentences[:3]  # Limit to 3 hints

        return info

    def get_game_progress_hints(self) -> List[str]:
        """
        Analyze current state and provide hints about next logical steps.
        Considers inventory, location, unexplored areas, etc.
        """
        hints = []

        # Check for unexplored directions
        current_exits = set(self.state.known_map.get(self.state.location, {}).get("exits", []))
        visited_rooms = set(self.state.known_map.keys())
        unexplored_exits = current_exits - visited_rooms

        if unexplored_exits:
            hints.append(f"You haven't explored: {', '.join(unexplored_exits)}")

        # Check for items that might be usable together
        if len(self.state.inventory) >= 2:
            hints.append("You have multiple items - try combining them or using them in different locations")

        # Check health status
        if self.state.health < 50:
            hints.append("Your health is low - look for healing items or safe areas")

        # Check for dead ends (rooms with only one exit)
        dead_ends = []
        for room, info in self.state.known_map.items():
            exits = info.get("exits", [])
            if len(exits) == 1:
                dead_ends.append(room)

        if dead_ends and self.state.location in dead_ends:
            hints.append("You're in a dead end - consider going back")

        return hints[:5]  # Limit hints to avoid overwhelming

    def process_timers_and_chains(self) -> List[str]:
        """
        Process active timers and chain reactions.
        Returns list of triggered effects for the LLM to narrate.
        Call this at the start of each turn.
        """
        triggered_events = []

        # Process timers
        timers_to_remove = []
        for flag_name, flag_value in list(self.state.game_flags.items()):
            if isinstance(flag_value, dict) and flag_value.get("type") == "timer":
                timer = flag_value
                timer["remaining"] -= 1

                if timer["remaining"] <= 0:
                    # Timer expired - execute action
                    action = timer.get("action", "")
                    value = timer.get("value")

                    if action == "take_damage" and isinstance(value, (int, float)):
                        old_health = self.state.health
                        self.change_health(-abs(int(value)))
                        triggered_events.append(f"Timer '{flag_name}' triggered: took {value} damage ({old_health} -> {self.state.health})")

                    elif action == "heal" and isinstance(value, (int, float)):
                        old_health = self.state.health
                        self.change_health(abs(int(value)))
                        triggered_events.append(f"Timer '{flag_name}' triggered: healed {value} HP ({old_health} -> {self.state.health})")

                    elif action == "remove_item" and isinstance(value, str):
                        if value in self.state.inventory:
                            self.remove_item(value)
                            triggered_events.append(f"Timer '{flag_name}' triggered: {value} disappeared from inventory")

                    elif action == "move_to" and isinstance(value, str):
                        old_location = self.state.location
                        self.move_to(value)
                        triggered_events.append(f"Timer '{flag_name}' triggered: teleported from {old_location} to {value}")

                    else:
                        triggered_events.append(f"Timer '{flag_name}' triggered: {action} with value {value}")

                    timers_to_remove.append(flag_name)
                else:
                    # Timer still counting down
                    triggered_events.append(f"Timer '{flag_name}' counting down: {timer['remaining']} turns remaining")

        # Remove expired timers
        for timer_name in timers_to_remove:
            del self.state.game_flags[timer_name]

        # Process chain reactions (check for triggers)
        for flag_name, flag_value in list(self.state.game_flags.items()):
            if isinstance(flag_value, dict) and flag_value.get("type") == "chain_reaction":
                chain = flag_value
                if not chain.get("active", False):
                    continue

                trigger = chain.get("trigger", "")
                # Simple trigger detection (can be expanded)
                triggered = False

                # Note: room movement triggers are handled in apply_llm_directives
                # Here we focus on flag-based and item-based triggers
                if trigger.startswith("used_"):
                    item = trigger[5:]
                    # Check if item was recently used (this is a simple heuristic)
                    if item in self.state.inventory:
                        triggered = True
                elif trigger in self.state.game_flags and self.state.game_flags[trigger]:
                    triggered = True

                if triggered:
                    effects = chain.get("effects", [])
                    for effect in effects:
                        effect_type = effect.get("type")
                        effect_value = effect.get("value")

                        if effect_type == "change_health" and isinstance(effect_value, (int, float)):
                            old_health = self.state.health
                            self.change_health(int(effect_value))
                            triggered_events.append(f"Chain reaction: health {effect_value:+d} ({old_health} -> {self.state.health})")

                        elif effect_type == "move_to" and isinstance(effect_value, str):
                            old_location = self.state.location
                            self.move_to(effect_value)
                            triggered_events.append(f"Chain reaction: moved to {effect_value}")

                        elif effect_type == "add_item" and isinstance(effect_value, str):
                            self.add_item(effect_value)
                            triggered_events.append(f"Chain reaction: gained {effect_value}")

                        elif effect_type == "remove_item" and isinstance(effect_value, str):
                            if effect_value in self.state.inventory:
                                self.remove_item(effect_value)
                                triggered_events.append(f"Chain reaction: lost {effect_value}")

                    # Deactivate chain reaction after triggering (unless permanent)
                    if not chain.get("permanent", False):
                        chain["active"] = False

        return triggered_events

    def describe_map(self) -> str:
        """Player-facing map: ONLY rooms the player has actually visited.
        World-bible rooms that haven't been entered yet are hidden. Use
        describe_full_map() for the designer / debug view."""
        # CHANGE (visited-map UI fix): "Known" should mean "I've been there",
        # not "the world bible mentions it". Filter known_map by visited_rooms.
        visited = list(self.state.visited_rooms or [])
        # Legacy / bootstrap case: visited_rooms not populated yet (e.g. older save,
        # or current location set but never via move_to). Fall back to current room.
        if not visited and self.state.location:
            visited = [self.state.location]
        if not visited:
            return "(map is empty — explore to discover rooms)"
        visited_set = set(visited)
        lines: List[str] = []
        # Iterate visited_rooms in visit order (most-recent feel) but skip duplicates.
        for room in visited:
            info = self.state.known_map.get(room) or {}
            # Only show exits to OTHER VISITED rooms in player-facing map; unseen
            # rooms beyond known exits stay as "(unexplored)".
            all_exits = info.get("exits", []) or []
            seen_exits = [e for e in all_exits if e in visited_set]
            unseen_count = sum(1 for e in all_exits if e not in visited_set)
            exits_parts: List[str] = []
            if seen_exits:
                exits_parts.append(", ".join(seen_exits))
            if unseen_count:
                exits_parts.append(f"{unseen_count} unexplored")
            exits_str = " · ".join(exits_parts) or "none"
            notes = info.get("notes", "") or ""
            note_str = f" | {notes[:80]}{'…' if len(notes) > 80 else ''}" if notes else ""
            here = " ◀ here" if room == self.state.location else ""
            lines.append(f"• {room}{here} → {exits_str}{note_str}")
        return "\n".join(lines)

    def describe_full_map(self) -> str:
        """Designer / debug view: ALL rooms in known_map (which includes the
        world-bible seed), with ✓ on visited, ? on unseen. Used by the
        'Show full map' toggle in the Play tab."""
        if not self.state.known_map:
            return "(no rooms known)"
        visited_set = set(self.state.visited_rooms or [])
        lines: List[str] = []
        for room, info in self.state.known_map.items():
            exits_str = ", ".join(info.get("exits", []) or []) or "None"
            notes = info.get("notes", "") or ""
            note_str = f" | {notes[:80]}{'…' if len(notes) > 80 else ''}" if notes else ""
            if room == self.state.location:
                marker = " ◀ here"
            elif room in visited_set:
                marker = " ✓"
            else:
                marker = " ?"
            lines.append(f"• {room}{marker} → {exits_str}{note_str}")
        return "\n".join(lines)

    # -------------------- Persistence --------------------
    def save(self, label: Optional[str] = None) -> str:
        timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        label = label or f"{self.state.player_name}_{timestamp}"
        # Create a comprehensive .pkl file with everything
        pkl_name = f"Adv_{label}.pkl"
        pkl_path = os.path.join(SAVE_DIR, pkl_name)
        try:
            # Collect all image data
            image_data = {}
            if os.path.isdir(IN_GAME_IMG_DIR):
                for root, _, files in os.walk(IN_GAME_IMG_DIR):
                    for fn in files:
                        full = os.path.join(root, fn)
                        try:
                            with open(full, "rb") as img_f:
                                image_data[fn] = img_f.read()
                        except Exception:
                            pass
            
            # Create comprehensive save data
            save_data = {
                'game_state': self.state,
                'world_bible': WORLD_BIBLE,
                'images': image_data,
                # Optional UI extras for richer reloads (backward compatible)
                'narration': getattr(self, "_current_narration", ""),
                'debug': getattr(self, "_current_debug", []),
                'save_version': '2.0'  # Version marker for new format
            }
            
            with open(pkl_path, "wb") as f:
                pickle.dump(save_data, f)
            print(f"[save] Created comprehensive .pkl file: {pkl_name}")
        except Exception as e:
            print(f"[save] pkl creation failed: {e}")
        
        return pkl_path

    def load(self, file_path: str) -> None:
        global WORLD_BIBLE
        
        # Handle new comprehensive .pkl files only
        if file_path.lower().endswith(".pkl") and os.path.isfile(file_path):
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                
                # Check if it's the new comprehensive format (dict with save_version)
                if isinstance(data, dict) and 'save_version' in data:
                    print(f"[load] Loading .pkl file (version {data.get('save_version')})")
                    
                    # Load game state
                    self.state = data['game_state']
                    # LEGACY COMPAT: ensure new fields exist on older saves (one-line default)
                    self.state.recent_history = getattr(self.state, 'recent_history', [])
                    self.state.story_context = getattr(self.state, 'story_context', "")
                    self.state.game_flags = getattr(self.state, 'game_flags', {})
                    
                    # Load world bible
                    WORLD_BIBLE = data.get('world_bible')
                    
                    # Restore images
                    image_data = data.get('images', {})
                    for filename, img_bytes in image_data.items():
                        try:
                            img_path = os.path.join(IN_GAME_IMG_DIR, filename)
                            with open(img_path, "wb") as img_f:
                                img_f.write(img_bytes)
                        except Exception as e:
                            print(f"[load] Failed to restore image {filename}: {e}")
                    
                    print(f"[load] Successfully loaded .pkl file with {len(image_data)} images")
                    # Optional UI extras (if present)
                    try:
                        self._loaded_narration = data.get('narration', None)
                        dbg = data.get('debug', None)
                        # Normalize debug to list[str]
                        if isinstance(dbg, str):
                            self._loaded_debug = [line for line in dbg.splitlines() if line.strip()]
                        elif isinstance(dbg, list):
                            self._loaded_debug = dbg
                        else:
                            self._loaded_debug = None
                    except Exception:
                        self._loaded_narration = None
                        self._loaded_debug = None
                    return
                
                else:
                    print(f"[load] This appears to be an old/unsupported .pkl format. Please create a new save.")
                    raise Exception("Unsupported save format - please create a new save")
                    
            except Exception as e:
                print(f"[load] Failed to load .pkl file: {e}")
                raise Exception(f"Failed to load save file: {e}")
        
        # Handle .tkl files (new archive format)
        if file_path.lower().endswith(".tkl") and os.path.isfile(file_path):
            try:
                with zipfile.ZipFile(file_path, "r") as zf:
                    # Extract into SAVE_DIR/tmp_<stamp>
                    tmp_dir = os.path.join(SAVE_DIR, f"tmp_{int(time.time())}")
                    os.makedirs(tmp_dir, exist_ok=True)
                    zf.extractall(tmp_dir)
                    # Load state (first Adv_*.json found)
                    json_files = [p for p in os.listdir(tmp_dir) if p.endswith('.json') and p.startswith('Adv_')]
                    if json_files:
                        try:
                            with open(os.path.join(tmp_dir, json_files[0]), "r", encoding="utf-8") as f:
                                self.state = GameState.from_json(f.read())
                        except UnicodeDecodeError as e:
                            print(f"[load] UTF-8 decode error in main JSON file {json_files[0]}: {e}")
                            # Try with different encoding
                            with open(os.path.join(tmp_dir, json_files[0]), "r", encoding="latin-1") as f:
                                self.state = GameState.from_json(f.read())
                    # Load world bible if present
                    wb_files = [p for p in os.listdir(tmp_dir) if p.endswith('.world_bible.json')]
                    if wb_files:
                        try:
                            with open(os.path.join(tmp_dir, wb_files[0]), "r", encoding="utf-8") as f:
                                WORLD_BIBLE = json.load(f)
                        except UnicodeDecodeError as e:
                            print(f"[load] UTF-8 decode error in world bible {wb_files[0]}: {e}")
                            try:
                                with open(os.path.join(tmp_dir, wb_files[0]), "r", encoding="latin-1") as f:
                                    WORLD_BIBLE = json.load(f)
                            except Exception:
                                WORLD_BIBLE = None
                        except Exception:
                            WORLD_BIBLE = None
                    # Merge images into IN_GAME_IMG_DIR
                    img_src = os.path.join(tmp_dir, "in_game_images")
                    if os.path.isdir(img_src):
                        for fn in os.listdir(img_src):
                            s = os.path.join(img_src, fn)
                            d = os.path.join(IN_GAME_IMG_DIR, fn)
                            try:
                                shutil.copyfile(s, d)
                            except Exception:
                                pass
                    # Cleanup
                    try:
                        shutil.rmtree(tmp_dir)
                    except Exception:
                        pass
                return
            except Exception as e:
                print(f"[load] failed to load tkl: {e}")
                # fallthrough to try JSON
        
        # Handle .json files (direct JSON state)
        with open(file_path, "r", encoding="utf-8") as f:
            json_str = f.read()
        self.state = GameState.from_json(json_str)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def llm_choice_is_ollama(model_path: str) -> bool:
    """True if this dropdown / CLI value selects Ollama (ollama:<tag>) instead of an MLX weights folder."""
    return bool(model_path) and model_path.startswith(OLLAMA_CHOICE_PREFIX)


def ollama_model_from_choice(model_path: str) -> str:
    """Strip ollama: prefix; return tag for ollama.chat(model=...)."""
    if llm_choice_is_ollama(model_path):
        return model_path[len(OLLAMA_CHOICE_PREFIX) :]
    return model_path


def ollama_chat_generate(
    model_name: str, system_prompt: str, user_prompt: str, max_tokens: int, temperature: float,
    json_mode: bool = False,
) -> str:
    # CHANGE: Ollama path for LLMEngine — uses native chat API (no MLX load).
    # CHANGE (Qwen3.6/Gemma4 calibration): when json_mode=True we pass `format="json"`
    # to Ollama. Ollama's runtime will then ENFORCE strict JSON output at decode time,
    # which is the single biggest reliability win for world-bible generation on local
    # 30B models. (Ignored on the MLX backend; we keep prompt-level JSON guidance there.)
    if ollama_pkg is None:
        return "(LLM error: ollama package not installed; pip install ollama)"
    try:
        kwargs = dict(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_predict": max_tokens, "temperature": temperature},
        )
        if json_mode:
            kwargs["format"] = "json"
        r = ollama_pkg.chat(**kwargs)
        if hasattr(r, "message") and r.message is not None:
            content = getattr(r.message, "content", None) or ""
        elif isinstance(r, dict):
            content = ((r.get("message") or {}) or {}).get("content") or ""
        else:
            content = ""
        return (content or "").strip()
    except Exception as e:
        return f"(LLM error: {e})"


class LLMEngine:
    def __init__(self, model_path: str, max_new_tokens: int = 1500, temperature: float = 0.7) -> None:
        self.model_path = model_path
        self._uses_ollama = llm_choice_is_ollama(model_path)
        if self._uses_ollama:
            self.model_id = ollama_model_from_choice(model_path)
        else:
            self.model_id = os.path.basename(model_path) if model_path else ""
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        if model_path:
            self._load()

    def _load(self):
        if self._uses_ollama:
            print(f"[LLMEngine] Ollama model {self.model_id!r} (no MLX weights load).")
            return
        print(f"[LLMEngine] Loading model from {self.model_path} ...")
        self.model, self.tokenizer = mlx_load(self.model_path)
        print(f"[LLMEngine] Model loaded: {self.model_id}")

    def _unload(self):
        self.model = None
        self.tokenizer = None
        if self._uses_ollama:
            return
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass

    def generate(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        # CHANGE (Qwen3.6/Gemma4 calibration): json_mode flag plumbs Ollama's strict
        # JSON output mode through to the runtime. World-bible generators set this to
        # True; runtime narrator leaves it False (it needs free-form prose + a JSON block).
        if self._uses_ollama:
            return ollama_chat_generate(
                self.model_id, system_prompt, user_prompt, self.max_new_tokens, self.temperature,
                json_mode=json_mode,
            )
        if self.model is None or self.tokenizer is None:
            return "(LLM error: no model loaded)"
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            prompt_tokens = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            sampler = make_sampler(temp=self.temperature)
            text = mlx_generate(
                self.model, self.tokenizer, prompt=prompt_tokens,
                max_tokens=self.max_new_tokens, sampler=sampler, verbose=False,
            )
            return text.strip()
        except Exception as e:
            return f"(LLM error: {e})"


# ═══════════════════════════════════════════════════════════════════════════════
# JSON PARSING & ERROR RECOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def fix_common_json_errors(json_str: str) -> str:
    """
    Fix common LLM JSON generation mistakes to make parsing more robust.
    Enhanced with better error recovery and partial directive handling.
    """
    import re

    # Remove trailing commas before closing braces/brackets
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    # Fix missing commas between key-value pairs
    # Look for patterns like "key1": value1 "key2": value2 and add comma
    json_str = re.sub(r'("\w+")\s*:\s*([^,}]+)\s+(?="[^"]+"\s*:)', r'\1: \2,', json_str)

    # Fix unquoted keys (common LLM mistake: {name: "value"} instead of {"name": "value"})
    # This is tricky to do safely, so we'll be conservative
    # Only fix if it looks like a simple key without spaces or special chars
    def fix_unquoted_keys(match):
        key = match.group(1)
        # Only fix keys that look like valid identifiers
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
            return f'"{key}": '
        return match.group(0)  # Leave as-is if it doesn't look safe

    json_str = re.sub(r'(\w+):\s', fix_unquoted_keys, json_str)

    # Fix incomplete JSON by adding missing closing braces/brackets
    # Count braces and brackets
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')

    # Add missing closing braces
    while close_braces < open_braces:
        json_str += '}'
        close_braces += 1

    # Add missing closing brackets
    while close_brackets < open_brackets:
        json_str += ']'
        close_brackets += 1

    # Fix single quotes to double quotes (but be careful about apostrophes in strings)
    # This is complex, so we'll skip for now to avoid breaking valid strings

    # Remove any extra closing braces/brackets at the end that might have been added incorrectly
    # This is a simple heuristic - if we have more closing than opening, remove extras
    while json_str.count('}') > json_str.count('{'):
        json_str = json_str.rstrip('}')
    while json_str.count(']') > json_str.count('['):
        json_str = json_str.rstrip(']')

    return json_str


def parse_json_with_fallbacks(json_str: str, recognized_keys: List[str] = None) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Parse JSON with multiple fallback strategies for maximum robustness.
    Returns (parsed_dict, error_message) where error_message is empty on success.
    """
    if recognized_keys is None:
        recognized_keys = ["state_updates", "images"]

    # Strategy 1: Direct parsing
    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result, ""
    except Exception as e:
        pass

    # Strategy 2: Fix common errors and retry
    try:
        fixed_json = fix_common_json_errors(json_str)
        result = json.loads(fixed_json)
        if isinstance(result, dict):
            return result, ""
    except Exception as e:
        pass

    # Strategy 3: Extract and parse individual key-value pairs
    # Look for patterns like "key": value or key: value
    partial_result = {}
    error_msgs = []

    # Pattern for quoted keys
    quoted_pattern = re.compile(r'"(' + '|'.join(recognized_keys) + r')"?\s*:\s*({.*?}|\[.*?\]|"[^"]*"|\d+|true|false|null)', re.IGNORECASE | re.DOTALL)

    # Pattern for unquoted keys (common LLM mistake)
    unquoted_pattern = re.compile(r'(?<!")\b(' + '|'.join(recognized_keys) + r')\b(?!")\s*:\s*({.*?}|\[.*?\]|"[^"]*"|\d+|true|false|null)', re.IGNORECASE | re.DOTALL)

    for pattern in [quoted_pattern, unquoted_pattern]:
        for match in pattern.finditer(json_str):
            key, value_str = match.groups()
            try:
                # Try to parse the value
                value = json.loads(value_str)
                partial_result[key] = value
            except Exception as e:
                error_msgs.append(f"Failed to parse {key}: {str(e)[:50]}")

    if partial_result:
        return partial_result, f"Partial parse succeeded: {', '.join(error_msgs)}"

    return None, f"All parsing strategies failed. Raw text: {json_str[:200]}..."


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    """
    Extract and parse JSON objects from text using multiple strategies.
    Returns a list of successfully parsed JSON objects.
    Enhanced with better extraction and fallback parsing.
    """
    candidates = []
    recognized_keys = ["state_updates", "images"]

    # Strategy 1: Extract complete JSON objects using balanced brace counting
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        json_start = start_idx

        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object - try to parse it
                    json_candidate = text[json_start:i+1]
                    parsed, error = parse_json_with_fallbacks(json_candidate, recognized_keys)
                    if parsed:
                        candidates.append(parsed)
                    # Look for next JSON object
                    remaining = text[i+1:]
                    next_start = remaining.find('{')
                    if next_start != -1:
                        json_start = i + 1 + next_start
                        brace_count = 0
                    else:
                        break

    # Strategy 2: Look for partial JSON structures that might contain our keys
    # This catches cases where the LLM starts JSON but doesn't complete it
    if not candidates:
        # Look for any text that contains our recognized keys near braces
        key_patterns = []
        for key in recognized_keys:
            # Match key followed by colon and some structure
            pattern = rf'"{key}"?\s*:\s*({{.*?}}|\[.*?\]|"[^"]*"|\d+|true|false|null)'
            key_patterns.append(pattern)

        combined_pattern = '|'.join(f'({p})' for p in key_patterns)
        matches = re.finditer(combined_pattern, text, re.IGNORECASE | re.DOTALL)

        partial_dict = {}
        for match in matches:
            # Try to extract key-value pairs
            matched_text = match.group(0)
            # Simple key-value extraction
            colon_idx = matched_text.find(':')
            if colon_idx > 0:
                key_part = matched_text[:colon_idx].strip().strip('"')
                value_part = matched_text[colon_idx+1:].strip()
                try:
                    value = json.loads(value_part)
                    partial_dict[key_part] = value
                except:
                    pass

        if partial_dict:
            candidates.append(partial_dict)

    return candidates


def strip_llm_thinking_blocks(text: str) -> str:
    """
    Remove chain-of-thought / analysis wrappers so balanced-brace JSON scan
    starts at the real world-bible object (not a tiny snippet inside reasoning).
    Also used in apply_llm_directives so player-facing narration drops reasoning prefixes.
    """
    t = text
    # Common tag pairs (reasoning models; strip so the first `{` is the world bible / json)
    for pat in (
        r"<think>[\s\S]*?</think>",
        r"<redacted_thinking>[\s\S]*?</redacted_thinking>",
        r"<\|think\|>[\s\S]*?<\|/think\|>",
        r"<\|channel\|>thought[\s\S]*?<\|channel\|>",
        r"<\|channel\|>[\s\S]*?<\|channel\|>",
        r"<channel\|>[\s\S]*?\|>",
    ):
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    # Unclosed reasoning tail (cap length so we do not wipe a whole reply by mistake)
    t = re.sub(r"<think>[\s\S]{0,50000}$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"<redacted_thinking>[\s\S]{0,50000}$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^[\s\n]*<channel\|>\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^[\s\n]*<\|channel\|>\s*", "", t, flags=re.IGNORECASE)

    # CHANGE (Qwen3.6/Gemma4 calibration): chat-template control tokens that occasionally
    # leak through when sampling near a turn boundary. Stripping them is always safe —
    # they are never legitimate game content.
    qwen_gemma_tokens = [
        r"<\|im_start\|>(?:system|user|assistant)?\s*",
        r"<\|im_end\|>",
        r"<\|endoftext\|>",
        r"<\|begin_of_text\|>",
        r"<\|end_of_text\|>",
        r"<\|start_header_id\|>(?:system|user|assistant)?<\|end_header_id\|>",
        r"<\|eot_id\|>",
        r"<start_of_turn>(?:user|model)?\s*",  # Gemma
        r"<end_of_turn>",                       # Gemma
        r"<bos>",
        r"<eos>",
        r"<pad>",
    ]
    for pat in qwen_gemma_tokens:
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    return t.strip()


def score_world_bible_candidate(d: Dict[str, Any]) -> int:
    """Heuristic: prefer objects that look like a full world bible, not inline JSON fragments."""
    if not isinstance(d, dict):
        return -1
    s = 0
    ob = d.get("objectives")
    if isinstance(ob, list) and len(ob) > 0:
        s += 80 + min(len(ob), 10) * 8
    loc = d.get("locations")
    if isinstance(loc, list) and len(loc) > 0:
        s += 80 + min(len(loc), 12) * 12
    ki = d.get("key_items")
    if isinstance(ki, list) and len(ki) > 0:
        s += 80 + min(len(ki), 12) * 6
    # CHANGE (chunk G): structured win_condition scores higher than free-form string.
    wc = d.get("win_condition")
    if isinstance(wc, dict):
        # Structured form is preferred; reward presence of either required_items or required_location.
        if wc.get("required_items") or wc.get("required_location"):
            s += 130
        else:
            s += 60
    elif isinstance(wc, str) and wc.strip():
        s += 80
    for k in ("npcs", "monsters", "riddles", "mechanics", "progression_hints", "main_arc"):
        v = d.get(k)
        if isinstance(v, list) and len(v) > 0:
            s += 15 + min(len(v), 8) * 2
        elif isinstance(v, str) and v.strip():
            s += 12
    il = d.get("item_locations")
    if isinstance(il, dict) and len(il) > 0:
        s += 20 + min(len(il), 12) * 2
    if d.get("global_theme") or d.get("theme"):
        s += 15
    # Prefer larger coherent payloads (ties broken toward full bible)
    try:
        s += min(len(json.dumps(d, ensure_ascii=False)), 8000) // 80
    except Exception:
        pass
    return s


def pick_world_bible_from_candidates(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Local models often emit many valid JSON objects (reasoning snippets, examples).
    Pick one that validates; else the highest schema score (closest to recoverable).
    """
    if not candidates:
        return None
    dicts = [c for c in candidates if isinstance(c, dict)]
    if not dicts:
        return None
    ranked = sorted(dicts, key=score_world_bible_candidate, reverse=True)
    for c in ranked:
        ok, _ = validate_world_bible(c)
        if ok:
            return c
    return ranked[0]


def extract_fallback_directives(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract directives from text using flexible pattern matching when JSON fails.
    Supports formats like:
    - move_to: Cave Entrance
    - add_items: sword, shield
    - change_health: +10
    """
    directives = {}
    lines = text.split('\n')

    # Pattern matching for common directive formats
    patterns = {
        "move_to": re.compile(r'(?:move_to|move\s+to|go\s+to)\s*:\s*(.+)', re.IGNORECASE),
        "add_items": re.compile(r'(?:add_items|add\s+items?|gain|get)\s*:\s*(.+)', re.IGNORECASE),
        "remove_items": re.compile(r'(?:remove_items|remove\s+items?|drop|lose)\s*:\s*(.+)', re.IGNORECASE),
        "change_health": re.compile(r'(?:change_health|health|damage|heal)\s*:\s*([+-]?\d+)', re.IGNORECASE),
        "place_items": re.compile(r'(?:place_items|place\s+items?|put|leave)\s*:\s*(.+)', re.IGNORECASE),
        "room_take": re.compile(r'(?:room_take|take\s+from\s+room|pick\s+up|get)\s*:\s*(.+)', re.IGNORECASE),
        "connect": re.compile(r'(?:connect|link)\s*:\s*(.+)', re.IGNORECASE),
    }

    state_updates = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue

        for directive_type, pattern in patterns.items():
            match = pattern.search(line)
            if match:
                value = match.group(1).strip()

                if directive_type in ["add_items", "remove_items", "place_items", "room_take"]:
                    # Parse comma-separated lists
                    items = [item.strip().strip('"').strip("'") for item in value.split(',')]
                    items = [item for item in items if item]  # Remove empty items
                    if items:
                        state_updates[directive_type] = items

                elif directive_type == "change_health":
                    try:
                        state_updates[directive_type] = int(value)
                    except ValueError:
                        continue

                elif directive_type == "connect":
                    # Parse room connections like "RoomA-RoomB" or "RoomA to RoomB"
                    connections = []
                    parts = re.split(r'\s*(?:to|-|<>|↔)\s*', value)
                    if len(parts) >= 2:
                        room_a = parts[0].strip()
                        room_b = parts[1].strip()
                        connections.append([room_a, room_b])
                        state_updates[directive_type] = connections

                elif directive_type == "move_to":
                    state_updates[directive_type] = value

                break  # Only match first pattern per line

    # Extract image prompts from lines containing image-related keywords
    image_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in ['image', 'picture', 'show', 'display', 'visual']):
            # Extract the actual description
            image_match = re.search(r'(?:image|picture|show|display|visual)(?:\s+of)?\s*:\s*(.+)', line, re.IGNORECASE)
            if image_match:
                image_lines.append(image_match.group(1).strip())

    if state_updates:
        directives["state_updates"] = state_updates

    if image_lines:
        directives["images"] = image_lines

    return directives if directives else None


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION & WORLD BIBLE
# ═══════════════════════════════════════════════════════════════════════════════

def validate_and_fix_directives(directives: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Validate and fix common JSON schema issues in LLM directives.
    Returns (fixed_directives, warning_messages)
    """
    if not isinstance(directives, dict):
        return directives, ["Directives must be a dictionary"]

    warnings = []
    fixed = {}

    # Validate state_updates
    if "state_updates" in directives:
        state_updates = directives["state_updates"]
        if not isinstance(state_updates, dict):
            warnings.append("state_updates should be a dictionary")
            fixed["state_updates"] = {}
        else:
            fixed_updates = {}

            # Validate move_to
            if "move_to" in state_updates:
                move_to = state_updates["move_to"]
                if isinstance(move_to, str) and move_to.strip():
                    fixed_updates["move_to"] = move_to.strip()
                else:
                    warnings.append("move_to should be a non-empty string")

            # Validate arrays that should contain strings
            array_fields = ["add_items", "remove_items", "place_items", "room_take"]
            for field in array_fields:
                if field in state_updates:
                    value = state_updates[field]
                    if isinstance(value, list):
                        # Ensure all items are strings
                        fixed_items = [str(item).strip() for item in value if item]
                        fixed_updates[field] = fixed_items
                    elif isinstance(value, str):
                        # Convert single string to array
                        fixed_updates[field] = [value.strip()]
                    else:
                        warnings.append(f"{field} should be a list of strings")

            # Validate change_health
            if "change_health" in state_updates:
                health = state_updates["change_health"]
                try:
                    fixed_updates["change_health"] = int(health)
                except (ValueError, TypeError):
                    warnings.append("change_health should be a number")

            # Validate connect
            if "connect" in state_updates:
                connect = state_updates["connect"]
                if isinstance(connect, list):
                    fixed_connections = []
                    for conn in connect:
                        if isinstance(conn, list) and len(conn) == 2:
                            fixed_connections.append([str(conn[0]), str(conn[1])])
                        else:
                            warnings.append("connect entries should be [room_a, room_b] pairs")
                    if fixed_connections:
                        fixed_updates["connect"] = fixed_connections
                else:
                    warnings.append("connect should be a list of room pairs")

            # Validate other string fields
            string_fields = ["add_note", "set_context"]
            for field in string_fields:
                if field in state_updates:
                    value = state_updates[field]
                    if isinstance(value, str):
                        fixed_updates[field] = value.strip()
                    else:
                        warnings.append(f"{field} should be a string")

            # Validate set_flag
            if "set_flag" in state_updates:
                flag = state_updates["set_flag"]
                if isinstance(flag, dict) and "name" in flag:
                    fixed_updates["set_flag"] = flag
                else:
                    warnings.append("set_flag should be a dict with 'name' field")

            # Validate mechanics
            if "mechanics" in state_updates:
                mech = state_updates["mechanics"]
                if isinstance(mech, dict) and "action" in mech and "effect" in mech:
                    fixed_updates["mechanics"] = mech
                else:
                    warnings.append("mechanics should have 'action' and 'effect' fields")

            if fixed_updates:
                fixed["state_updates"] = fixed_updates

    # Validate images
    if "images" in directives:
        images = directives["images"]
        if isinstance(images, list):
            # Ensure all items are strings
            fixed_images = [str(img).strip() for img in images if img]
            fixed["images"] = fixed_images
        elif isinstance(images, str):
            # Convert single string to array
            fixed["images"] = [images.strip()]
        else:
            warnings.append("images should be a list of strings")

    # Validate advanced mechanics
    advanced_fields = ["conditional_action", "timer_event", "chain_reaction"]
    for field in advanced_fields:
        if field in directives:
            value = directives[field]
            if isinstance(value, dict):
                fixed[field] = value
            else:
                warnings.append(f"{field} should be a dictionary")

    return fixed, warnings


def validate_world_bible(world_bible: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate that a world bible creates a playable, non-trivial game.
    Returns (is_valid, list_of_issues)
    """
    issues = []
    
    # Check for required fields
    required_fields = ["objectives", "locations", "key_items", "win_condition"]
    for field in required_fields:
        if field not in world_bible or not world_bible[field]:
            issues.append(f"Missing or empty required field: {field}")

    # CHANGE (chunk G): allow either the new struct or a string (string is normalized later).
    wc = world_bible.get("win_condition")
    if wc is not None:
        if isinstance(wc, dict):
            if not wc.get("required_items") and not wc.get("required_location"):
                issues.append("win_condition has no required_items and no required_location — not winnable")
        elif isinstance(wc, str):
            if not wc.strip():
                issues.append("win_condition is an empty string")
        else:
            issues.append(f"win_condition must be a dict or string, got {type(wc).__name__}")

    # Check minimum complexity
    if "locations" in world_bible:
        if len(world_bible["locations"]) < 5:
            issues.append(f"Too few locations ({len(world_bible['locations'])}), need at least 5 for interesting gameplay")
    
    if "key_items" in world_bible:
        if len(world_bible["key_items"]) < 4:
            issues.append(f"Too few items ({len(world_bible['key_items'])}), need at least 4 for puzzle solving")
    
    # Check for interconnection hints
    if "progression_hints" in world_bible:
        if len(world_bible.get("progression_hints", [])) < 2:
            issues.append("Not enough progression hints for player guidance")
    
    # Check that items have purposes
    if "key_items" in world_bible:
        for item in world_bible["key_items"]:
            if isinstance(item, dict) and not item.get("purpose"):
                issues.append(f"Item '{item.get('name', 'unknown')}' has no purpose defined")
    
    # Check that locations are described
    if "locations" in world_bible:
        for loc in world_bible["locations"]:
            if isinstance(loc, dict) and not loc.get("description"):
                issues.append(f"Location '{loc.get('name', 'unknown')}' has no description")
    
    return len(issues) == 0, issues


# CHANGE (chunk A): solvability validator. Builds the room graph from `locations[].exits`,
# verifies reachability from the start to the win-condition location, checks every key_item
# is placed somewhere, every blocker references a real item, and the solution_chain steps
# move along the actual graph. Returns (ok, gaps, human_readable_report) — the report is
# what we print so users can see WHY a generated bible failed.
def validate_world_bible_solvability(world_bible: Dict[str, Any]) -> Tuple[bool, List[str], str]:
    gaps: List[str] = []
    report: List[str] = []
    if not isinstance(world_bible, dict):
        return False, ["world_bible is not a dict"], "world_bible is not a dict"

    # ---- Build room graph (treat exits as bidirectional, same as the engine) ----
    locs = world_bible.get("locations") or []
    loc_names: List[str] = []
    graph: Dict[str, List[str]] = {}
    for loc in locs:
        if not isinstance(loc, dict):
            continue
        nm = str(loc.get("name") or "").strip()
        if not nm:
            continue
        loc_names.append(nm)
        graph[nm] = [str(e).strip() for e in (loc.get("exits") or []) if isinstance(e, str)]
    loc_set = set(loc_names)
    if not loc_set:
        return False, ["no usable locations in world bible"], "no usable locations"

    sym: Dict[str, Set[str]] = {n: set() for n in loc_set}
    for a, neigh in graph.items():
        for b in neigh:
            if b in loc_set:
                sym[a].add(b)
                sym[b].add(a)
            else:
                gaps.append(f"location '{a}' lists exit to unknown room '{b}'")

    # Reachability from start (first listed location)
    start = loc_names[0]
    seen: Set[str] = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nxt in sym.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    unreachable = sorted(loc_set - seen)
    if unreachable:
        gaps.append(f"unreachable rooms from start '{start}': {', '.join(unreachable)}")

    # ---- Items ----
    item_names: List[str] = []
    for ki in world_bible.get("key_items") or []:
        if isinstance(ki, dict):
            n = str(ki.get("name") or "").strip()
            if n:
                item_names.append(n)
    item_set = set(item_names)

    item_locations = world_bible.get("item_locations") or {}
    placed_count = 0
    for it in item_names:
        if it not in item_locations:
            gaps.append(f"key_item '{it}' has no entry in item_locations (player cannot obtain it)")
            continue
        loc_desc = str(item_locations[it])
        if any(rn.lower() in loc_desc.lower() for rn in loc_set):
            placed_count += 1
        else:
            gaps.append(f"item_locations['{it}'] = '{loc_desc}' references no known room")

    # ---- Win condition (structured form from chunk G) ----
    parsed_win = _parse_win_condition(world_bible)
    win_loc = parsed_win.get("required_location")
    win_items = parsed_win.get("required_items") or []
    if win_loc and win_loc not in loc_set:
        gaps.append(f"win_condition.required_location '{win_loc}' is not a known room")
    elif win_loc and win_loc not in seen:
        gaps.append(f"win_condition.required_location '{win_loc}' not reachable from start '{start}'")
    for it in win_items:
        if it not in item_set:
            gaps.append(f"win_condition.required_items references '{it}' missing from key_items")
    if not win_items and not win_loc:
        gaps.append("win_condition has no required_items and no required_location — cannot determine victory")

    # ---- NPCs / monsters / riddles must live in known rooms ----
    for who_key, label in (("npcs", "npc"), ("monsters", "monster"), ("riddles", "riddle")):
        for entry in world_bible.get(who_key, []) or []:
            if not isinstance(entry, dict):
                continue
            loc_at = str(entry.get("location") or "").strip()
            who = str(entry.get("name") or entry.get("hint") or "?")
            if loc_at and loc_at not in loc_set:
                gaps.append(f"{label} '{who}' lives in unknown room '{loc_at}'")

    # ---- Monster weaknesses should reference at least one key_item ----
    for m in world_bible.get("monsters", []) or []:
        if not isinstance(m, dict):
            continue
        wk = str(m.get("weakness") or "").lower()
        if wk and not any(it.lower() in wk for it in item_set):
            gaps.append(
                f"monster '{m.get('name', '?')}' weakness '{m.get('weakness')}' "
                "does not reference any key_item — player has no defined way to defeat it"
            )

    # ---- solution_chain consistency (the strongest solvability signal) ----
    chain = world_bible.get("solution_chain") or []
    if not chain:
        gaps.append("no solution_chain provided — cannot verify the puzzle is winnable")
    else:
        prev_loc: Optional[str] = None
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                gaps.append(f"solution_chain[{i}] is not an object")
                continue
            sl = str(step.get("location") or "").strip()
            req = step.get("requires_item")
            if sl and sl not in loc_set:
                gaps.append(f"solution_chain[{i}].location '{sl}' is unknown")
            if isinstance(req, str) and req and req not in item_set:
                gaps.append(f"solution_chain[{i}].requires_item '{req}' missing from key_items")
            if prev_loc and sl in loc_set and prev_loc in loc_set and sl != prev_loc:
                # Allow multi-hop reachability via the graph
                sub = {prev_loc}
                st = [prev_loc]
                while st:
                    cu = st.pop()
                    for nx in sym.get(cu, ()):
                        if nx not in sub:
                            sub.add(nx)
                            st.append(nx)
                if sl not in sub:
                    gaps.append(f"solution_chain[{i}] jumps from '{prev_loc}' to '{sl}' with no path")
            prev_loc = sl
        if win_loc and isinstance(chain[-1], dict):
            last_loc = str(chain[-1].get("location") or "").strip()
            if last_loc and last_loc != win_loc:
                gaps.append(f"solution_chain ends at '{last_loc}' but win_condition requires '{win_loc}'")

    # ---- Build the report ----
    report.append(f"[solvability] rooms={len(loc_set)} reachable_from_'{start}'={len(seen)}/{len(loc_set)}")
    report.append(f"[solvability] items={len(item_set)} placed_in_known_rooms={placed_count}/{len(item_set)}")
    if win_loc:
        report.append(
            f"[solvability] win_location='{win_loc}' reachable={'yes' if win_loc in seen else 'NO'}"
        )
    report.append(f"[solvability] solution_chain steps={len(chain)}")
    if gaps:
        report.append(f"[solvability] gaps ({len(gaps)}):")
        for g in gaps[:25]:
            report.append(f"  - {g}")
        if len(gaps) > 25:
            report.append(f"  - … {len(gaps) - 25} more")
    else:
        report.append("[solvability] gaps: none — looks solvable")
    return (len(gaps) == 0), gaps, "\n".join(report)


# CHANGE (chunk M): per-gap LLM micro-repair.
# Asking a 30B-class model to regenerate the whole world bible to fix one issue
# is unreliable — it tends to introduce new mistakes elsewhere. Instead, ask
# ONE narrow multiple-choice question per gap and apply the answer in code.
# This plays to the strength of small instruction-following models (pick one
# from a list) and avoids exercising their weakness (long, cross-referenced
# JSON generation).
def _ask_one_choice(
    wb_engine: "LLMEngine", system_msg: str, question: str, choices: List[str]
) -> Optional[str]:
    """Ask the LLM to pick ONE entry from `choices`. Returns the matched choice or None.

    The model only has to copy a string from a list — far more reliable than
    open-ended JSON. We accept exact, case-insensitive, and substring matches.
    """
    if not choices:
        return None
    prompt = (
        f"{question}\n\n"
        "Pick exactly ONE option from this list. Reply with the chosen text on a single line, "
        "with no quotes, no JSON, no explanation, no markdown — just the option text.\n"
        "Options:\n"
        + "\n".join(f"  - {c}" for c in choices)
    )
    resp = wb_engine.generate(system_msg, prompt) or ""
    cleaned = strip_llm_thinking_blocks(resp).strip().strip("`'\" \n")
    if not cleaned:
        return None
    first_line = cleaned.splitlines()[0].strip().strip("`'\" -*")
    if not first_line:
        return None
    # Exact match
    for c in choices:
        if c == first_line:
            return c
    # Case-insensitive
    fl = first_line.lower()
    for c in choices:
        if c.lower() == fl:
            return c
    # Substring match in either direction
    for c in choices:
        if c.lower() in fl or fl in c.lower():
            return c
    return None


def micro_repair_world_bible(
    world_bible: Dict[str, Any], wb_engine: "LLMEngine", system_msg: str
) -> List[str]:
    """Run targeted, narrow LLM questions for gaps that benefit from a thematic
    choice (monster weaknesses, chain step items, broken exits). Each fix is one
    short LLM call with a constrained answer set. Returns repair messages.
    """
    repairs: List[str] = []
    if not isinstance(world_bible, dict):
        return repairs

    loc_names = [
        str((l or {}).get("name", "")).strip()
        for l in (world_bible.get("locations") or [])
        if isinstance(l, dict)
    ]
    loc_names = [l for l in loc_names if l]
    loc_set = set(loc_names)

    item_names = [
        str((i or {}).get("name", "")).strip()
        for i in (world_bible.get("key_items") or [])
        if isinstance(i, dict)
    ]
    item_names = [n for n in item_names if n]

    # Items that auto_repair (chunk H) speculatively added end up with purposes
    # like "Defeats Golem" or "Reward of riddle …". They're placeholders the LLM
    # should replace with thematically real items when possible.
    h_placeholders = set()
    for ki in world_bible.get("key_items", []) or []:
        if not isinstance(ki, dict):
            continue
        purpose = str(ki.get("purpose") or "")
        if purpose.startswith("Defeats ") or purpose.startswith("Reward of "):
            n = str(ki.get("name") or "").strip()
            if n:
                h_placeholders.add(n)

    real_items = [n for n in item_names if n not in h_placeholders]

    # ─── (1) Monster weaknesses pointing at H-placeholder items ────────────
    for m in world_bible.get("monsters", []) or []:
        if not isinstance(m, dict):
            continue
        wk = str(m.get("weakness") or "").strip()
        if wk and wk in h_placeholders and real_items:
            question = (
                f"Monster '{m.get('name','?')}' (in {m.get('location','?')}) currently has "
                f"weakness '{wk}', which was auto-generated and is not a real game item. "
                "Choose the most thematic existing item the player would use to defeat this monster."
            )
            chosen = _ask_one_choice(wb_engine, system_msg, question, real_items)
            if chosen and chosen != wk:
                m["weakness"] = chosen
                # Drop the placeholder item so it doesn't pollute the inventory.
                world_bible["key_items"] = [
                    ki for ki in world_bible.get("key_items", []) or []
                    if not (isinstance(ki, dict) and ki.get("name") == wk)
                ]
                repairs.append(
                    f"micro: monster '{m.get('name','?')}' weakness '{wk}' → existing item '{chosen}' "
                    f"(removed placeholder item)"
                )
                # Keep our local lists in sync
                if wk in item_names:
                    item_names.remove(wk)
                h_placeholders.discard(wk)
                real_items = [n for n in item_names if n not in h_placeholders]

    # ─── (2) Solution-chain steps whose requires_item was cleared by H ─────
    chain = world_bible.get("solution_chain") or []
    if isinstance(chain, list):
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue
            if step.get("requires_item") is None:
                blk = str(step.get("blocker") or "").strip()
                sl = str(step.get("location") or "?")
                if blk and len(real_items) >= 2:
                    question = (
                        f"In solution_chain step {i+1} at room '{sl}', the player faces blocker "
                        f"'{blk}'. Choose ONE existing item the player would need to overcome this "
                        "blocker, or pick 'NONE' if no item is needed."
                    )
                    chosen = _ask_one_choice(
                        wb_engine, system_msg, question, real_items + ["NONE"]
                    )
                    if chosen and chosen != "NONE":
                        step["requires_item"] = chosen
                        repairs.append(
                            f"micro: solution_chain[{i}].requires_item set to '{chosen}' "
                            f"(blocker: {blk[:40]})"
                        )

    # ─── (3) Room exits pointing at unknown rooms ──────────────────────────
    for loc in world_bible.get("locations", []) or []:
        if not isinstance(loc, dict):
            continue
        nm = str(loc.get("name") or "").strip()
        ex_list = loc.get("exits")
        if not isinstance(ex_list, list):
            continue
        new_list: List[str] = []
        for ex in ex_list:
            if not isinstance(ex, str):
                continue
            ex_s = ex.strip()
            if not ex_s:
                continue
            if ex_s in loc_set:
                if ex_s not in new_list:
                    new_list.append(ex_s)
                continue
            others = [l for l in loc_names if l != nm and l not in new_list]
            if not others:
                continue
            question = (
                f"Room '{nm}' lists an exit to '{ex_s}', but '{ex_s}' is not a real room in "
                "this game. Choose ONE real room name from the list to replace it (the player "
                "should be able to move from one to the other)."
            )
            chosen = _ask_one_choice(wb_engine, system_msg, question, others)
            if chosen and chosen not in new_list:
                new_list.append(chosen)
                repairs.append(
                    f"micro: exit '{nm}' → '{ex_s}' replaced with real room '{chosen}'"
                )
        if new_list != ex_list:
            loc["exits"] = new_list

    # Invalidate cached parsed win condition so it re-derives if needed
    world_bible.pop("_win_parsed", None)
    return repairs


# CHANGE (chunk H): deterministic, surgical auto-repair of common world-bible gaps.
# Most generation problems on local models are mechanical: a name typo, a missing
# item_locations entry, an unreachable room, an NPC parked in an invented room.
# Fixing these in code BEFORE asking the LLM to retry saves a generation round-trip
# and produces playable bibles even from sloppy generators. Mutates world_bible.
def auto_repair_world_bible(world_bible: Dict[str, Any]) -> List[str]:
    """
    Apply local fixes to world_bible IN PLACE. Returns a list of human-readable
    repair messages so the generator can show the user exactly what changed.
    Idempotent: running it twice on a clean bible returns [].
    """
    repairs: List[str] = []
    if not isinstance(world_bible, dict):
        return repairs

    locs = world_bible.get("locations")
    if not isinstance(locs, list) or not locs:
        return repairs

    loc_names: List[str] = []
    for loc in locs:
        if isinstance(loc, dict):
            nm = str(loc.get("name") or "").strip()
            if nm:
                loc_names.append(nm)
    if not loc_names:
        return repairs

    loc_set = set(loc_names)
    start_room = loc_names[0]

    # ─── Items: collect names from key_items ────────────────────────────────
    key_items = world_bible.get("key_items")
    if not isinstance(key_items, list):
        key_items = []
        world_bible["key_items"] = key_items
    item_names: List[str] = []
    for it in key_items:
        if isinstance(it, dict):
            n = str(it.get("name") or "").strip()
            if n:
                item_names.append(n)
    item_set = set(item_names)

    # ─── (1) Add items referenced by win/chain/weakness/reward but missing  ──
    referenced: Dict[str, str] = {}  # name -> derived purpose hint

    wc = world_bible.get("win_condition")
    if isinstance(wc, dict):
        for it in wc.get("required_items") or []:
            it = str(it).strip()
            if it and it not in item_set:
                referenced.setdefault(it, "Required to win the game")

    chain = world_bible.get("solution_chain") or []
    if isinstance(chain, list):
        for step in chain:
            if not isinstance(step, dict):
                continue
            req = step.get("requires_item")
            if isinstance(req, str) and req.strip() and req.strip() not in item_set:
                referenced.setdefault(req.strip(), f"Used at {step.get('location','?')}")

    for m in world_bible.get("monsters", []) or []:
        if isinstance(m, dict):
            wk = str(m.get("weakness") or "").strip()
            # Only treat short noun-phrase weaknesses as item references; long
            # sentences like "fears fire from a torch" are kept as-is.
            if wk and len(wk) < 40 and wk not in item_set and re.match(r"^[A-Za-z][A-Za-z\s'\-]+$", wk):
                referenced.setdefault(wk, f"Defeats {m.get('name','enemy')}")

    for rid in world_bible.get("riddles", []) or []:
        if isinstance(rid, dict):
            rew = str(rid.get("reward") or "").strip()
            if rew and len(rew) < 40 and rew not in item_set and re.match(r"^[A-Za-z][A-Za-z\s'\-]+$", rew):
                referenced.setdefault(rew, f"Reward of riddle at {rid.get('location','?')}")

    for new_item, purpose in referenced.items():
        key_items.append({"name": new_item, "purpose": purpose})
        item_set.add(new_item)
        item_names.append(new_item)
        repairs.append(f"added missing key_item '{new_item}' ({purpose})")

    # ─── (2) item_locations: ensure each key_item has an entry that names a real room
    item_locations = world_bible.get("item_locations")
    if not isinstance(item_locations, dict):
        item_locations = {}
        world_bible["item_locations"] = item_locations

    chain_loc_for_item: Dict[str, Optional[str]] = {}
    if isinstance(chain, list):
        for step in chain:
            if not isinstance(step, dict):
                continue
            loc_at = str(step.get("location") or "").strip()
            loc_at = loc_at if loc_at in loc_set else None
            req = step.get("requires_item")
            if isinstance(req, str) and req.strip():
                chain_loc_for_item.setdefault(req.strip(), loc_at)
            res_text = str(step.get("result") or "")
            for it in item_names:
                if it and it.lower() in res_text.lower():
                    chain_loc_for_item.setdefault(it, loc_at)

    for it in item_names:
        if it not in item_locations:
            target_loc = chain_loc_for_item.get(it) or start_room
            item_locations[it] = f"{target_loc}: placed by auto-repair"
            repairs.append(f"placed item '{it}' at '{target_loc}' (was missing from item_locations)")
        else:
            loc_desc = str(item_locations[it])
            if not any(rn.lower() in loc_desc.lower() for rn in loc_set):
                target_loc = chain_loc_for_item.get(it) or start_room
                item_locations[it] = f"{target_loc}: {loc_desc}"
                repairs.append(f"item_locations['{it}']: prefixed with real room '{target_loc}'")

    # ─── (3) Connectivity: from start, BFS through exits; for any unreachable
    # room, attach an exit to its previous neighbor in the locations list.
    sym: Dict[str, Set[str]] = {n: set() for n in loc_set}
    for loc in locs:
        if not isinstance(loc, dict):
            continue
        nm = str(loc.get("name") or "").strip()
        if not nm:
            continue
        for ex in loc.get("exits") or []:
            if isinstance(ex, str) and ex.strip() in loc_set:
                sym[nm].add(ex.strip())
                sym[ex.strip()].add(nm)

    seen: Set[str] = {start_room}
    stack = [start_room]
    while stack:
        cur = stack.pop()
        for nxt in sym.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    if seen != loc_set:
        unreachable = [n for n in loc_names if n not in seen]
        for room in unreachable:
            idx = loc_names.index(room)
            anchor = loc_names[idx - 1] if idx > 0 else start_room
            for loc in locs:
                if not isinstance(loc, dict):
                    continue
                nm = str(loc.get("name") or "").strip()
                if nm == anchor:
                    ex_list = loc.setdefault("exits", [])
                    if isinstance(ex_list, list) and room not in ex_list:
                        ex_list.append(room)
                if nm == room:
                    ex_list = loc.setdefault("exits", [])
                    if isinstance(ex_list, list) and anchor not in ex_list:
                        ex_list.append(anchor)
            sym[anchor].add(room)
            sym[room].add(anchor)
            repairs.append(f"auto-connected '{anchor}' <-> '{room}' (room was unreachable from start)")

    # ─── (4) NPCs / monsters / riddles in unknown rooms: relocate ────────────
    for who_key in ("npcs", "monsters", "riddles"):
        for entry in world_bible.get(who_key, []) or []:
            if not isinstance(entry, dict):
                continue
            loc_at = str(entry.get("location") or "").strip()
            if loc_at and loc_at not in loc_set:
                target = start_room
                # Try to find a chain step that mentions this entry's name
                ent_name = str(entry.get("name") or entry.get("hint") or "").lower()
                if ent_name and isinstance(chain, list):
                    for step in chain:
                        if not isinstance(step, dict):
                            continue
                        if (
                            ent_name in str(step.get("blocker", "")).lower()
                            or ent_name in str(step.get("result", "")).lower()
                        ):
                            sl = str(step.get("location") or "").strip()
                            if sl in loc_set:
                                target = sl
                                break
                entry["location"] = target
                label = who_key[:-1]
                repairs.append(f"{label} '{entry.get('name','?')}' moved to '{target}' (was unknown room '{loc_at}')")

    # ─── (5) win_condition.required_location must be a real room ────────────
    if isinstance(wc, dict):
        rl = wc.get("required_location")
        if rl and rl not in loc_set:
            wc["required_location"] = start_room
            repairs.append(f"win_condition.required_location '{rl}' set to start room '{start_room}'")
        # Also drop required_items entries that don't exist (shouldn't happen post step 1, but defensive)
        ri = wc.get("required_items") or []
        if isinstance(ri, list):
            cleaned_ri = [x for x in ri if isinstance(x, str) and x.strip() in item_set]
            if cleaned_ri != ri:
                wc["required_items"] = cleaned_ri
                repairs.append(f"win_condition.required_items pruned to items present in key_items")

    # ─── (6) solution_chain steps: rebind unknown locations / items ─────────
    if isinstance(chain, list):
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue
            sl = str(step.get("location") or "").strip()
            if sl and sl not in loc_set:
                step["location"] = start_room
                repairs.append(f"solution_chain[{i}].location '{sl}' set to '{start_room}' (unknown room)")
            req = step.get("requires_item")
            if isinstance(req, str) and req.strip() and req.strip() not in item_set:
                step["requires_item"] = None
                repairs.append(f"solution_chain[{i}].requires_item '{req}' cleared (not in key_items)")

    # Invalidate cached parsed win condition so the next read normalizes the new shape.
    world_bible.pop("_win_parsed", None)
    return repairs


# ─────────────────────────────────────────────────────────────────────────────
# CHANGE (chunk F): Two-pass world-bible generation.
# Pass 1 ("skeleton") generates ONLY the structural pieces a small model can
# reliably emit at once: locations[name+description+exits], key_items[name+purpose],
# objectives[], structured win_condition. Pass 2 ("expansion") receives the
# skeleton verbatim and fills in npcs, monsters, riddles, item_locations,
# progression_hints, mechanics, main_arc, and solution_chain — every reference
# is constrained to the skeleton's exact room and item names. This is far more
# reliable than asking a 27B-class local model to produce a perfectly cross-
# referenced 12-field JSON in one shot. If either pass fails, we fall back to
# the legacy single-shot prompt below.
# ─────────────────────────────────────────────────────────────────────────────


def _wb_robust_parse(text: str) -> Optional[Dict[str, Any]]:
    """Reusable JSON-from-LLM-text parser used by both world-bible passes.
    Mirrors the multi-strategy logic from the legacy single-shot path so callers
    don't duplicate it. Returns the best parsed dict or None.
    """
    if not text:
        return None
    cleaned = text.replace("```json", "").replace("```", "").strip()
    cleaned = strip_llm_thinking_blocks(cleaned)

    # Strategy 1: balanced-brace candidates, pick highest-scoring.
    cands = extract_json_from_text(cleaned)
    if cands:
        chosen = pick_world_bible_from_candidates(cands)
        if isinstance(chosen, dict) and chosen:
            return chosen

    # Strategy 2: outer-most { ... } slice + json.loads + fix_common_json_errors.
    s, e = cleaned.find("{"), cleaned.rfind("}") + 1
    if s >= 0 and e > s:
        try:
            return json.loads(fix_common_json_errors(cleaned[s:e]))
        except Exception:
            pass

    # Strategy 3: aggressive multi-round fix on the whole string.
    js = cleaned.strip()
    if not js.startswith("{"):
        i = js.find("{")
        if i >= 0:
            js = js[i:]
    for _ in range(3):
        js = fix_common_json_errors(js)
        try:
            return json.loads(js)
        except Exception:
            continue
    return None


def _validate_skeleton(sk: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Light validation specific to pass 1: enough rooms, items, structured win_condition."""
    issues: List[str] = []
    if not isinstance(sk, dict):
        return False, ["skeleton is not a dict"]
    locs = sk.get("locations") or []
    if not isinstance(locs, list) or len(locs) < 5:
        issues.append(f"skeleton needs >=5 locations, got {len(locs) if isinstance(locs, list) else 'none'}")
    items = sk.get("key_items") or []
    if not isinstance(items, list) or len(items) < 4:
        issues.append(f"skeleton needs >=4 key_items, got {len(items) if isinstance(items, list) else 'none'}")
    # Each location must have a name and exits list (auto-filled empty is OK at this stage).
    for i, loc in enumerate(locs if isinstance(locs, list) else []):
        if not isinstance(loc, dict) or not str(loc.get("name") or "").strip():
            issues.append(f"locations[{i}] has no name")
    # win_condition must be a dict with at least one of required_items/required_location.
    wc = sk.get("win_condition")
    if not isinstance(wc, dict):
        issues.append("win_condition must be an object (required_items / required_location / description)")
    elif not (wc.get("required_items") or wc.get("required_location")):
        issues.append("win_condition needs at least one required_item or a required_location")
    return (len(issues) == 0), issues


def _generate_world_skeleton(
    wb_engine: "LLMEngine", enforced_theme: str, system_msg: str
) -> Optional[Dict[str, Any]]:
    """Pass 1: ONLY the structural skeleton. Smaller prompt, easier to keep coherent."""
    prompt = f"""You are designing a SOLVABLE text-adventure game.

This is PASS 1 of 2. In this pass output ONLY the structural skeleton. Do NOT
include npcs, monsters, riddles, item_locations, progression_hints, mechanics,
main_arc, or solution_chain — those are pass 2.

Return ONE JSON object with EXACTLY these fields:
- objectives: 4 short goal strings, in the order the player should accomplish them.
- global_theme: set this string EXACTLY to: {enforced_theme}
- locations: 5 rooms. Each room is an object with:
    - name: short title (no punctuation, used as a stable identifier)
    - description: ONE sentence describing what the player sees on entering
    - exits: list of OTHER `name`s reachable from this room (graph must be connected
      so the starting room reaches every other room)
  The FIRST location in this list is the starting room.
- key_items: 6-8 items. Each item is an object with:
    - name: short noun phrase (used as a stable identifier)
    - purpose: ONE sentence saying what the player uses it for
- win_condition: an OBJECT with these keys (NOT a string):
    - required_items: list of EXACT item names from `key_items` the player must hold
    - required_location: EXACT room name from `locations` (use null only if location is irrelevant)
    - description: one sentence the narrator can show on victory

Hard rules:
- Start room reaches every other room via exits (treated bidirectionally).
- The win_condition.required_location MUST be one of the locations you listed.
- All required_items MUST be names that appear in key_items.
- Output ONLY the JSON object. Begin with {{ and end with }}. No markdown, no commentary."""

    print("[world_bible/F] PASS 1 (skeleton) — prompting...")
    # CHANGE (Qwen3.6/Gemma4 calibration): json_mode forces strict JSON on Ollama.
    resp = wb_engine.generate(system_msg, prompt, json_mode=True)
    print(f"[world_bible/F] pass-1 raw ({len(resp)} chars):")
    print("=" * 80)
    print(resp)
    print("=" * 80)
    parsed = _wb_robust_parse(resp)
    if not parsed:
        print("[world_bible/F] pass-1 parse failed")
        return None
    ok, issues = _validate_skeleton(parsed)
    if not ok:
        print("[world_bible/F] pass-1 invalid skeleton — gaps:")
        for i in issues:
            print(f"  - {i}")
        # ONE retry with the gaps fed back
        retry_prompt = (
            "Your previous PASS 1 skeleton has these issues:\n"
            + "\n".join(f"  - {i}" for i in issues[:8])
            + "\n\nReturn a CORRECTED single JSON object with the same field set "
            "(objectives, global_theme, locations[name+description+exits], "
            "key_items[name+purpose], win_condition[required_items+required_location+description])."
            " Output ONLY the JSON. Begin with { and end with }."
        )
        retry_resp = wb_engine.generate(system_msg, retry_prompt, json_mode=True)
        retry_parsed = _wb_robust_parse(retry_resp)
        if retry_parsed:
            ok2, issues2 = _validate_skeleton(retry_parsed)
            if ok2:
                print("[world_bible/F] pass-1 retry succeeded")
                # Force theme to the enforced UI value.
                retry_parsed["global_theme"] = enforced_theme
                return retry_parsed
            print(f"[world_bible/F] pass-1 retry still invalid ({len(issues2)} gaps); skipping two-pass")
        return None
    parsed["global_theme"] = enforced_theme
    return parsed


def _validate_expansion(ex: Dict[str, Any], sk: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Light validation that the expansion only references skeleton names."""
    issues: List[str] = []
    if not isinstance(ex, dict):
        return False, ["expansion is not a dict"]
    sk_locs = {str(l.get("name") or "").strip() for l in (sk.get("locations") or []) if isinstance(l, dict)}
    sk_items = {str(i.get("name") or "").strip() for i in (sk.get("key_items") or []) if isinstance(i, dict)}
    for who_key, label in (("npcs", "npc"), ("monsters", "monster"), ("riddles", "riddle")):
        for entry in ex.get(who_key, []) or []:
            if not isinstance(entry, dict):
                continue
            loc_at = str(entry.get("location") or "").strip()
            if loc_at and loc_at not in sk_locs:
                issues.append(f"{label} '{entry.get('name', entry.get('hint', '?'))}' uses unknown location '{loc_at}'")
    il = ex.get("item_locations") or {}
    if isinstance(il, dict):
        for it in il:
            if it not in sk_items:
                issues.append(f"item_locations references unknown key_item '{it}'")
    chain = ex.get("solution_chain") or []
    if not isinstance(chain, list) or not chain:
        issues.append("solution_chain is empty")
    else:
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue
            sl = str(step.get("location") or "").strip()
            req = step.get("requires_item")
            if sl and sl not in sk_locs:
                issues.append(f"solution_chain[{i}].location '{sl}' is not a skeleton room")
            if isinstance(req, str) and req and req not in sk_items:
                issues.append(f"solution_chain[{i}].requires_item '{req}' is not a skeleton item")
    return (len(issues) == 0), issues


def _generate_world_expansion(
    wb_engine: "LLMEngine", skeleton: Dict[str, Any], system_msg: str
) -> Optional[Dict[str, Any]]:
    """Pass 2: NPCs, monsters, riddles, item_locations, hints, mechanics, arc, solution_chain.
    The skeleton is provided so all references stay consistent with pass 1."""
    sk_compact = {
        "locations": [
            {"name": l.get("name"), "exits": l.get("exits", [])}
            for l in skeleton.get("locations", []) if isinstance(l, dict)
        ],
        "key_items": [{"name": i.get("name"), "purpose": i.get("purpose", "")}
                      for i in skeleton.get("key_items", []) if isinstance(i, dict)],
        "win_condition": skeleton.get("win_condition"),
    }
    sk_json = json.dumps(sk_compact, ensure_ascii=False, indent=2)

    prompt = f"""You already designed PASS 1 (the skeleton). It is fixed.
Below is the skeleton. Do NOT invent new room names or new key_item names — every
reference you write MUST use the EXACT names from the skeleton.

SKELETON (do not modify, just reference):
{sk_json}

This is PASS 2 of 2. Return ONE JSON object with EXACTLY these fields:
- npcs: 2-3 characters. Each: {{name, location, personality, provides}}.
  `location` MUST be a skeleton room name.
- monsters: 2-3 enemies. Each: {{name, location, difficulty, weakness}}.
  `location` MUST be a skeleton room name. `weakness` MUST be the EXACT name
  of a skeleton key_item (so the player has a way to defeat it).
- riddles: 2 puzzles. Each: {{location, hint, solution, reward}}.
  `location` MUST be a skeleton room name. `reward` SHOULD be a skeleton item.
- item_locations: object mapping EVERY skeleton key_item name to a string of the
  form "<RoomName>: short hint" using ONLY skeleton room names.
- progression_hints: 3-4 short hint strings.
- mechanics: 3 cause-effect rules. Each: {{action, effect, location}}. `location`
  MUST be a skeleton room name.
- main_arc: 2-sentence story summary.
- solution_chain: ORDERED list of steps proving the game is winnable. Each step:
  {{step:int, location:RoomName, requires_item:item_or_null, blocker:str, result:str}}.
  - Use ONLY skeleton room names and skeleton item names.
  - Each step's `location` must be reachable from the previous step's `location`
    via the skeleton's `exits` graph (multi-hop is fine).
  - The final step's `location` must equal `win_condition.required_location` and
    by then the player must hold every `win_condition.required_items`.

Output ONLY the JSON object. Begin with {{ and end with }}. No markdown, no commentary."""

    print("[world_bible/F] PASS 2 (expansion) — prompting...")
    # CHANGE (Qwen3.6/Gemma4 calibration): json_mode forces strict JSON on Ollama.
    resp = wb_engine.generate(system_msg, prompt, json_mode=True)
    print(f"[world_bible/F] pass-2 raw ({len(resp)} chars):")
    print("=" * 80)
    print(resp)
    print("=" * 80)
    parsed = _wb_robust_parse(resp)
    if not parsed:
        print("[world_bible/F] pass-2 parse failed")
        return None
    ok, issues = _validate_expansion(parsed, skeleton)
    if not ok:
        print("[world_bible/F] pass-2 has consistency issues:")
        for i in issues[:8]:
            print(f"  - {i}")
        retry_prompt = (
            "Your previous PASS 2 expansion uses names that are not in the skeleton. "
            "Fix these issues:\n"
            + "\n".join(f"  - {i}" for i in issues[:10])
            + "\n\nReturn a CORRECTED single JSON object with the same fields. "
            "Use ONLY skeleton room names and skeleton key_item names. "
            "Output ONLY the JSON. Begin with { and end with }."
        )
        retry_resp = wb_engine.generate(system_msg, retry_prompt, json_mode=True)
        retry_parsed = _wb_robust_parse(retry_resp)
        if retry_parsed:
            ok2, issues2 = _validate_expansion(retry_parsed, skeleton)
            if ok2 or len(issues2) < len(issues):
                print(f"[world_bible/F] pass-2 retry produced {len(issues2)} issues (was {len(issues)}) — using retry")
                return retry_parsed
        print("[world_bible/F] pass-2 retry did not help; using original")
    return parsed


def _merge_skeleton_and_expansion(skel: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    """Combine the two passes into a full world-bible dict.
    Skeleton fields win on conflict (the structural truth)."""
    plan: Dict[str, Any] = {}
    plan.update(exp or {})
    plan.update(skel or {})  # skeleton overrides
    # Make sure both halves are present
    for k in ("npcs", "monsters", "riddles", "mechanics"):
        if k not in plan:
            plan[k] = exp.get(k, []) if isinstance(exp, dict) else []
    if "item_locations" not in plan:
        plan["item_locations"] = exp.get("item_locations", {}) if isinstance(exp, dict) else {}
    if "progression_hints" not in plan:
        plan["progression_hints"] = exp.get("progression_hints", []) if isinstance(exp, dict) else []
    if "main_arc" not in plan:
        plan["main_arc"] = exp.get("main_arc", "") if isinstance(exp, dict) else ""
    if "solution_chain" not in plan:
        plan["solution_chain"] = exp.get("solution_chain", []) if isinstance(exp, dict) else []
    return plan


def generate_world_bible(state_mgr: StateManager, heavy_model_id: Optional[str], theme: Optional[str] = None, max_tokens: int = 4000) -> Optional[Dict[str, Any]]:
    """
    SIMPLE world bible generation - just call LLM and return result.
    Returns None if generation fails or produces invalid results.
    """
    # Ensure max_tokens is valid
    if max_tokens is None or not isinstance(max_tokens, int) or max_tokens < 250:
        max_tokens = 4096  # Single-shot world JSON needs headroom (was 600; caused truncation)

    # Default cave adventure - well-designed and always available
    default_plan = {
        "objectives": [
            "Light your way through the darkness",
            "Gain passage across the underground river",
            "Unlock the ancient temple",
            "Claim the legendary treasure"
        ],
        "theme": "Dark fantasy cave with glowing crystals, ancient stone architecture, misty atmosphere",
        # CHANGE (chunk B): each location now declares its `exits` so the solvability validator
        # in chunk A can build a real connectivity graph (start -> win-condition reachable).
        "locations": [
            {"name": "Entrance", "description": "rocky cave mouth with supplies left by previous explorers", "exits": ["Dark Passage"]},
            {"name": "Dark Passage", "description": "pitch black tunnel that requires light", "exits": ["Entrance", "Underground River"]},
            {"name": "Underground River", "description": "rushing water with narrow ledge", "exits": ["Dark Passage", "Ancient Temple"]},
            {"name": "Ancient Temple", "description": "locked stone doors with riddle inscription", "exits": ["Underground River", "Treasury"]},
            {"name": "Treasury", "description": "golden vault guarded by stone sentinel", "exits": ["Ancient Temple"]}
        ],
        "npcs": [
            {"name": "Old Hermit", "location": "Entrance", "personality": "helpful guide", "provides": "torch and warning about troll"},
            {"name": "Cave Spirit", "location": "Ancient Temple", "personality": "riddling guardian", "provides": "temple key if riddle solved"}
        ],
        "monsters": [
            {"name": "River Troll", "location": "Underground River", "difficulty": "medium", "weakness": "fears fire from torch"},
            {"name": "Stone Golem", "location": "Treasury", "difficulty": "hard", "weakness": "ancient sword pierces its core"}
        ],
        "riddles": [
            {"location": "Ancient Temple", "hint": "I have cities but no houses, forests but no trees", "solution": "map", "reward": "temple access"},
            {"location": "Treasury", "hint": "Feed me and I grow, give me water and I die", "solution": "fire/torch", "reward": "golem becomes dormant"}
        ],
        "key_items": [
            {"name": "torch", "purpose": "lights dark areas, scares troll"},
            {"name": "rope", "purpose": "crosses river safely"},
            {"name": "map", "purpose": "solves temple riddle"},
            {"name": "temple key", "purpose": "unlocks inner sanctum"},
            {"name": "ancient sword", "purpose": "defeats stone golem"},
            {"name": "golden treasure", "purpose": "quest completion"}
        ],
        "item_locations": {
            "torch": "Given by hermit at entrance",
            "rope": "Found in dark passage once lit",
            "map": "Dropped by fleeing troll",
            "temple key": "Reward for solving spirit's riddle",
            "ancient sword": "Hidden in temple altar",
            "golden treasure": "Treasury vault after defeating golem"
        },
        "progression_hints": [
            "The hermit's gifts are essential for your journey",
            "Fire is both light and weapon",
            "The troll guards something important",
            "Ancient words unlock ancient doors"
        ],
        "mechanics": [
            {"action": "light torch in dark passage", "effect": "reveals hidden rope", "location": "Dark Passage"},
            {"action": "wave torch at troll", "effect": "troll flees dropping map", "location": "Underground River"},
            {"action": "place sword in altar slot", "effect": "opens secret treasury door", "location": "Ancient Temple"}
        ],
        "main_arc": "A brave explorer seeks legendary treasure in mysterious caves. They must use wit and courage to overcome guardians and puzzles.",
        # CHANGE (chunk G): structured win_condition replaces the free-form string. The runtime
        # checks `required_items` ⊂ inventory AND location == required_location.
        "win_condition": {
            "required_items": ["golden treasure"],
            "required_location": "Entrance",
            "description": "Carry the golden treasure back to the cave Entrance to escape with the prize.",
        },
        # CHANGE (chunk B): explicit ordered solution that the validator (chunk A) checks.
        # Each step references a real location, the item required (must be in key_items), the
        # blocker it overcomes, and the result. This is the puzzle dependency graph in plain JSON.
        "solution_chain": [
            {"step": 1, "location": "Entrance", "requires_item": None, "blocker": "no light for the passage", "result": "Receive torch from Old Hermit"},
            {"step": 2, "location": "Dark Passage", "requires_item": "torch", "blocker": "darkness", "result": "Reveals rope on the floor"},
            {"step": 3, "location": "Underground River", "requires_item": "torch", "blocker": "River Troll", "result": "Troll flees, drops map"},
            {"step": 4, "location": "Ancient Temple", "requires_item": "map", "blocker": "Cave Spirit's riddle", "result": "Receive temple key"},
            {"step": 5, "location": "Ancient Temple", "requires_item": "temple key", "blocker": "locked inner sanctum", "result": "Pick up ancient sword"},
            {"step": 6, "location": "Treasury", "requires_item": "ancient sword", "blocker": "Stone Golem", "result": "Defeat golem; pick up golden treasure"},
            {"step": 7, "location": "Entrance", "requires_item": "golden treasure", "blocker": "return trip", "result": "Win condition met"}
        ]
    }

    # If no theme provided, return the default cave adventure
    if not theme or not theme.strip():
        print("[world_bible] No theme specified, using default cave adventure")
        return default_plan

    # If no model, fail (don't provide default)
    if not heavy_model_id:
        print("[world_bible] No model provided")
        return None
    
    try:
        # Simple prompt
        snapshot = {
            "location": state_mgr.state.location,
            "rooms": list(state_mgr.state.known_map.keys())[:5]
        }
        
        # Use provided theme parameter, or fall back to existing WORLD_BIBLE theme, or default
        enforced_theme = theme
        if not enforced_theme or not enforced_theme.strip():
            try:
                wb = _get_world_bible()
                if wb:
                    enforced_theme = wb.get("global_theme")
            except Exception:
                pass

        # If still no theme, use default
        if not enforced_theme or not enforced_theme.strip():
            enforced_theme = DEFAULT_IMAGE_THEME
        
        # CHANGE (chunk B): require `exits` per location and a `solution_chain` so the
        # solvability validator (chunk A) can verify the puzzle is actually winnable before
        # play starts. Names referenced inside item_locations / solution_chain must match
        # entries in `locations` and `key_items` exactly.
        prompt = f"""Create a SOLVABLE adventure game as ONE JSON object for: {snapshot}

Theme (set "global_theme" to this exact string): {enforced_theme}

Required JSON fields:
- objectives: 4 goals in order
- locations: 5 rooms; EACH room MUST include `exits: ["RoomName", ...]` listing which
  other rooms (by exact `name`) you can travel to from it. The graph must connect the
  starting room to the room used in `win_condition`.
- npcs: 2-3 characters (name, location, personality, provides). `location` MUST match a
  room name from `locations`.
- monsters: 2-3 enemies (name, location, difficulty, weakness). `location` MUST match a
  room name. `weakness` SHOULD reference an item name from `key_items` so the player has
  a way to defeat it.
- riddles: 2 puzzles (location, hint, solution, reward). `location` MUST match a room.
- key_items: 6-8 items (name, purpose). EVERY item used in `solution_chain.requires_item`,
  `monsters.weakness`, `riddles.reward`, or `win_condition` MUST appear here.
- item_locations: {{ "item_name": "where the player FIRST gets it" }}. The location string
  MUST contain the exact name of a room from `locations`. Every item in `key_items` SHOULD
  have an entry.
- progression_hints: 3-4 hints
- mechanics: 3 cause-effect rules (action, effect, location)
- main_arc: 2-sentence story summary
- win_condition: an OBJECT with these keys (do NOT use a free-form string):
    - required_items: list of EXACT item names from `key_items` the player must hold to win
      (often just the final treasure / artifact). At least one entry.
    - required_location: EXACT room name from `locations` the player must be in to win
      (often the starting room for "return" quests). Use `null` only if the game is
      strictly an inventory completion with no location requirement.
    - description: one sentence the narrator can show the player on victory.
- solution_chain: ORDERED list of steps proving the game is winnable. Each step is an
  object: {{"step": int, "location": "RoomName", "requires_item": "item_or_null",
  "blocker": "what stops the player", "result": "what the step gives the player"}}.
  Steps must respect map connectivity: each step's `location` must be reachable from the
  previous step's `location` via `exits`. The final step must satisfy `win_condition`.

Example flow: torch (Entrance) -> lights Dark Passage -> reveals rope -> get map from
troll -> solve Temple riddle -> sword unlocks Treasury -> claim treasure -> return to
Entrance.

Every item in `key_items` must be referenced by something (a blocker, a riddle, the win
condition, or another item's purpose). Every NPC and monster must live in a room that
appears in `locations`. Every room in `locations` should be reachable from the starting
room. The game MUST be winnable using only items the player can actually obtain.

Keep each description one short sentence so the entire object fits in one reply.

Output ONLY valid JSON. No markdown, no commentary, no reasoning — your message must
begin with {{ and end with }}."""

        # Build model path: Ollama uses ollama:<tag>; MLX uses folder under MLX_MODELS_DIR or an absolute dir.
        if llm_choice_is_ollama(heavy_model_id):
            wb_model_path = heavy_model_id
        elif os.path.isdir(heavy_model_id):
            wb_model_path = heavy_model_id
        else:
            wb_model_path = os.path.join(MLX_MODELS_DIR, heavy_model_id)

        # Reasoning models emit many small JSON-like objects before the real bible; need token headroom.
        gen_tokens = max_tokens if isinstance(max_tokens, int) else 4096
        if gen_tokens < 4096:
            gen_tokens = 4096

        # DEBUG: Show full prompt being sent to LLM
        print(f"[world_bible] PROMPT BEING SENT TO {heavy_model_id} ({len(prompt)} chars):")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        print(f"[world_bible] Calling {heavy_model_id} with max_new_tokens={gen_tokens} (requested was {max_tokens})")

        # CHANGE (Qwen3.6/Gemma4 calibration): drop generation temperature 0.5 → 0.3.
        # 30B local models still produce creative content at 0.3 (the world bible's
        # creativity comes from the theme + structure, not from sampling), and lower
        # temperature gives much cleaner JSON / less drift on names.
        wb_engine = LLMEngine(model_path=wb_model_path, max_new_tokens=gen_tokens, temperature=0.3)
        system_msg = (
            "You are a game designer. Output ONLY one valid JSON object for the whole game design. "
            "Do not wrap output in tags, do not write analysis before or after the JSON, and do not use markdown. "
            "The first character of your reply must be { and the last must be }."
        )

        # CHANGE (chunk F): try TWO-PASS generation first. Pass 1 = skeleton, pass 2 = expansion.
        # If both succeed and validate, skip the legacy single-shot prompt entirely. If the two-pass
        # path fails at any step, fall through to the existing single-shot logic below.
        plan = None
        try:
            skeleton = _generate_world_skeleton(wb_engine, enforced_theme, system_msg)
            if skeleton is not None:
                expansion = _generate_world_expansion(wb_engine, skeleton, system_msg)
                if expansion is not None:
                    plan = _merge_skeleton_and_expansion(skeleton, expansion)
                    print(f"[world_bible/F] Two-pass produced full plan ({len(json.dumps(plan, ensure_ascii=False))} chars)")
                else:
                    # Skeleton without expansion is still playable, just sparse — use it.
                    plan = dict(skeleton)
                    print("[world_bible/F] Two-pass: pass 2 failed; using skeleton-only plan")
        except Exception as ex_e:
            print(f"[world_bible/F] Two-pass crashed ({ex_e}); falling back to single-shot")
            plan = None

        # CHANGE (chunk F): only run the legacy single-shot path if two-pass did not produce a plan.
        # When two-pass succeeded, we set `text = ""` so the parsing strategies below find nothing
        # and leave `plan` untouched (each strategy after the first already guards on `plan is None`).
        if plan is None:
            # CHANGE (Qwen3.6/Gemma4 calibration): single-shot fallback also gets JSON mode.
            raw_response = wb_engine.generate(system_msg, prompt, json_mode=True)

            print(f"[world_bible] RAW LLM RESPONSE ({len(raw_response)} chars):")
            print("=" * 80)
            print(raw_response)
            print("=" * 80)

            # Extract JSON from response - be very forgiving of LLM mistakes
            text = raw_response
            # Remove markdown if present, then strip reasoning wrappers so brace-scanning finds the real bible
            text = text.replace("```json", "").replace("```", "").strip()
            text = strip_llm_thinking_blocks(text)

            print(f"[world_bible] FULL RAW RESPONSE FROM LLM ({len(text)} chars):")
            print("=" * 80)
            print(text)
            print("=" * 80)
        else:
            text = ""  # two-pass produced a plan; nothing to parse below

        # Strategy 1: Try to extract complete JSON objects using balanced braces
        # NOTE: extract_json_from_text returns already-parsed dicts, not strings
        candidates = extract_json_from_text(text)
        if candidates and plan is None:
            # Prefer the object that validates (reasoning models often emit many small JSON snippets first)
            plan = pick_world_bible_from_candidates(candidates)
            sel_score = score_world_bible_candidate(plan) if plan else -1
            print(
                f"[world_bible] Parsed {len(candidates)} JSON object(s); "
                f"selected candidate score={sel_score}"
            )

        # Strategy 2: If no candidates worked, try basic { } extraction as fallback
        if plan is None:
            start = text.find("{")
            end = text.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = text[start:end]
                print(f"[world_bible] Attempting to parse extracted JSON ({len(json_str)} chars):")
                print(json_str)
                try:
                    # Try to fix common LLM JSON mistakes
                    json_str = fix_common_json_errors(json_str)
                    plan = json.loads(json_str)
                    print(f"[world_bible] Successfully parsed with JSON fixes")
                except Exception as e:
                    print(f"[world_bible] Basic JSON parsing failed: {e}")
                    print(f"[world_bible] Failed JSON string: {json_str}")

        # Strategy 3: More aggressive JSON completion and fixing
        if plan is None:
            try:
                # Try to extract and fix the entire response as potential JSON
                json_str = text.strip()
                if not json_str.startswith('{'):
                    start = json_str.find('{')
                    if start >= 0:
                        json_str = json_str[start:]

                # Apply multiple rounds of fixing
                for _ in range(3):  # Try up to 3 rounds of fixing
                    json_str = fix_common_json_errors(json_str)
                    try:
                        plan = json.loads(json_str)
                        print(f"[world_bible] Successfully parsed with aggressive JSON fixes")
                        break
                    except json.JSONDecodeError:
                        continue

            except Exception as e:
                print(f"[world_bible] All JSON parsing strategies failed: {e}")
                plan = None

        # If all parsing failed, retry once with simpler prompt
        if plan is None:
            print("[world_bible] First attempt failed — retrying with simpler prompt")
            retry_prompt = (
                f"Create adventure game JSON for theme: {enforced_theme}\n"
                "Include: objectives (4 goals), locations (5 rooms with name/description), "
                "npcs (2 with name/location/personality/provides), "
                "monsters (2 with name/location/difficulty/weakness), "
                "key_items (6 with name/purpose), win_condition.\n"
                "Output valid JSON only."
            )
            retry_response = wb_engine.generate(system_msg, retry_prompt, json_mode=True)
            retry_text = retry_response.replace("```json", "").replace("```", "").strip()
            retry_text = strip_llm_thinking_blocks(retry_text)
            retry_candidates = extract_json_from_text(retry_text)
            if retry_candidates:
                plan = pick_world_bible_from_candidates(retry_candidates)
                print("[world_bible] Retry succeeded")
            else:
                # Last resort: basic { } extraction
                s, e = retry_text.find("{"), retry_text.rfind("}") + 1
                if s >= 0 and e > s:
                    try:
                        plan = json.loads(fix_common_json_errors(retry_text[s:e]))
                        print("[world_bible] Retry succeeded with JSON fixes")
                    except Exception:
                        pass

        # CHANGE (chunk A): defer unload until AFTER the solvability check + optional retry,
        # otherwise we cannot ask the model to fix the gaps. Unload happens in two places
        # below depending on the path taken.
        if plan is None:
            print("[world_bible] All attempts failed — no valid JSON")
            wb_engine._unload()
            return None

        # NORMALIZE THEME KEY: prefer single canonical key "global_theme" (from UI)
        try:
            # Always set to enforced theme from UI to avoid drift
            plan["global_theme"] = enforced_theme
        except Exception:
            pass
        
        # Validate the generated world bible (shape check)
        is_valid, issues = validate_world_bible(plan)

        if not is_valid:
            print(f"[world_bible] Generated plan has shape issues:")
            for issue in issues:
                print(f"  - {issue}")
            print(f"[world_bible] Plan rejected - generation failed")
            wb_engine._unload()
            return None

        # CHANGE (chunk H): deterministic auto-repair BEFORE the LLM solvability retry.
        # Saves a generation round trip when problems are mechanical (missing
        # item_locations entry, NPC in invented room, isolated room, etc).
        repairs = auto_repair_world_bible(plan)
        if repairs:
            print(f"[world_bible/H] Auto-repaired {len(repairs)} issue(s):")
            for r in repairs[:30]:
                print(f"  - {r}")
            if len(repairs) > 30:
                print(f"  - … {len(repairs) - 30} more")

        # CHANGE (chunk M): per-gap LLM micro-repair for the things that benefit from
        # a thematic choice (monster weaknesses, chain step items, broken exits). Each
        # call asks ONE narrow question with a constrained answer set — far better suited
        # to a 30B-class model than a full bible regeneration.
        try:
            mrepairs = micro_repair_world_bible(plan, wb_engine, system_msg)
            if mrepairs:
                print(f"[world_bible/M] Micro-repaired {len(mrepairs)} item(s):")
                for r in mrepairs[:20]:
                    print(f"  - {r}")
                if len(mrepairs) > 20:
                    print(f"  - … {len(mrepairs) - 20} more")
        except Exception as _mr_e:
            print(f"[world_bible/M] micro-repair skipped ({_mr_e})")

        # CHANGE (chunk A): solvability check. If gaps remain after auto-repair, do ONE
        # retry feeding the residual gaps back to the LLM. If still broken, accept the
        # plan but print the report so the user knows exactly which puzzle pieces are missing.
        ok_solv, gaps, report = validate_world_bible_solvability(plan)
        print("[world_bible] Solvability report:")
        for line in report.splitlines():
            print(line)

        if not ok_solv and wb_engine is not None:
            top_gaps = "\n".join(f"  - {g}" for g in gaps[:12])
            fix_prompt = (
                "Your previous JSON has solvability gaps. Return a CORRECTED single JSON "
                "object with the same schema (same fields). Fix these specific issues:\n"
                f"{top_gaps}\n\n"
                "Hard requirements: every location.exits must be bidirectional via the "
                "engine, the start room must reach the win-condition room, every "
                "key_item must appear in item_locations referencing a real room, every "
                "monster.weakness must mention a key_item by name, and solution_chain "
                "must walk only along exits in `locations`.\n"
                "Output ONLY the JSON. Begin with { and end with }."
            )
            print("[world_bible] Retrying once with solvability gaps fed back...")
            retry_resp = wb_engine.generate(system_msg, fix_prompt, json_mode=True)
            retry_text = retry_resp.replace("```json", "").replace("```", "").strip()
            retry_text = strip_llm_thinking_blocks(retry_text)
            retry_plan = None
            cands = extract_json_from_text(retry_text)
            if cands:
                retry_plan = pick_world_bible_from_candidates(cands)
            if retry_plan is None:
                rs, re_ = retry_text.find("{"), retry_text.rfind("}") + 1
                if rs >= 0 and re_ > rs:
                    try:
                        retry_plan = json.loads(fix_common_json_errors(retry_text[rs:re_]))
                    except Exception:
                        retry_plan = None
            if retry_plan is not None:
                # Re-shape-check first so we don't replace a good plan with a broken retry
                ok2, issues2 = validate_world_bible(retry_plan)
                if ok2:
                    try:
                        retry_plan["global_theme"] = enforced_theme
                    except Exception:
                        pass
                    # CHANGE (chunk H): also auto-repair the retry plan so simple LLM mistakes
                    # don't sink an otherwise-better attempt.
                    retry_repairs = auto_repair_world_bible(retry_plan)
                    if retry_repairs:
                        print(f"[world_bible/H] Auto-repaired {len(retry_repairs)} issue(s) on retry plan:")
                        for r in retry_repairs[:15]:
                            print(f"  - {r}")
                    ok_solv2, gaps2, report2 = validate_world_bible_solvability(retry_plan)
                    print("[world_bible] Solvability report (after retry):")
                    for line in report2.splitlines():
                        print(line)
                    if len(gaps2) < len(gaps):
                        plan = retry_plan
                        ok_solv, gaps = ok_solv2, gaps2
                        print(f"[world_bible] Retry improved gap count: {len(gaps2)} (was {len(gaps)})")
                    else:
                        print(f"[world_bible] Retry did not improve solvability ({len(gaps2)} gaps); keeping original")
                else:
                    print(f"[world_bible] Retry produced an invalid shape; keeping original. Issues: {issues2[:3]}")

        if not ok_solv:
            print(f"[world_bible] Accepting plan with {len(gaps)} solvability gap(s) — game may be unfair until the LLM fills them in via play")

        # Cache the parsed win condition struct for runtime checks (chunk E).
        try:
            _parse_win_condition(plan)
        except Exception:
            pass

        # CHANGE (chunk A): finally unload the heavy model now that we are done with it.
        try:
            wb_engine._unload()
        except Exception:
            pass

        print(f"[world_bible] Successfully generated {len(str(plan))} char plan")
        return plan

    except Exception as e:
        print(f"[world_bible] Generation failed ({e})")

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM INSTRUCTIONS (LLM prompts)
# ═══════════════════════════════════════════════════════════════════════════════

# Core instructions — reliable with models ≥ 7B
SYSTEM_INSTRUCTIONS_CORE = (
    "You are a text adventure narrator.\n\n"
    "NEVER paste or summarize these instructions: no markdown headings (e.g. **Output Format:**), "
    "no numbered self-checklists, no 'Game State/Context' dumps, and no analysis before the story. "
    "Write only what the character experiences, then the ```json``` block.\n"
    # CHANGE (silent-LLM bugfix): explicit guard against reasoning-mode models spending
    # the entire token budget inside <think>...</think> with no prose for the player.
    "DO NOT use reasoning / chain-of-thought tags such as <think>, <reflect>, or "
    "<analysis>. Reply with the in-world narration directly so the player always sees "
    "a response within your token budget.\n\n"

    "YOUR OUTPUT FORMAT (follow this EXACTLY every turn):\n"
    "1. Write 2-5 sentences of narration describing what the player sees.\n"
    "2. After the narration, write a JSON block inside ```json``` fences.\n"
    "3. The JSON block MUST look like this:\n"
    '   ```json\n'
    '   {"state_updates": { ... }, "images": ["short image prompt"]}\n'
    '   ```\n\n'

    "BEFORE YOU RESPOND, CHECK THE GAME STATE:\n"
    "- location: where player is now\n"
    "- inventory: what player carries\n"
    "- current_room_items: what is in this room\n"
    "- current_exits: where the player can go\n"
    "- story_context: what happened before (YOUR MEMORY)\n"
    "- game_flags: puzzle states and conditions\n\n"

    "AVAILABLE TOOLS (put inside state_updates):\n"
    "  move_to: \"Room Name\"              - move player to a room\n"
    "  connect: [[\"RoomA\", \"RoomB\"]]     - link rooms bidirectionally\n"
    "  place_items: [\"item\"]              - put item in current room\n"
    "  room_take: [\"item\"]               - player PICKS UP item from room into inventory\n"
    "  add_items: [\"item\"]               - magically give item (NOT for picking up!)\n"
    "  remove_items: [\"item\"]            - destroy/consume item from inventory (NOT for picking up!)\n"
    "  change_health: -5 or +10           - damage or heal\n"
    "  set_context: \"summary\"            - save what happened (YOUR MEMORY)\n"
    "  set_flag: {\"name\": \"x\", \"value\": true} - track puzzle/event states\n"
    "  add_note: \"quest log entry\"       - add to quest log\n\n"
    "IMPORTANT: When player picks up an item, use room_take, NOT add_items+remove_items.\n"
    "room_take moves the item from the room to inventory in one step.\n\n"

    "IMAGE PROMPTS (put inside images array):\n"
    "  images: [\"short visual description of the scene\"]\n"
    "  ALWAYS include at least one image prompt.\n\n"

    "EXAMPLE 1 - Entering a new room:\n"
    "The passage opens into a pitch-black chamber. You hear dripping water. Without light, you can barely see.\n\n"
    "```json\n"
    "{\"state_updates\": {"
    "\"move_to\": \"Dark Chamber\", "
    "\"connect\": [[\"Entrance Hall\", \"Dark Chamber\"]], "
    "\"place_items\": [\"Old Torch\"], "
    "\"set_context\": \"Entered dark room. Torch on ground. Needs light.\"}, "
    "\"images\": [\"pitch-black cavern chamber, faint water dripping\"]}\n"
    "```\n\n"

    "EXAMPLE 2 - Picking up an item (use room_take!):\n"
    "You grab the torch from the dusty floor. It feels solid in your hand.\n\n"
    "```json\n"
    "{\"state_updates\": {"
    "\"room_take\": [\"Torch\"], "
    "\"set_context\": \"Picked up torch from floor. Now in inventory.\"}, "
    "\"images\": [\"adventurer holding a torch in a dim cave\"]}\n"
    "```\n\n"

    "EXAMPLE 3 - Using/consuming an item:\n"
    "You insert the rusty key into the lock. It turns with a click and the door swings open.\n\n"
    "```json\n"
    "{\"state_updates\": {"
    "\"remove_items\": [\"Rusty Key\"], "
    "\"set_flag\": {\"name\": \"treasury_unlocked\", \"value\": true}, "
    "\"move_to\": \"Treasury\", "
    "\"connect\": [[\"Locked Corridor\", \"Treasury\"]], "
    "\"set_context\": \"Used key to unlock treasury. Key consumed.\"}, "
    "\"images\": [\"ancient treasury door opening, golden light\"]}\n"
    "```\n\n"

    "RULES:\n"
    "1. ALWAYS check inventory before letting player use items\n"
    "2. ALWAYS check current_room_items before room_take\n"
    "3. ALWAYS use connect when moving to a NEW room\n"
    "4. ALWAYS update set_context with what happened this turn\n"
    "5. ALWAYS include at least one image prompt\n"
    "6. JSON goes AFTER narration, inside ```json``` fences\n"
    "7. Describe what the player SEES\n\n"

    "JSON FORMAT REMINDER:\n"
    "- Use double quotes for all keys and string values\n"
    "- No trailing commas\n"
    "- Close all { } and [ ] brackets\n"
)

# Advanced directive instructions — appended for larger / more capable models
SYSTEM_INSTRUCTIONS_ADVANCED = (
    "\nADVANCED TOOLS (also inside state_updates):\n"
    "  timer_event: {\"name\": \"poison\", \"duration\": 3, \"action\": \"take_damage\", \"value\": 5}\n"
    "  conditional_action: {\"condition\": \"has_key\", \"action\": \"unlock_door\", \"fallback\": \"door is locked\"}\n"
    "  chain_reaction: {\"trigger\": \"has_item_and_at_location\", \"effects\": [...]}\n"
    "  mechanics: {\"action\": \"light torch\", \"effect\": \"reveals hidden passage\", \"location\": \"Dark Cave\"}\n"
)


def get_system_instructions() -> str:
    """Return the appropriate system instructions based on ADVANCED_DIRECTIVES flag."""
    if ADVANCED_DIRECTIVES:
        return SYSTEM_INSTRUCTIONS_CORE + SYSTEM_INSTRUCTIONS_ADVANCED
    return SYSTEM_INSTRUCTIONS_CORE


def seed_game_state_from_world_bible(state_mgr: StateManager) -> List[str]:
    """
    Pre-fill known_map, exits, starting location, and room_items from WORLD_BIBLE
    so opening narration and current_room_items match the design document.
    Safe to call when no bible or incomplete bible — no-op.
    """
    debug: List[str] = []
    wb = _get_world_bible()
    if not wb:
        return debug
    locations = wb.get("locations") or []
    if not locations:
        return debug
    room_names: List[str] = []
    for loc in locations:
        if isinstance(loc, dict) and loc.get("name"):
            room_names.append(str(loc["name"]).strip())
        elif isinstance(loc, str) and loc.strip():
            room_names.append(loc.strip())
    if not room_names:
        return debug

    # known_map + exits from bible (or linear chain if exits omitted)
    for i, loc in enumerate(locations):
        if not isinstance(loc, dict):
            continue
        nm = str(loc.get("name") or "").strip()
        if not nm:
            continue
        desc = (loc.get("description") or "").strip()
        exits = loc.get("exits") if isinstance(loc.get("exits"), list) else []
        if nm not in state_mgr.state.known_map:
            state_mgr.state.known_map[nm] = {"exits": [], "notes": (desc[:1200] if desc else "")}
        else:
            if desc and not (state_mgr.state.known_map[nm].get("notes") or "").strip():
                state_mgr.state.known_map[nm]["notes"] = desc[:1200]
        for e in exits:
            if isinstance(e, str) and e.strip():
                state_mgr.connect_rooms(nm, e.strip())
        if not exits and i + 1 < len(locations):
            nxt = locations[i + 1]
            if isinstance(nxt, dict) and str(nxt.get("name") or "").strip():
                state_mgr.connect_rooms(nm, str(nxt["name"]).strip())

    start = room_names[0]
    state_mgr.move_to(start)
    debug.append(f"[seed] location={start}, rooms={len(room_names)}")

    il = wb.get("item_locations") or {}
    if isinstance(il, dict):
        for item_name, loc_desc in il.items():
            if not item_name or not isinstance(loc_desc, str):
                continue
            ld = loc_desc.lower().strip()
            placed_room: Optional[str] = None
            for rn in room_names:
                rnl = rn.lower()
                if rnl in ld or ld in rnl:
                    placed_room = rn
                    break
            if placed_room is None and start.lower() in ld:
                placed_room = start
            if placed_room is None:
                for w in re.findall(r"[a-z]{4,}", start.lower()):
                    if w in ld:
                        placed_room = start
                        break
            if placed_room:
                state_mgr.place_item_in_room(str(item_name), room=placed_room)
                debug.append(f"[seed] {item_name} @ {placed_room}")
    return debug


def opening_narration_fallback_from_bible(state_mgr: StateManager) -> str:
    """Short player-facing opening when the LLM returns no usable prose (planning dump, empty)."""
    wb = _get_world_bible()
    if not wb:
        return ""
    here = (state_mgr.state.location or "").strip()
    desc = ""
    for loc in wb.get("locations", []) or []:
        if isinstance(loc, dict) and str(loc.get("name", "")).strip().lower() == here.lower():
            desc = (loc.get("description") or "").strip()
            break
    items = state_mgr.list_room_items(here)
    parts = [f"You stand in {here}."]
    if desc:
        parts.append(desc)
    if items:
        parts.append(f"You can see: {', '.join(items)}.")
    for npc in wb.get("npcs", wb.get("key_characters", [])) or []:
        if isinstance(npc, dict) and str(npc.get("location", "")).strip().lower() == here.lower():
            n = npc.get("name", "Someone")
            p = (npc.get("personality") or "").strip()
            parts.append(f"{n} is here{f' ({p})' if p else ''}.")
    return " ".join(parts).strip()


def _strip_untagged_planning_preface(text: str) -> str:
    """
    Drop obvious meta/planning lines at the start when the model forgets delimiters.
    (JSON is extracted separately from the raw reply — do not cut at ```json here.)
    """
    if not text:
        return text
    lines = text.splitlines()
    out: List[str] = []
    started = False
    bad_starts = (
        "the user wants", "i need to check", "current location:", "items in room:",
        "wait,", "looking at the provided", "the prompt says", "action:", "json:",
        "one detail:", "let's refine", "i will output", "state updates:",
        "analyze user", "check constraints", "draft narration", "let's construct",
        "everything looks solid", "output matches", "requirements:", "json format:",
        "game state/context", "crucial constraint", "opening setup (already",
        "output format:", "game setup:", "sentence 1:", "sentence 2:", "sentence 3:",
        "sentence 4:", "sentence 5:", "*   output", "*   game", "*   json",
    )
    for line in lines:
        ls = line.strip()
        if not ls:
            if started:
                out.append(line)
            continue
        lsl = ls.lower()
        if not started:
            if any(lsl.startswith(b) for b in bad_starts):
                continue
            if len(ls) > 55 and not lsl.startswith("the prompt"):
                started = True
        if started:
            out.append(line)
    joined = "\n".join(out).strip()
    return joined if len(joined) > 40 else text


def _is_instruction_echo_line(line: str) -> bool:
    """True when the line is model meta (echoing the prompt / checklist), not in-world narration."""
    s = line.strip()
    if not s:
        return False
    sl = s.lower()
    if sl.startswith("**") and any(
        k in sl
        for k in (
            "output format",
            "json structure",
            "game state",
            "constraint",
            "requirements",
            "analyze",
            "check constraints",
            "draft narration",
        )
    ):
        return True
    if re.match(r"^\d+[\.)]\s+\*\*", s):
        return True
    if re.match(r"^[-*]\s+\*\*", s):
        return True
    if "[output generation]" in sl or "self-correction" in sl or "[final check" in sl:
        return True
    if sl.startswith(("proceeds.", "proceeds", "ready.✅", "ready.")) and len(s) < 120:
        return True
    if "✅" in s and len(s) < 100:
        return True
    if sl.startswith("*(note:") or sl.startswith("*(self-") or sl.startswith("*("):
        return True
    if "opening setup (already in engine state" in sl:
        return True
    return False


def _is_spec_rubric_echo_line(line: str) -> bool:
    """True when the line echoes grading checklists / backtick keys from the prompt, not in-world prose."""
    s = line.strip()
    if not s:
        return False
    sl = s.lower()
    if re.match(r"^sentence\s+\d+:", sl):
        return True
    if sl.endswith("? yes.") or sl.endswith("? correct.") or sl.endswith("? yes") or sl.endswith("? correct"):
        return True
    if re.match(r"^[\*\-•]+\s+", s) and any(
        x in sl
        for x in (
            "output format:",
            "json format correct",
            "state_updates included",
            "images included",
            "room_take not used",
            "connect used for",
            "game setup:",
            "objective:",
            "`move_to`",
            "`connect`",
            "`place_items`",
            "`set_context`",
            "2-5 sentences?",
            "json block inside",
            "json block inside ?",
        )
    ):
        return True
    return False


def _strip_instruction_echo_lines(text: str) -> str:
    """Remove lines that echo system markdown / planning (anywhere in the reply)."""
    if not text:
        return text
    lines = text.splitlines()
    kept = [
        ln
        for ln in lines
        if not _is_instruction_echo_line(ln) and not _is_spec_rubric_echo_line(ln)
    ]
    out = "\n".join(kept).strip()
    if len(out) < 30:
        return text
    # Drop paragraphs that are still mostly meta (double-newline blocks)
    paras = [p.strip() for p in out.split("\n\n") if p.strip()]
    if len(paras) <= 1:
        return out
    junk_markers = (
        "**output format**",
        "**json structure**",
        "game state/context",
        "crucial constraint",
        "check constraints &",
        "let's construct the json",
        "[output generation]",
    )
    filtered: List[str] = []
    for p in paras:
        pl = p.lower()
        if any(m in pl for m in junk_markers):
            continue
        if pl.count("**") > 4 and not re.search(r"\b(you|your|the)\s+[a-z]+", pl[:120], re.I):
            continue
        filtered.append(p)
    if not filtered:
        return out
    return "\n\n".join(filtered).strip()


def _collapse_runaway_repetition(text: str, max_chars: int = 6000) -> str:
    """Some MLX models loop the same planning paragraph dozens of times — keep one copy."""
    if len(text) <= max_chars:
        return text
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paras) < 12:
        return text[:max_chars]
    out: List[str] = []
    seen_recent: List[str] = []
    for p in paras:
        if p in seen_recent[-4:]:
            break
        out.append(p)
        seen_recent.append(p)
        seen_recent = seen_recent[-6:]
        if sum(len(x) for x in out) > max_chars - 200:
            break
    s = "\n\n".join(out).strip()
    return s if s else text[:max_chars]


# ═══════════════════════════════════════════════════════════════════════════════
# GAME LOOP CORE (start_story, build_user_prompt, apply_llm_directives)
# ═══════════════════════════════════════════════════════════════════════════════

def start_story(state_mgr: StateManager, llm: "LLMEngine", image_gen: Optional["MfluxImageGenerator"], theme: Optional[str] = None) -> Tuple[str, List[str]]:
    """Ask the LLM to produce an opening scene and seed initial state via JSON.

    Returns: (opening_narration, generated_image_paths)
    """
    # Prefer theme from world bible when available
    try:
        wb = _get_world_bible()
        chosen_theme = theme or (wb.get("global_theme") if wb else None) or "An atmospheric fantasy in caverns and ruins beneath an ancient forest."
    except Exception:
        chosen_theme = theme or "An atmospheric fantasy in caverns and ruins beneath an ancient forest."
    kickoff = (
        "Start a new adventure. The player has just arrived.\n"
        "DESCRIBE WHAT THE PLAYER SEES when they look around: the room, the atmosphere, "
        "notable objects, exits, and any characters present. Write 3-5 vivid sentences.\n"
        "Set an initial location with at least 2 exits. Place a useful starting item "
        "(e.g., Lantern, Rope, Map) in the room.\n"
        "IMPORTANT: You MUST include an image prompt in your JSON to illustrate this opening scene.\n\n"
        f"Theme: {chosen_theme}"
    )
    # If a world bible exists, pass key context to guide the LLM
    wb_hint = ""
    try:
        _wb = _get_world_bible()
        if _wb:
            hints = []

            if objs := _wb.get("objectives", []):
                hints.append(f"objectives: {'; '.join(objs[:3])}")

            if npcs := _wb.get("npcs", _wb.get("key_characters", [])):
                npc_info = [f"{n.get('name', 'Someone')} ({n.get('personality', n.get('role', 'NPC'))})" for n in npcs[:2]]
                if npc_info:
                    hints.append(f"NPCs: {', '.join(npc_info)}")

            if monsters := _wb.get("monsters", []):
                monster_info = [f"{m.get('name', 'creature')} ({m.get('difficulty', 'unknown')})" for m in monsters[:2]]
                if monster_info:
                    hints.append(f"monsters: {', '.join(monster_info)}")

            if state_mgr.state.health < 50:
                hints.append(f"player health: {state_mgr.state.health}/100 (wounded)")
            elif state_mgr.state.health < 100:
                hints.append(f"player health: {state_mgr.state.health}/100")

            if locations := _wb.get("locations", []):
                # Include all location descriptions to guide narrative
                loc_descriptions = []
                for loc in locations:
                    if isinstance(loc, dict):
                        name = loc.get("name")
                        desc = loc.get("description")
                        if name and desc:
                            loc_descriptions.append(f"{name}: {desc}")
                    elif isinstance(loc, str):
                        loc_descriptions.append(loc)
                if loc_descriptions:
                    hints.append(f"Locations: {'; '.join(loc_descriptions[:3])}")  # Limit to first 3 to avoid token bloat

            # SIMPLE IMAGE TRACKING (belongs in game state context)
            # Direct connection to existing map/item structures
            if state_mgr.state.rooms_with_images:
                hints.append(f"Rooms with images: {', '.join(state_mgr.state.rooms_with_images[:5])}")
            if state_mgr.state.items_with_images:
                hints.append(f"Items with images: {', '.join(state_mgr.state.items_with_images[:5])}")

            if hints:
                wb_hint = f"\n\nWorld context: {'; '.join(hints)}"
    except Exception:
        wb_hint = ""

    # Pre-seed map, starting room, and items from world bible so opening matches the design doc
    try:
        for line in seed_game_state_from_world_bible(state_mgr):
            print(line)
    except Exception as e:
        print(f"[start_story] seed_game_state_from_world_bible skipped: {e}")

    try:
        if _get_world_bible():
            here = state_mgr.state.location
            ex = state_mgr.state.known_map.get(here, {}).get("exits", [])
            seen = state_mgr.list_room_items(here)
            kickoff += (
                "\n\nOPENING SETUP (already in engine state — describe this faithfully, do not contradict): "
                f"room={here}; exits: {', '.join(ex) if ex else '(use JSON connect if you add exits)'}; "
                f"visible items here now: {', '.join(seen) if seen else '(none seeded)'}. "
                "Then output your ```json``` block with state_updates + images."
            )
    except Exception:
        pass

    llm_text = llm.generate(get_system_instructions() + wb_hint, kickoff)
    final_text, new_images, payloads, tool_logs = apply_llm_directives(state_mgr, llm_text, image_gen)
    # If LLM failed to seed a location, ensure we have at least 'Start'.
    if not state_mgr.state.location:
        state_mgr.move_to("Start")

    ft = (final_text or "").strip()
    if len(ft) < 50:
        fb = opening_narration_fallback_from_bible(state_mgr)
        if fb:
            final_text = fb
            print("[start_story] Used world-bible narration fallback (LLM prose too short or empty)")

    ft = (final_text or "").strip()
    narr_for_img = None
    if ft and len(ft) < 3500 and ft.lower().count("the user wants") < 2:
        narr_for_img = ft

    # ENSURE opening room image: If no images were generated, generate one directly
    # Pass narrative context so image matches what was described (skip if narration looks like CoT)
    if not new_images and state_mgr.state.location and image_gen:
        start_image = generate_room_image_if_needed(
            state_mgr, image_gen, state_mgr.state.location,
            narrative_context=narr_for_img,
        )
        if start_image:
            new_images.append(start_image)

    return final_text or "Your journey begins...", new_images


def _generate_turn_images(
    state_mgr: StateManager,
    updates: Dict[str, Any],
    llm_img_prompts: List[str],
    image_gen: Optional[MfluxImageGenerator],
    debug_lines: List[str],
) -> List[str]:
    """Generate and filter images for the current turn.

    Handles auto-generation for new rooms/items, filtering against cached images,
    theme enforcement, and duplicate prevention. Returns list of new image paths.
    """
    if not image_gen:
        return []

    image_paths: List[str] = []
    img_prompts = list(llm_img_prompts)
    img_prompts_lower = [p.lower() for p in img_prompts if isinstance(p, str)]

    # --- Auto-generate image for new rooms ---
    if isinstance(updates.get("move_to"), str):
        new_room = updates["move_to"]
        if not state_mgr.state.has_room_image(new_room):
            auto_prompt = f"atmospheric fantasy {new_room.lower()}, detailed environment, adventure game location"
            if not any(new_room.lower() in p for p in img_prompts_lower):
                img_prompts.append(auto_prompt)
                debug_lines.append(f"Auto-generating image for room: {new_room}")
        else:
            state_mgr.state.images_reused += 1
            debug_lines.append(f"REUSING existing image for room: {new_room}")

    # --- Auto-generate images for newly placed/taken items ---
    for item in list(updates.get("place_items") or []) + list(updates.get("room_take") or []):
        item_str = str(item)
        already_imaged = any(
            item_str.lower() in img.get("prompt", "").lower()
            for img in state_mgr.state.last_images
        )
        if not already_imaged:
            if not any(item_str.lower() in p for p in img_prompts_lower):
                img_prompts.append(f"detailed close-up of {item_str}, fantasy adventure game item")
                debug_lines.append(f"Auto-generating image for item: {item_str}")

    if not img_prompts:
        return image_paths

    # --- Build set of known items for filtering ---
    known_items: Set[str] = set()
    for v in state_mgr.state.room_items.values():
        known_items.update(str(it) for it in v)
    known_items.update(str(it) for it in state_mgr.state.inventory)
    wb = _get_world_bible() or {}
    for it in wb.get("key_items", []):
        nm = it.get("name") if isinstance(it, dict) else str(it)
        if nm:
            known_items.add(str(nm))

    # --- Helper: check if we already have an image for a subject ---
    def _has_image_for(text: str) -> bool:
        t = str(text).strip().lower()
        if any(t == str(r).strip().lower() for r in state_mgr.state.rooms_with_images):
            state_mgr.state.images_reused += 1
            return True
        if any(t == str(i).strip().lower() for i in state_mgr.state.items_with_images):
            state_mgr.state.images_reused += 1
            return True
        return False

    # --- Filter prompts: skip images we already have ---
    move_room = updates.get("move_to") if isinstance(updates.get("move_to"), str) else None
    target_room = move_room or state_mgr.state.location
    has_room_img = bool(target_room and _has_image_for(str(target_room)))

    filtered_prompts: List[str] = []
    for p in img_prompts:
        if not isinstance(p, str) or not p.strip():
            continue
        lc = p.lower()
        # Skip room prompt if we already have that room's image
        if move_room and move_room.lower() in lc and _has_image_for(move_room):
            debug_lines.append(f"Reuse existing image for room: {move_room}")
            continue
        # Skip all environment prompts if we have a room image (unless close-up)
        if has_room_img:
            is_closeup = any(kw in lc for kw in ("close-up", "close up", "closeup"))
            if not is_closeup:
                debug_lines.append(f"Reuse existing room image for: {target_room}")
                continue
        # Skip item prompts if we already have that item's image
        skip = False
        for it in known_items:
            if it.lower() in lc and _has_image_for(it):
                debug_lines.append(f"Reuse existing image for item: {it}")
                skip = True
                break
        if skip:
            continue
        filtered_prompts.append(p)

    debug_lines.append(f"Image filter: {len(img_prompts) - len(filtered_prompts)} skipped, {len(filtered_prompts)} to generate")

    # --- Generate filtered prompts ---
    theme_suffix = get_theme_suffix()
    generated_rooms_this_batch: Set[str] = set()

    for p in filtered_prompts:
        final_prompt = p.strip()

        # Reuse if LLM passed an existing filename
        base = os.path.basename(final_prompt)
        if base.lower().endswith(('.png', '.jpg', '.jpeg')):
            for cand in [os.path.join(ART_DIR, base), os.path.join(IN_GAME_IMG_DIR, base)]:
                if os.path.exists(cand) and cand not in image_paths:
                    image_paths.append(cand)
                    debug_lines.append(f"[reuse] Existing file: {base}")
                    break
            else:
                # Try room/item fallback
                room_path = get_current_room_image(state_mgr)
                if room_path and room_path not in image_paths:
                    image_paths.append(room_path)
                    debug_lines.append(f"[reuse] Room image fallback: {os.path.basename(room_path)}")
            continue

        # Identify target room/item
        prompt_lower = p.strip().lower()
        target_room_for_prompt = None
        for room_name in state_mgr.state.known_map.keys():
            if room_name.lower() in prompt_lower:
                target_room_for_prompt = room_name
                break

        # Skip duplicate room images within this turn
        if target_room_for_prompt and target_room_for_prompt in generated_rooms_this_batch:
            debug_lines.append(f"Skip duplicate room image this turn: {target_room_for_prompt}")
            continue

        target_item_for_prompt = None
        all_items = set(state_mgr.state.inventory)
        for ri in state_mgr.state.room_items.values():
            all_items.update(ri)
        for item_name in all_items:
            if isinstance(item_name, str) and item_name.lower() in prompt_lower:
                target_item_for_prompt = item_name
                break
        if not target_item_for_prompt:
            for it in wb.get("key_items", []):
                nm = it.get("name") if isinstance(it, dict) else str(it)
                if isinstance(nm, str) and nm.lower() in prompt_lower:
                    target_item_for_prompt = nm
                    break

        # If we can't identify subject and LLM didn't request it, skip
        if not target_room_for_prompt and not target_item_for_prompt:
            if p not in llm_img_prompts:
                debug_lines.append("[skip] Unlinked auto-gen prompt skipped")
                continue

        # Enforce art style
        if theme_suffix and theme_suffix.lower() not in final_prompt.lower():
            final_prompt = f"{final_prompt}. World art style: {theme_suffix}. Strictly adhere to this style."

        debug_lines.append(f"Generating image: {final_prompt[:120]}")
        out_path = image_gen.generate(final_prompt)

        if out_path:
            original_prompt = p.strip().lower()
            # Track as room image
            room_identified = False
            for room_name in state_mgr.state.known_map.keys():
                if room_name.lower() in original_prompt:
                    state_mgr.state.add_room_image(room_name, out_path, p.strip())
                    generated_rooms_this_batch.add(room_name)
                    room_identified = True
                    image_paths.append(out_path)
                    debug_lines.append(f"[track] ROOM: '{room_name}' → {os.path.basename(out_path)}")
                    break
            # Track as item image
            if not room_identified:
                for item_name in all_items:
                    if item_name.lower() in original_prompt:
                        state_mgr.state.add_item_image(item_name, out_path, p.strip())
                        image_paths.append(out_path)
                        debug_lines.append(f"[track] ITEM: '{item_name}' → {os.path.basename(out_path)}")
                        break
                else:
                    debug_lines.append("[skip] Generated image not matched to room/item")
        else:
            debug_lines.append(f"Image generation failed for: {p.strip()[:80]}")

    debug_lines.append(f"Images cached: {len(state_mgr.state.last_images)}")
    return image_paths


def apply_llm_directives(state_mgr: StateManager, text: str, image_gen: Optional[MfluxImageGenerator]) -> Tuple[str, List[str], List[Dict[str, Any]], List[str]]:
    image_paths: List[str] = []
    cleaned_text = text
    debug_lines: List[str] = []
    json_payloads: List[Dict[str, Any]] = []

    # CHANGE (chunk D): snapshot state BEFORE applying directives so we can emit a one-line
    # [SUMMARY] of what changed at the end. Cheap shallow copies — no deep clone needed.
    _pre_state = {
        "location": state_mgr.state.location,
        "inventory": list(state_mgr.state.inventory),
        "flags": list(state_mgr.state.game_flags.keys()),
        "health": state_mgr.state.health,
    }

    # Strip reasoning / channel blocks first (local models often prefix narration with this).
    _pre_think = len(text)
    text = strip_llm_thinking_blocks(text)
    if len(text) + 80 < _pre_think:
        debug_lines.append("[clean] Stripped thinking/channel blocks from LLM output")
    # Legacy tags not covered by strip_llm_thinking_blocks
    think_pat = re.compile(r"<think>[\s\S]*?</think>|<redacted_thinking>[\s\S]*?</redacted_thinking>", re.IGNORECASE)
    if think_pat.search(text):
        text = think_pat.sub("", text).strip()
        debug_lines.append("[clean] Stripped <think> reasoning block from output")
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE).strip()

    # ENHANCED JSON EXTRACTION STRATEGY
    # Use improved extraction that handles partial and malformed JSON

    # 1) Extract fenced code blocks first
    fenced_pat = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    fenced_blocks = []
    for m in fenced_pat.finditer(text):
        fenced_blocks.append((m.start(), m.end()))
        block = m.group(1).strip()
        parsed, error = parse_json_with_fallbacks(block)
        if parsed:
            json_payloads.append(parsed)
            debug_lines.append(f"Extracted JSON from fenced block")
        else:
            debug_lines.append(f"Failed to parse fenced JSON: {error[:50]}")

    # Remove all fenced blocks
    if fenced_blocks:
        fenced_blocks.sort(reverse=True)  # Remove from end to start
        for start, end in fenced_blocks:
            text = text[:start] + text[end:]

    # 2) Extract JSON objects from remaining text using enhanced extraction
    extracted_jsons = extract_json_from_text(text)
    for parsed_json in extracted_jsons:
        json_payloads.append(parsed_json)
        debug_lines.append(f"Extracted JSON object with keys: {list(parsed_json.keys())}")

    # 3) Remove all successfully extracted JSON from text
    # This is more aggressive and comprehensive than before
    recognized_keys = ["state_updates", "images"]

    # Remove complete JSON objects (balanced braces)
    json_object_pattern = re.compile(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', re.DOTALL)
    text = json_object_pattern.sub('', text)

    # Remove JSON-like fragments containing our keys
    for key in recognized_keys:
        # Remove lines starting with our keys in JSON context
        text = re.sub(rf'\n\s*"{key}"?\s*:.*?(?=\n|$)', '', text, flags=re.MULTILINE | re.DOTALL)
        text = re.sub(rf'\n\s*{key}\s*:.*?(?=\n|$)', '', text, flags=re.MULTILINE | re.DOTALL)

    # Clean up remaining JSON fragments and artifacts
    text = re.sub(r'\{\s*"(?:state_updates|images)"[^}]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\{\s*"(?:state_updates|images)"[^\n]*', '', text)
    text = re.sub(r',\s*"images":\s*\[\s*"[^"]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'"images":\s*\[\s*"[^"]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r',\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*,\s*"images":\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\{\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\[\s*"[^"]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\]\s*,?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\}\s*,?\s*$', '', text, flags=re.MULTILINE)

    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()

    cleaned_text = text
    cleaned_text = _strip_untagged_planning_preface(cleaned_text)
    t_echo = _strip_instruction_echo_lines(cleaned_text)
    if len(t_echo) + 20 < len(cleaned_text):
        debug_lines.append("[clean] Stripped instruction-echo / markdown checklist lines from narration")
    cleaned_text = t_echo

    # FLEXIBLE DIRECTIVE FALLBACK SYSTEM
    # If no JSON was found but text contains directive-like patterns, try to extract them
    if not json_payloads:
        fallback_directives = extract_fallback_directives(cleaned_text)
        if fallback_directives:
            json_payloads.append(fallback_directives)
            debug_lines.append(f"Extracted fallback directives: {list(fallback_directives.keys())}")

    # Narration-only fallback — no state changes if all parsing failed
    if not json_payloads:
        debug_lines.append("[WARNING] No valid JSON found — narration only, no state changes")
        # Preserve narrative continuity even without JSON
        if cleaned_text.strip():
            state_mgr.state.story_context = f"[narration-only] {cleaned_text[:200]}"

    # 3) Apply directives
    last_room_moved_to = None  # Track room moves for image display
    image_paths = []  # Initialize image paths list
    
    for directives in json_payloads:
        try:
            # Validate and fix directive schema
            fixed_directives, schema_warnings = validate_and_fix_directives(directives)
            for warning in schema_warnings:
                debug_lines.append(f"[schema] {warning}")

            directives = fixed_directives  # Use the fixed version
            updates = directives.get("state_updates", {}) if isinstance(directives, dict) else {}

            # IMPORTANT: Process room item actions BEFORE move_to
            # This ensures items can be placed/taken in the current room before moving

            # CHANGE (chunk C): track which rooms got connected this turn so we can detect
            # `move_to` to a room that has no path from the current location. We do not
            # auto-fix the move (LLM remains the GM) — we just log a warning the user can act on.
            _connected_this_turn: Set[str] = set()
            _loc_at_apply = state_mgr.state.location

            # Room connections (can happen anytime)
            if isinstance(updates.get("connect"), list):
                for pair in updates["connect"]:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        a, b = str(pair[0]), str(pair[1])
                        state_mgr.connect_rooms(a, b)
                        _connected_this_turn.add(a)
                        _connected_this_turn.add(b)
                        debug_lines.append(f"tool.connect_rooms: {a} <-> {b}")

            # Room item placement (before take, before move)
            for item in updates.get("place_items", []) or []:
                state_mgr.place_item_in_room(str(item))
                debug_lines.append(f"tool.place_item_in_room: {item} @ {state_mgr.state.location}")

            # Room take (after place, before move) - so items can be placed and taken in same turn
            for item in updates.get("room_take", []) or []:
                taken = state_mgr.remove_item_from_room(str(item))
                if taken:
                    state_mgr.add_item(str(item))
                    debug_lines.append(f"tool.room_take: {item}")
                else:
                    # CHANGE (chunk C): explicit [WARN] with what WAS in the room so the
                    # next turn's prompt context shows the contradiction clearly.
                    here_items = state_mgr.list_room_items()
                    debug_lines.append(
                        f"[WARN] room_take MISS: '{item}' not in room "
                        f"'{_loc_at_apply}' (room has: {here_items or 'nothing'})"
                    )
                    # Don't auto-correct - let LLM learn from its mistakes

            # Now move the player (after all room actions are complete)
            if isinstance(updates.get("move_to"), str):
                target = updates["move_to"]
                # CHANGE (chunk C): check linkage BEFORE applying the move.
                exits_now = list(state_mgr.state.known_map.get(_loc_at_apply, {}).get("exits", []))
                linked = (
                    target == _loc_at_apply
                    or target in exits_now
                    or target in _connected_this_turn
                )
                if not linked and _loc_at_apply:
                    debug_lines.append(
                        f"[WARN] move_to UNLINKED: '{target}' not connected to '{_loc_at_apply}' "
                        f"(known exits: {exits_now or 'none'}); LLM should emit a `connect` pair"
                    )
                state_mgr.move_to(target)
                debug_lines.append(f"tool.move_to: {target}")
                last_room_moved_to = target  # Track the move

            # Inventory updates (can happen anytime)
            # FIX: Detect when LLM sends both add_items and remove_items for the
            # same item — this is a common small-model mistake where it means
            # "take from room" but uses add+remove instead of room_take.
            add_list = [str(i) for i in (updates.get("add_items", []) or [])]
            remove_list = [str(i) for i in (updates.get("remove_items", []) or [])]
            # Items in BOTH lists → treat as room_take (add to inventory, remove from room)
            overlap = set(add_list) & set(remove_list)
            if overlap:
                for item in overlap:
                    state_mgr.add_item(item)
                    state_mgr.remove_item_from_room(item)
                    # CHANGE (chunk C): tag as [WARN] — auto-fixing a model mistake the user
                    # should know about so they can tune the prompt or the model choice.
                    debug_lines.append(
                        f"[WARN] auto-corrected add+remove → room_take('{item}') "
                        "(LLM used wrong tool; prefer 'room_take' for picking items up)"
                    )
                add_list = [i for i in add_list if i not in overlap]
                remove_list = [i for i in remove_list if i not in overlap]
            for item in add_list:
                if item in state_mgr.state.inventory:
                    debug_lines.append(f"[WARN] add_items DUP: '{item}' already in inventory")
                state_mgr.add_item(item)
                debug_lines.append(f"tool.add_item: {item}")
            for item in remove_list:
                if item not in state_mgr.state.inventory:
                    # CHANGE (chunk C): explicit miss so the user knows the LLM tried to remove
                    # something the player never had — common after teleporty narration.
                    debug_lines.append(
                        f"[WARN] remove_items MISS: '{item}' not in inventory "
                        f"({state_mgr.state.inventory or 'empty'})"
                    )
                state_mgr.remove_item(item)
                debug_lines.append(f"tool.remove_item: {item}")
            # Health and notes
            if updates.get("change_health") is not None:
                try:
                    delta = int(updates.get("change_health", 0))
                    state_mgr.change_health(delta)
                    debug_lines.append(f"tool.change_health: {delta} -> {state_mgr.state.health}")
                except Exception:
                    pass
            if isinstance(updates.get("add_note"), str):
                state_mgr.add_note(updates["add_note"]) 
                debug_lines.append("tool.add_note")
            
            # Story context - LLM's narrative memory
            if isinstance(updates.get("set_context"), str):
                state_mgr.state.story_context = updates["set_context"]
                debug_lines.append(f"tool.set_context: {updates['set_context'][:50]}...")
            
            # Game flags - flexible state tracking
            if isinstance(updates.get("set_flag"), dict):
                flag_data = updates["set_flag"]
                if "name" in flag_data:
                    flag_name = str(flag_data["name"])
                    flag_value = flag_data.get("value", True)
                    state_mgr.state.game_flags[flag_name] = flag_value
                    debug_lines.append(f"tool.set_flag: {flag_name} = {flag_value}")
            
            # Store game mechanics as persistent flags
            if isinstance(updates.get("mechanics"), dict):
                mech = updates["mechanics"]
                # Store as a flag with structured data
                mech_id = f"mech_{len(state_mgr.state.game_flags)}"
                state_mgr.state.game_flags[mech_id] = mech
                # Also update context for narrative continuity
                if mech.get("action") and mech.get("effect"):
                    context_update = f"Mechanic: {mech['action']} causes {mech['effect']}"
                    if state_mgr.state.story_context:
                        state_mgr.state.story_context = state_mgr.state.story_context + "; " + context_update
                    else:
                        state_mgr.state.story_context = context_update
                    debug_lines.append(f"tool.mechanics: {mech['action']} -> {mech['effect']}")

            # Advanced game mechanics - conditional actions, timers, chain reactions
            if isinstance(updates.get("conditional_action"), dict):
                cond_action = updates["conditional_action"]
                condition = cond_action.get("condition", "")
                action = cond_action.get("action", "")
                fallback = cond_action.get("fallback", "")

                # Simple condition evaluation (can be extended)
                condition_met = False
                if condition.startswith("has_"):
                    item = condition[4:]  # Remove "has_" prefix
                    condition_met = item in state_mgr.state.inventory
                elif condition.startswith("at_"):
                    room = condition[3:]  # Remove "at_" prefix
                    condition_met = state_mgr.state.location == room
                elif condition in state_mgr.state.game_flags:
                    condition_met = bool(state_mgr.state.game_flags[condition])

                if condition_met and action:
                    # Execute the conditional action — actually change state
                    if action.startswith("unlock_"):
                        target = action[7:]
                        state_mgr.state.game_flags[f"{target}_unlocked"] = True
                        debug_lines.append(f"tool.conditional_action: {condition} met -> unlocked {target}")
                    elif action.startswith("reveal_"):
                        target = action[7:]
                        state_mgr.state.game_flags[f"{target}_revealed"] = True
                        debug_lines.append(f"tool.conditional_action: {condition} met -> revealed {target}")
                    elif action.startswith("move_"):
                        target = action[5:]
                        state_mgr.move_to(target)
                        debug_lines.append(f"tool.conditional_action: {condition} met -> moved to {target}")
                    else:
                        # Generic: set the action string as a flag
                        state_mgr.state.game_flags[action] = True
                        debug_lines.append(f"tool.conditional_action: {condition} met -> flag '{action}' set")
                elif fallback:
                    debug_lines.append(f"tool.conditional_action: {condition} not met -> {fallback}")

            # Timer events for delayed effects
            if isinstance(updates.get("timer_event"), dict):
                timer = updates["timer_event"]
                timer_name = timer.get("name", f"timer_{len(state_mgr.state.game_flags)}")
                duration = timer.get("duration", 1)
                action = timer.get("action", "")
                value = timer.get("value")

                # Store timer as a flag with turn counter
                timer_data = {
                    "type": "timer",
                    "duration": duration,
                    "remaining": duration,
                    "action": action,
                    "value": value,
                    "created_turn": len(state_mgr.state.recent_history)
                }
                state_mgr.state.game_flags[timer_name] = timer_data
                debug_lines.append(f"tool.timer_event: {timer_name} set for {duration} turns -> {action}")

            # Chain reactions for complex multi-step effects
            if isinstance(updates.get("chain_reaction"), dict):
                chain = updates["chain_reaction"]
                trigger = chain.get("trigger", "")
                effects = chain.get("effects", [])

                # Store as a persistent mechanic that can be triggered later
                chain_id = f"chain_{len(state_mgr.state.game_flags)}"
                chain_data = {
                    "type": "chain_reaction",
                    "trigger": trigger,
                    "effects": effects,
                    "active": True
                }
                state_mgr.state.game_flags[chain_id] = chain_data
                debug_lines.append(f"tool.chain_reaction: {trigger} -> {len(effects)} effects stored")

            # Generate images for this turn (extracted to helper for clarity)
            llm_img_prompts = directives.get("images", []) if isinstance(directives, dict) else []
            new_turn_images = _generate_turn_images(
                state_mgr, updates, llm_img_prompts, image_gen, debug_lines
            )
            image_paths.extend(new_turn_images)
        except Exception:
            continue

    # 4) No heuristic item extraction: items are ONLY updated via LLM JSON (place_items/room_take).

    # Add reused room image to display list (if room move happened and image exists)
    if last_room_moved_to:
        # Find existing room image to add to display
        for img_data in state_mgr.state.last_images:
            if img_data.get("type") == "room" and img_data.get("subject") == last_room_moved_to:
                existing_path = img_data.get("path")
                if existing_path and os.path.exists(existing_path) and existing_path not in image_paths:
                    image_paths.append(existing_path)
                    debug_lines.append(f"Added reused room image to display: {os.path.basename(existing_path)}")
                    break

    # Final cleanup of any extra whitespace
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip()
    cleaned_text = _strip_instruction_echo_lines(cleaned_text)
    cleaned_text = _collapse_runaway_repetition(cleaned_text)
    if len(cleaned_text) > 14000:
        cleaned_text = cleaned_text[:14000].rstrip() + "\n…"
        debug_lines.append("[clean] Truncated extremely long narration/planning spill")

    # CHANGE (chunk D): emit one [SUMMARY] line capturing the turn deltas. Always at the END
    # so all earlier debug_lines (including [WARN]s appended by guards in chunk C) are counted.
    _post_state = {
        "location": state_mgr.state.location,
        "inventory": list(state_mgr.state.inventory),
        "flags": list(state_mgr.state.game_flags.keys()),
        "health": state_mgr.state.health,
    }
    _warn_count = sum(1 for ln in debug_lines if ln.startswith("[WARN]") or ln.startswith("[WARNING]"))
    debug_lines.append(_format_turn_summary(_pre_state, _post_state, _warn_count, len(json_payloads)))

    return cleaned_text, image_paths, json_payloads, debug_lines


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def check_mlx_available() -> bool:
    if platform.system() != "Darwin":
        print("MLX requires macOS with Apple Silicon.")
        return False
    if not os.path.isdir(MLX_MODELS_DIR):
        print(f"MLX models directory not found: {MLX_MODELS_DIR}")
        return False
    return True


def list_mlx_models() -> List[str]:
    if not os.path.isdir(MLX_MODELS_DIR):
        print(f"MLX models directory not found: {MLX_MODELS_DIR}")
        return []
    names: List[str] = []
    for entry in sorted(os.listdir(MLX_MODELS_DIR)):
        full = os.path.join(MLX_MODELS_DIR, entry)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
            names.append(entry)
    return names


def list_ollama_model_choices() -> List[str]:
    """Return dropdown-ready ids: ollama:<tag> for each running Ollama model, plus preferred tag if missing."""
    # CHANGE: surface Ollama tags alongside MLX dirs; prefix avoids collision with MLX folder names.
    if ollama_pkg is None:
        return []
    raw: List[str] = []
    try:
        response = ollama_pkg.list()
        if hasattr(response, "models"):
            for model in response.models:
                if hasattr(model, "model"):
                    raw.append(model.model)
                else:
                    raw.append(str(model))
        elif isinstance(response, dict) and "models" in response:
            for model in response["models"]:
                if isinstance(model, dict) and "name" in model:
                    raw.append(model["name"])
                elif isinstance(model, dict) and "model" in model:
                    raw.append(model["model"])
                else:
                    raw.append(str(model))
    except Exception as e:
        print(f"[ollama] list failed: {e}")
        raw = []

    prefixed = [OLLAMA_CHOICE_PREFIX + n for n in sorted(set(raw))]
    preferred = OLLAMA_CHOICE_PREFIX + OLLAMA_PREFERRED_MODEL
    if preferred not in prefixed:
        prefixed.insert(0, preferred)
    else:
        prefixed = [preferred] + [x for x in prefixed if x != preferred]
    return prefixed


def diffusion_models_root() -> str:
    env_root = os.environ.get("DIFFUSION_MODELS_DIR")
    if env_root and os.path.isdir(env_root):
        return env_root

    # Check common locations
    candidates = [
        "/Users/jonathanrothberg/Diffusion_Models",  # macOS
        "/home/jonathan/Models_Diffusers",          # Linux
        "/data/Diffusion_Models",                   # Alternative Linux
        "./Diffusion_Models",                       # Relative path
    ]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate

    # Fallback
    default_model_root = "/Users/jonathanrothberg" if sys.platform == "darwin" else "/data"
    return os.path.join(default_model_root, "Diffusion_Models")


def list_local_diffusers() -> List[str]:
    """List FLUX models available locally for MFLUX."""
    root = diffusion_models_root()
    if not os.path.isdir(root):
        return []
    try:
        names = [n for n in os.listdir(root)
                 if os.path.isdir(os.path.join(root, n)) and n.startswith("FLUX")]
        return sorted(names)
    except Exception:
        return []


def combined_diffuser_choices(include_skip: bool = True) -> List[str]:
    local = list_local_diffusers()
    return local if local else ["FLUX2-klein-9B-mlx-8bit"]


def select_llm_interactively() -> str:
    mlx = list_mlx_models() if check_mlx_available() else []
    ollama_opts = list_ollama_model_choices()
    models = ollama_opts + mlx
    if not models:
        print(f"No LLM choices: MLX dirs in {MLX_MODELS_DIR} or Ollama models (ollama list).")
        raise SystemExit(1)
    print("Select an LLM (Ollama entries first; same list as GUI):")
    for idx, name in enumerate(models, 1):
        print(f"  {idx}. {name}")
    choice = input("Enter a number (default 1): ").strip() or "1"
    try:
        idx = max(1, min(len(models), int(choice))) - 1
    except Exception:
        idx = 0
    return models[idx]


def maybe_select_diffuser_interactively() -> Optional[str]:
    print("Image generation is optional. Select an MFLUX model or press Enter to skip:")
    candidates = combined_diffuser_choices(include_skip=True)
    for i, mid in enumerate(candidates, 1):
        label = mid or "(Skip images)"
        print(f"  {i}. {label}")
    default_idx = len(candidates)
    choice = input(f"Enter a number (default {default_idx}): ").strip() or str(default_idx)
    try:
        idx = max(1, min(len(candidates), int(choice))) - 1
    except Exception:
        idx = len(candidates) - 1
    model_id = candidates[idx]
    return model_id or None


# CHANGE (chunk J): classify the player's intent so we can attach the most relevant
# few-shot example to the prompt (better than dumping all examples every turn).
def _classify_intent(player_input: str) -> str:
    text = (player_input or "").strip().lower()
    if not text:
        return "other"
    first = text.split()[0] if text.split() else text
    move_words = {"go", "move", "enter", "head", "travel", "north", "south", "east",
                  "west", "up", "down", "n", "s", "e", "w", "u", "d", "climb"}
    if first in move_words:
        return "move"
    take_words = {"take", "get", "grab", "pick"}
    if first in take_words or text.startswith("pick up"):
        return "take"
    use_words = {"use", "open", "unlock", "light", "burn", "place", "insert", "wear", "drink", "eat", "read"}
    if first in use_words:
        return "use"
    talk_words = {"talk", "ask", "say", "greet", "tell", "answer"}
    if first in talk_words:
        return "talk"
    combat_words = {"attack", "fight", "kill", "hit", "swing", "stab", "slash", "shoot", "throw", "wave"}
    if first in combat_words:
        return "combat"
    return "other"


def _few_shot_for_intent(intent: str) -> str:
    """Short, focused example showing the JSON for ONE common turn type. Less is more
    on small models — a single targeted example outperforms 3 generic ones."""
    examples = {
        "move": (
            "Example for movement (creates the connection if new):\n"
            "Narration: You step from the cavern into a dim hall lit by a single torch.\n"
            "```json\n"
            "{\"state_updates\": {\"move_to\": \"Echoing Hall\", "
            "\"connect\": [[\"Cavern\", \"Echoing Hall\"]], "
            "\"set_context\": \"Moved to Echoing Hall.\"}, "
            "\"images\": [\"echoing hall lit by a torch\"]}\n"
            "```"
        ),
        "take": (
            "Example for picking up an item (use room_take, NEVER add+remove):\n"
            "Narration: You crouch and lift the iron key from the dust.\n"
            "```json\n"
            "{\"state_updates\": {\"room_take\": [\"Iron Key\"], "
            "\"set_context\": \"Picked up Iron Key.\"}, "
            "\"images\": [\"adventurer lifting an iron key from dusty stone\"]}\n"
            "```"
        ),
        "use": (
            "Example for USING an item (consume with remove_items, set a flag):\n"
            "Narration: You insert the iron key. The lock clicks open.\n"
            "```json\n"
            "{\"state_updates\": {\"remove_items\": [\"Iron Key\"], "
            "\"set_flag\": {\"name\": \"door_open\", \"value\": true}, "
            "\"set_context\": \"Iron Key used; door is open.\"}, "
            "\"images\": [\"ancient door swinging open\"]}\n"
            "```"
        ),
        "talk": (
            "Example for talking to an NPC (no inventory change unless they give you something):\n"
            "Narration: The hermit nods and presses a small lantern into your palm.\n"
            "```json\n"
            "{\"state_updates\": {\"add_items\": [\"Lantern\"], "
            "\"set_context\": \"Hermit gave you a lantern.\"}, "
            "\"images\": [\"old hermit handing a lantern to a traveler\"]}\n"
            "```"
        ),
        "combat": (
            "Example for combat (change_health for damage; remove the foe via set_flag):\n"
            "Narration: You wave the torch; the troll flinches and flees.\n"
            "```json\n"
            "{\"state_updates\": {\"change_health\": -5, "
            "\"set_flag\": {\"name\": \"river_troll_defeated\", \"value\": true}, "
            "\"place_items\": [\"Map\"], "
            "\"set_context\": \"Drove off the river troll; map dropped.\"}, "
            "\"images\": [\"troll fleeing into the dark\"]}\n"
            "```"
        ),
    }
    return examples.get(intent, "")


def build_user_prompt(state_mgr: StateManager, player_input: str) -> str:
    state = state_mgr.state
    current_items = state.room_items.get(state.location, [])
    current_exits = state.known_map.get(state.location, {}).get("exits", [])

    # Trim known_map: current room + adjacent rooms only (saves tokens for small models)
    adjacent_rooms = set(current_exits) | {state.location}
    trimmed_map = {}
    for room_name, room_data in state.known_map.items():
        if room_name in adjacent_rooms:
            trimmed_map[room_name] = room_data
    # Cap at 5 entries even after filtering
    if len(trimmed_map) > 5:
        trimmed_map = dict(list(trimmed_map.items())[:5])

    # Cap game_flags at 10 most recent entries
    trimmed_flags = state.game_flags
    if len(trimmed_flags) > 10:
        trimmed_flags = dict(list(trimmed_flags.items())[-10:])

    # Build compact context — no debug_info, no image lists, no visited_rooms
    context: Dict[str, Any] = {
        "location": state.location,
        "health": state.health,
        "inventory": state.inventory,
        "current_exits": current_exits,
        "current_room_items": current_items,
        "known_map": trimmed_map,
        "story_context": state.story_context,
        "recent_conversation": state.recent_history[-2:],
        "notes": state.notes[-5:] if len(state.notes) > 5 else state.notes,
        "game_flags": trimmed_flags,
    }

    # Only include non-default fields when they have useful info
    if state.player_name:
        context["player_name"] = state.player_name

    wb = _get_world_bible()
    if wb:
        wc = wb.get("win_condition")
        if wc:
            context["win_condition"] = wc

    # Health warning for LLM narrative guidance
    if state.health <= 0:
        context["WARNING"] = "Player is dead! Describe their demise. Do not allow further actions."
    elif state.health <= 20:
        context["WARNING"] = "Player health is critically low! Create tension but give a fair chance to survive."

    # CHANGE (chunk N): only build the heavy world-bible cue block on the FIRST turn the
    # player is in this room. On subsequent turns the description is already in the
    # LLM's recent_history, so re-injecting it is pure token waste on a 30B-class model.
    # We always include light cues (NPC / monster / riddle / item-here) because those
    # are what the model needs to act on this turn.
    first_visit = (state.last_described_room or "").lower() != (state.location or "").lower()

    # World bible context cues (compact, location-relevant excerpts)
    bible_line = ""
    try:
        if wb:
            cues = []

            if first_visit:
                # Current location description (only on first visit to this room)
                for loc in wb.get("locations", []):
                    if isinstance(loc, dict) and loc.get("name", "").lower() == state.location.lower():
                        desc = loc.get("description")
                        if desc:
                            cues.append(f"Location: {desc}")
                        break

            # NPCs in current location
            for npc in wb.get("npcs", wb.get("key_characters", [])):
                if npc.get("location", "").lower() == state.location.lower():
                    provides = npc.get('provides', '')
                    cues.append(f"NPC here: {npc.get('name')} - {npc.get('personality', 'mysterious')}")
                    if provides:
                        cues.append(f"  (can provide: {provides})")

            # Monsters in current location
            for monster in wb.get("monsters", []):
                if monster.get("location", "").lower() == state.location.lower():
                    cues.append(f"Monster here: {monster.get('name')} ({monster.get('difficulty', 'dangerous')})")
                    weakness = monster.get('weakness', '')
                    if weakness:
                        cues.append(f"  (weakness: {weakness})")

            # Riddles/puzzles in current location
            for riddle in wb.get("riddles", []):
                if riddle.get("location", "").lower() == state.location.lower():
                    cues.append(f"Puzzle: {riddle.get('hint', 'something mysterious')}")
                    if riddle.get('reward'):
                        cues.append(f"  (solving grants: {riddle.get('reward')})")

            # Mechanics in current location
            for mech in wb.get("mechanics", []):
                if mech.get("location", "").lower() == state.location.lower():
                    cues.append(f"Mechanic: {mech.get('action')} -> {mech.get('effect')}")

            # Items expected at this location
            for item, loc_desc in wb.get("item_locations", {}).items():
                if state.location.lower() in loc_desc.lower() and item not in state.inventory:
                    item_info = next((i for i in wb.get("key_items", []) if i.get("name") == item), {})
                    if item_info:
                        cues.append(f"Item available: {item} - {item_info.get('purpose', 'useful item')}")

            # Current objective — track via game_flags, not hardcoded item names
            if objectives := wb.get("objectives", []):
                completed = sum(
                    1 for i in range(len(objectives))
                    if state.game_flags.get(f"objective_{i}", False)
                )
                current_obj_idx = min(completed, len(objectives) - 1)
                cues.append(f"Current goal: {objectives[current_obj_idx]}")

            # Progression hints when player seems stuck
            if len(state.inventory) < 2 and (hints := wb.get("progression_hints", [])):
                cues.append(f"Hint: {hints[0]}")

            if cues:
                bible_line = "\nWorld context:\n" + "\n".join(f"  {c}" for c in cues)
    except Exception:
        bible_line = ""

    # CHANGE (chunk O): engine notes injection is OFF by default — local 30B models do well
    # with the system prompt + recent_history alone. Set ADV_INJECT_ENGINE_NOTES=1 to
    # re-enable. Notes are still ALWAYS captured for [SUMMARY] / debug; this flag only
    # controls whether they're pushed back into the model's user prompt.
    notes_block = ""
    pending_notes = list(getattr(state, "engine_notes_for_next_turn", []) or [])
    if pending_notes and ADV_INJECT_ENGINE_NOTES:
        notes_block = (
            "\nEngine notes from your last turn — the player saw the result, fix this now:\n"
            + "\n".join(f"  - {n}" for n in pending_notes[:6])
            + "\n"
        )
    # Either way, consume them so they only get a chance to appear once.
    state.engine_notes_for_next_turn = []

    # CHANGE (chunk N): few-shot is added ONLY when it's likely to help — first few turns
    # of the session OR when last turn produced a [WARN]. After that, the model has the
    # JSON contract from the system prompt + its own recent_history; extra examples are
    # token waste and can crowd creative narration.
    fewshot_block = ""
    needs_fewshot = (
        getattr(state, "session_turns", 0) < 3
        or bool(pending_notes)
    )
    if needs_fewshot:
        intent = _classify_intent(player_input)
        fewshot = _few_shot_for_intent(intent)
        if fewshot:
            fewshot_block = f"\n{fewshot}\n"

    header = (
        "Game State (read-only; update via JSON directives only):\n" +
        json.dumps(context, ensure_ascii=False, indent=2) +
        bible_line +
        notes_block +
        fewshot_block +
        "\n---\n"
    )

    # CHANGE (chunk N): mark this room as described so the next turn skips the heavy
    # description block. Only set AFTER we've decided what to inject this turn.
    state.last_described_room = state.location or ""

    return header + f"Player says: {player_input}"


# CHANGE (chunk G): structured win_condition. The canonical shape stored on WORLD_BIBLE is now
#   "win_condition": {
#       "required_items":   [str, ...],   # ALL must be in inventory to win
#       "required_location": Optional[str],  # if set, player must be there
#       "description":      str,          # human-readable for the narrator + UI
#   }
# Old string-form bibles are normalized in place on first read (saves agreed throwaway).
# `_parse_win_condition` is now a pure normalizer; `check_win_condition` does direct struct
# checks and no longer does substring matching.
def _parse_win_condition(world_bible: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize world_bible["win_condition"] into the structured form (in place) and return it."""
    wc = world_bible.get("win_condition")

    # Already structured: trust required_items / required_location, but coerce types.
    if isinstance(wc, dict):
        items = wc.get("required_items") or []
        if isinstance(items, str):
            items = [items]
        items = [str(x).strip() for x in items if str(x).strip()]
        loc = wc.get("required_location")
        loc = str(loc).strip() if loc else None
        desc = str(wc.get("description") or "").strip()
        if not desc:
            # Synthesize a description from items + location.
            bits: List[str] = []
            if items:
                bits.append(f"have {', '.join(items)}")
            if loc:
                bits.append(f"reach {loc}")
            desc = "Win when you " + " and ".join(bits) + "." if bits else "Complete the adventure."
        normalized = {"required_items": items, "required_location": loc, "description": desc}
        world_bible["win_condition"] = normalized
        return normalized

    # Free-form string: heuristic conversion to struct (one-time, then stored back).
    raw = str(wc or "").strip()
    items: List[str] = []
    loc: Optional[str] = None
    if raw:
        wc_lower = raw.lower()
        for ki in world_bible.get("key_items", []) or []:
            if not isinstance(ki, dict):
                continue
            name = str(ki.get("name") or "").strip()
            if name and name.lower() in wc_lower:
                items.append(name)
        best_len = 0
        for entry in world_bible.get("locations", []) or []:
            if not isinstance(entry, dict):
                continue
            nm = str(entry.get("name") or "").strip()
            if nm and nm.lower() in wc_lower and len(nm) > best_len:
                loc = nm
                best_len = len(nm)
    normalized = {
        "required_items": items,
        "required_location": loc,
        "description": raw or "Complete the adventure.",
    }
    world_bible["win_condition"] = normalized
    return normalized


# CHANGE (chunk K): end-of-session post-mortem. The "rich teaching" deliverable: this
# is what tells the user (a) how the player did, and (b) which world-bible problems
# explain why. Triggered on win, death, quit, or restart. NEVER prints during normal
# play — only at end-of-session.
def format_session_postmortem(state_mgr: StateManager) -> str:
    state = state_mgr.state
    wb = _get_world_bible()
    lines: List[str] = []
    lines.append("═══════════════ SESSION POST-MORTEM ═══════════════")
    lines.append(f"Turns played: {getattr(state, 'session_turns', 0)}")
    lines.append(f"Final HP: {state.health}/100   Final location: {state.location or '?'}")

    if wb:
        all_rooms = [
            str((l or {}).get("name", "")).strip()
            for l in (wb.get("locations") or [])
            if isinstance(l, dict)
        ]
        all_rooms = [r for r in all_rooms if r]
        explored = sorted(set(state.known_map.keys()) & set(all_rooms))
        lines.append(f"Rooms explored: {len(explored)} / {len(all_rooms)}")
        if all_rooms and len(explored) < len(all_rooms):
            missed = [r for r in all_rooms if r not in set(explored)]
            lines.append(f"  unvisited: {', '.join(missed[:8])}")

        all_items = [
            str((i or {}).get("name", "")).strip()
            for i in (wb.get("key_items") or [])
            if isinstance(i, dict)
        ]
        all_items = [i for i in all_items if i]
        held = [i for i in all_items if i in state.inventory]
        lines.append(f"Items in inventory: {len(held)} / {len(all_items)}")
        if all_items and len(held) < len(all_items):
            missed_items = [i for i in all_items if i not in held]
            lines.append(f"  missing: {', '.join(missed_items[:8])}")

        chain = wb.get("solution_chain") or []
        steps_total = len(chain) if isinstance(chain, list) else 0
        steps_done = len(getattr(state, "chain_steps_completed", []) or [])
        lines.append(f"Solution chain: {steps_done} / {steps_total} steps reached")
        comp = getattr(state, "chain_steps_completed", []) or []
        if comp and isinstance(chain, list):
            last_idx = max(comp)
            if 0 <= last_idx < len(chain) and isinstance(chain[last_idx], dict):
                step_loc = chain[last_idx].get("location", "?")
                step_res = chain[last_idx].get("result", "")
                lines.append(f"  last reached: step {last_idx + 1} at {step_loc} ({step_res})")

    # Warnings histogram — what kept going wrong this session
    warnings = getattr(state, "warnings_history", []) or []
    if warnings:
        bucket: Dict[str, int] = {}
        for w in warnings:
            m = re.match(r"^\[(WARN|WARNING)\]\s*(.*)$", w)
            tail = (m.group(2) if m else w).strip()
            cat_tokens = tail.split()[:2]
            cat = " ".join(cat_tokens).rstrip(":") or "unknown"
            bucket[cat] = bucket.get(cat, 0) + 1
        lines.append(f"Warnings this session ({len(warnings)} total):")
        for cat, n in sorted(bucket.items(), key=lambda x: -x[1])[:8]:
            lines.append(f"  - {cat}: {n}")

    # World-bible solvability gaps that may have explained the warnings
    if wb:
        try:
            _ok, gaps, _report = validate_world_bible_solvability(wb)
            if gaps:
                lines.append("World-bible issues to fix (top 5):")
                for g in gaps[:5]:
                    lines.append(f"  - {g}")
        except Exception:
            pass

    lines.append("═══════════════════════════════════════════════════")
    return "\n".join(lines)


def check_win_condition(state_mgr: StateManager) -> Optional[str]:
    """Check if the win condition from the world bible is met. Returns a congrats message or None."""
    wb = _get_world_bible()
    if not wb:
        return None

    state = state_mgr.state

    # Explicit LLM-driven completion still wins (lets the narrator end on a story beat).
    if state.game_flags.get("game_complete") or state.game_flags.get("game_won"):
        struct = _parse_win_condition(wb)
        return (
            f"**VICTORY!** {struct.get('description') or 'You have completed the adventure!'}\n"
            f"Congratulations, {state.player_name or 'adventurer'}!"
        )

    # CHANGE (chunk G): direct struct check — no substring fallback.
    struct = _parse_win_condition(wb)
    items_needed = struct.get("required_items") or []
    loc_needed = struct.get("required_location")
    if not items_needed and not loc_needed:
        return None

    items_ok = all(item in state.inventory for item in items_needed) if items_needed else True
    loc_ok = (loc_needed is None) or (
        bool(state.location) and state.location.strip().lower() == loc_needed.strip().lower()
    )
    if items_ok and loc_ok:
        return (
            f"**VICTORY!** {struct.get('description') or 'You have completed the adventure!'}\n"
            f"Congratulations, {state.player_name or 'adventurer'}!"
        )
    return None


# CHANGE (fastpath naming bugfix): normalize item tokens so a player typing "the key"
# still matches an engine item named "runic_key" (LLM-generated bibles often store
# names in snake_case while the narrator writes them with spaces and articles).
_ARTICLE_RE = re.compile(r"^(the|a|an|some|my|that|this)\s+", re.IGNORECASE)


def _normalize_item_token(s: str) -> str:
    """Lowercase, drop leading article ('the', 'a', 'an', 'some', 'my'), unify
    underscores↔spaces, and strip apostrophes / surrounding punctuation. Used ONLY for
    matching — the original item name is what we display."""
    if not s:
        return ""
    t = s.strip().strip("\"' .,!?")
    t = _ARTICLE_RE.sub("", t).strip()
    # Drop apostrophes so "wizard's staff" ↔ "wizards_staff" match.
    t = t.replace("'", "").replace("’", "")
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).lower()
    return t


def _split_item_list(s: str) -> List[str]:
    """Split a multi-item command target on commas, ' and ', '&', or ' plus '.
    'pick up the key, lantern, and flint' → ['the key', 'lantern', 'flint']."""
    if not s:
        return []
    parts = re.split(r"\s*(?:,|\band\b|&|\bplus\b)\s*", s, flags=re.IGNORECASE)
    out = [p.strip() for p in parts if p and p.strip()]
    return out or ([s.strip()] if s.strip() else [])


def _match_item(target: str, candidates: List[str]) -> Optional[str]:
    """Best-effort match of `target` against an item name list. Tries exact,
    normalized exact, normalized substring, then word-token containment."""
    if not target or not candidates:
        return None
    t_norm = _normalize_item_token(target)
    if not t_norm:
        return None
    # Exact (case-insensitive, raw)
    for c in candidates:
        if c.lower() == target.lower():
            return c
    # Normalized exact
    for c in candidates:
        if _normalize_item_token(c) == t_norm:
            return c
    # Normalized substring (either direction)
    for c in candidates:
        cn = _normalize_item_token(c)
        if t_norm in cn or cn in t_norm:
            return c
    # Word-token containment: every word of target is a word in candidate
    t_tokens = set(t_norm.split())
    if t_tokens:
        for c in candidates:
            c_tokens = set(_normalize_item_token(c).split())
            if t_tokens.issubset(c_tokens):
                return c
    return None


# CHANGE (chunk I): match free-form direction text against the current room's exits.
def _match_exit(text: str, exits: List[str]) -> Optional[str]:
    """Return the best-matching exit name from `exits`, or None."""
    if not exits:
        return None
    t = (text or "").strip().lower()
    if not t:
        return None
    for ex in exits:
        if ex.lower() == t:
            return ex
    for ex in exits:
        if t in ex.lower() or ex.lower() in t:
            return ex
    # Cardinal letters or short prefixes (e.g. "n" -> any exit starting with N)
    if len(t) <= 5:
        for ex in exits:
            if ex.lower().startswith(t[0]):
                return ex
    return None


# CHANGE (chunk I): deterministic engine-level commands. Any time we can resolve the
# player's input WITHOUT the LLM, we should — it is faster, cheaper, and never
# contradicts state. Returns (narration, debug_lines) when handled, None otherwise.
# CHANGE (play robustness fix): bookkeeping that MUST run on every player turn,
# whether the LLM narrated it or the fastpath resolved it. Increments session_turns
# (used by post-mortem) and advances chain_steps_completed when the player is at a
# step's location AND holds its required_item. Cheap, idempotent, no LLM call.
def _post_turn_bookkeeping(state_mgr: StateManager) -> None:
    try:
        state_mgr.state.session_turns += 1
        wb = _get_world_bible()
        if not wb:
            return
        chain = wb.get("solution_chain") or []
        cur_loc = (state_mgr.state.location or "").lower()
        inventory_lower = {i.lower() for i in state_mgr.state.inventory}
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue
            sl = str(step.get("location") or "").strip()
            req = step.get("requires_item")
            holds = (
                not isinstance(req, str)
                or not req.strip()
                or req.strip().lower() in inventory_lower
            )
            if sl and cur_loc == sl.lower() and holds:
                if i not in state_mgr.state.chain_steps_completed:
                    state_mgr.state.chain_steps_completed.append(i)
    except Exception:
        pass


def try_local_command(state_mgr: StateManager, user_text: str) -> Optional[Tuple[str, List[str]]]:
    if not user_text:
        return None
    raw = user_text.strip()
    text = raw.lower()
    if not text:
        return None
    debug: List[str] = []
    state = state_mgr.state

    parts = text.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""

    # ── help ──────────────────────────────────────────────────────────────
    if cmd in ("help", "?"):
        debug.append("[fastpath] help")
        return (
            "**Engine commands (no LLM call, no token cost):**\n"
            "- `look` / `l` — describe your surroundings\n"
            "- `examine X` / `x X` / `look at X` — inspect an item, NPC, or monster you can see\n"
            "- `inventory` / `inv` / `i` — list what you carry\n"
            "- `take X` / `get X` / `pick up X` — pick up an item from the room\n"
            "- `drop X` / `put down X` — set an item down in the room\n"
            "- `go X` / `enter X` / `move X` — travel to an exit named X\n"
            "- `n/s/e/w/up/down` — travel to an exit whose name starts with that letter\n"
            "- `map` / `m` — show the known map\n"
            "- `wait` — pass time\n"
            "- anything else (combat, talk, riddle answers, story actions) goes to the storyteller LLM.",
            debug,
        )

    # ── inventory ─────────────────────────────────────────────────────────
    if cmd in ("inventory", "inv", "i"):
        debug.append("[fastpath] inventory")
        inv = state.inventory or []
        return (f"You are carrying: {', '.join(inv)}." if inv else "Your hands are empty."), debug

    # ── map ───────────────────────────────────────────────────────────────
    if cmd in ("map", "m"):
        debug.append("[fastpath] map")
        return state_mgr.describe_map() or "You have not explored anywhere yet.", debug

    # ── wait ──────────────────────────────────────────────────────────────
    if cmd == "wait":
        debug.append("[fastpath] wait")
        return "You pause for a moment, listening to the silence.", debug

    # ── look ──────────────────────────────────────────────────────────────
    if text in ("look", "l", "look around", "examine room"):
        debug.append("[fastpath] look")
        return (handle_look_command(state_mgr) or f"You stand in **{state.location}**."), debug

    # ── examine X ─────────────────────────────────────────────────────────
    examine_target: Optional[str] = None
    if cmd in ("examine", "x", "inspect"):
        examine_target = arg
    elif text.startswith("look at "):
        examine_target = text[len("look at "):].strip()
    if examine_target is not None:
        if not examine_target:
            return "Examine what?", debug
        # CHANGE (fastpath naming bugfix): use _match_item so "the key" / "lantern"
        # find "runic_key" / "elven_lantern". Articles + snake_case tolerance.
        target_l = examine_target.lower()
        wb = _get_world_bible()

        # Inventory
        inv_match = _match_item(examine_target, list(state.inventory))
        if inv_match:
            purpose = ""
            if wb:
                for ki in wb.get("key_items", []) or []:
                    if isinstance(ki, dict) and ki.get("name", "").lower() == inv_match.lower():
                        purpose = ki.get("purpose", "")
                        break
            debug.append(f"[fastpath] examine inv: {inv_match}")
            return f"**{inv_match}** (carried). " + (purpose or "It feels solid in your hand."), debug

        # Room items
        room_items = state.room_items.get(state.location, [])
        room_match = _match_item(examine_target, list(room_items))
        if room_match:
            debug.append(f"[fastpath] examine room: {room_match}")
            return f"**{room_match}** is here. Try `take {room_match}` to pick it up.", debug

        # NPCs / monsters in this room
        if wb:
            for npc in wb.get("npcs", []) or []:
                if not isinstance(npc, dict):
                    continue
                if target_l in (npc.get("name", "").lower()) and (
                    npc.get("location", "").lower() == state.location.lower()
                ):
                    debug.append(f"[fastpath] examine npc: {npc.get('name')}")
                    return (
                        f"**{npc.get('name')}** — {npc.get('personality','mysterious')}. "
                        f"Can provide: {npc.get('provides','?')}.",
                        debug,
                    )
            for mon in wb.get("monsters", []) or []:
                if not isinstance(mon, dict):
                    continue
                if target_l in (mon.get("name", "").lower()) and (
                    mon.get("location", "").lower() == state.location.lower()
                ):
                    debug.append(f"[fastpath] examine monster: {mon.get('name')}")
                    return (
                        f"**{mon.get('name')}** ({mon.get('difficulty','dangerous')}). "
                        f"Weakness: {mon.get('weakness','?')}.",
                        debug,
                    )

        # Not found locally — let the LLM narrate.
        return None

    # ── take / get / pick up ─────────────────────────────────────────────
    # CHANGE (fastpath naming bugfix): supports multi-item lists ("take A, B and C"),
    # leading articles ("the key"), and snake_case item names ("runic_key" matches "key").
    take_target: Optional[str] = None
    if cmd in ("take", "get", "grab"):
        take_target = arg
    elif text.startswith("pick up "):
        take_target = text[len("pick up "):].strip()
    if take_target is not None:
        if not take_target:
            return "Take what?", debug
        room_items = list(state.room_items.get(state.location, []))
        pieces = _split_item_list(take_target)
        taken: List[str] = []
        misses: List[str] = []
        for piece in pieces:
            match = _match_item(piece, room_items)
            if match and match not in taken:
                state_mgr.remove_item_from_room(match)
                state_mgr.add_item(match)
                taken.append(match)
                debug.append(f"[fastpath] tool.room_take: {match}")
                room_items.remove(match)  # avoid double-matching same item to two requests
            elif not match:
                misses.append(piece)

        if not taken and misses:
            still_here = state.room_items.get(state.location, [])
            debug.append(f"[WARN] [fastpath] take MISS: {misses} not in room={still_here}")
            target_show = ", ".join(misses)
            return (
                f"You don't see '{target_show}' here. "
                + (f"You see: {', '.join(still_here)}." if still_here else "The room is empty.")
            ), debug

        bits: List[str] = []
        if len(taken) == 1:
            bits.append(f"You pick up the **{taken[0]}**.")
        elif taken:
            bits.append(f"You pick up: **{', '.join(taken)}**.")
        if misses:
            still_here = state.room_items.get(state.location, [])
            for m in misses:
                debug.append(f"[WARN] [fastpath] take MISS (partial): '{m}' not in room={still_here}")
            bits.append(
                f"(You don't see '{', '.join(misses)}' here"
                + (f"; visible: {', '.join(still_here)}." if still_here else ".")
                + ")"
            )
        return " ".join(bits), debug

    # ── drop ──────────────────────────────────────────────────────────────
    # CHANGE (fastpath naming bugfix): same multi-item / article / snake_case tolerance.
    drop_target: Optional[str] = None
    if cmd in ("drop", "leave"):
        drop_target = arg
    elif text.startswith("put down "):
        drop_target = text[len("put down "):].strip()
    if drop_target is not None:
        if not drop_target:
            return "Drop what?", debug
        inventory = list(state.inventory)
        pieces = _split_item_list(drop_target)
        dropped: List[str] = []
        misses: List[str] = []
        for piece in pieces:
            match = _match_item(piece, inventory)
            if match and match not in dropped:
                state_mgr.remove_item(match)
                state_mgr.place_item_in_room(match)
                dropped.append(match)
                debug.append(f"[fastpath] tool.drop: {match}")
                inventory.remove(match)
            elif not match:
                misses.append(piece)

        if not dropped and misses:
            debug.append(f"[WARN] [fastpath] drop MISS: {misses} not in inventory={state.inventory}")
            return (
                f"You aren't carrying '{', '.join(misses)}'. "
                + (f"You have: {', '.join(state.inventory)}." if state.inventory else "Your hands are empty.")
            ), debug

        bits: List[str] = []
        if len(dropped) == 1:
            bits.append(f"You set the **{dropped[0]}** down in **{state.location}**.")
        elif dropped:
            bits.append(f"You set down in **{state.location}**: **{', '.join(dropped)}**.")
        if misses:
            for m in misses:
                debug.append(f"[WARN] [fastpath] drop MISS (partial): '{m}' not in inventory")
            bits.append(f"(You weren't carrying '{', '.join(misses)}'.)")
        return " ".join(bits), debug

    # ── movement (go/enter/move/cardinal/exit-name) ──────────────────────
    exits = state.known_map.get(state.location, {}).get("exits", []) or []
    move_target: Optional[str] = None

    if cmd in ("go", "move", "enter", "head", "travel") and arg:
        move_target = _match_exit(arg, exits)
    elif cmd in ("north", "south", "east", "west", "up", "down", "n", "s", "e", "w", "u", "d"):
        move_target = _match_exit(cmd, exits)
    elif exits and text in {x.lower() for x in exits}:
        move_target = next(x for x in exits if x.lower() == text)

    if move_target:
        state_mgr.move_to(move_target)
        debug.append(f"[fastpath] tool.move_to: {move_target}")
        wb = _get_world_bible()
        room_desc = ""
        if wb:
            for loc in wb.get("locations", []) or []:
                if isinstance(loc, dict) and loc.get("name", "").lower() == move_target.lower():
                    room_desc = loc.get("description", "")
                    break
        room_items = state.room_items.get(move_target, [])
        new_exits = state.known_map.get(move_target, {}).get("exits", [])
        bits = [f"You enter **{move_target}**."]
        if room_desc:
            bits.append(room_desc)
        if room_items:
            bits.append(f"You see: {', '.join(room_items)}.")
        if new_exits:
            bits.append(f"Exits: {', '.join(new_exits)}.")
        return " ".join(bits), debug

    return None


def handle_look_command(state_mgr: StateManager) -> Optional[str]:
    """Handle 'look' / 'look around' without calling the LLM.
    Returns a description string, or None if no info is available."""
    state = state_mgr.state
    parts = []

    # Room description from world bible
    wb = _get_world_bible()
    if wb:
        for loc in wb.get("locations", []):
            if isinstance(loc, dict) and loc.get("name", "").lower() == state.location.lower():
                desc = loc.get("description", "")
                if desc:
                    parts.append(f"**{state.location}**: {desc}")
                break

    if not parts:
        parts.append(f"You are in **{state.location}**.")

    # Room items
    room_items = state.room_items.get(state.location, [])
    if room_items:
        parts.append(f"You see: {', '.join(room_items)}")

    # Exits
    exits = state.known_map.get(state.location, {}).get("exits", [])
    if exits:
        parts.append(f"Exits: {', '.join(exits)}")

    # Health
    parts.append(f"Health: {state.health}")

    # Inventory
    if state.inventory:
        parts.append(f"Carrying: {', '.join(state.inventory)}")
    else:
        parts.append("Carrying: nothing")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI MODE
# ═══════════════════════════════════════════════════════════════════════════════

def interactive_loop(state_mgr: StateManager, llm: LLMEngine, image_gen: Optional[MfluxImageGenerator]) -> None:
    # Opening scene
    opening_text, opening_images = start_story(state_mgr, llm, image_gen)
    print(opening_text)
    if opening_images:
        print("[images] " + ", ".join(opening_images))
    print("\nType 'help' for commands. Type 'quit' to exit.")
    while True:
        try:
            user = input("\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n" + format_session_postmortem(state_mgr))
            print("\nGoodbye!")
            break

        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            print("\n" + format_session_postmortem(state_mgr))
            print("Goodbye!")
            break
        if user.lower() == "save":
            path = state_mgr.save()
            print(f"Saved: {path}")
            continue
        if user.lower() == "health":
            print(f"Health: {state_mgr.state.health}")
            continue
        if user.lower() == "images":
            if not state_mgr.state.last_images:
                print("No images generated yet.")
            else:
                for img in state_mgr.state.last_images[-10:]:
                    print(f"- {img['prompt']} -> {img['path']}")
            continue

        # CHANGE (chunk I): unified deterministic fast-path — handles look/take/drop/move/
        # examine/inventory/map/help/wait without an LLM call.
        local = try_local_command(state_mgr, user)
        if local is not None:
            local_text, _local_debug = local
            # CHANGE (play robustness fix): tick timers, count the turn, advance chain
            # progress, and check win condition so the CLI doesn't silently miss victories
            # / deaths from fast-path actions.
            timer_events_local = state_mgr.process_timers_and_chains()
            if timer_events_local:
                local_text += "\n\n" + " ".join(timer_events_local)

            _post_turn_bookkeeping(state_mgr)

            win_msg_local = check_win_condition(state_mgr)
            if win_msg_local:
                local_text += f"\n\n{win_msg_local}"
                try:
                    local_text += "\n\n" + format_session_postmortem(state_mgr)
                except Exception:
                    pass

            if state_mgr.state.health <= 0:
                local_text += "\n\n**You have fallen.**"
                try:
                    local_text += "\n\n" + format_session_postmortem(state_mgr)
                except Exception:
                    pass

            print("\n" + local_text)
            continue

        system_prompt = get_system_instructions()
        user_prompt = build_user_prompt(state_mgr, user)
        llm_text = llm.generate(system_prompt, user_prompt)
        final_text, new_images, payloads, tool_logs = apply_llm_directives(state_mgr, llm_text, image_gen)

        # CHANGE (silent-LLM bugfix, May 2026): mirror the Gradio fallback so the CLI
        # never shows just a blank line when reasoning ate the budget.
        if not (final_text or "").strip():
            raw_lower = (llm_text or "").lower()
            if "<think>" in raw_lower and "</think>" not in raw_lower:
                final_text = (
                    "[The narrator was still thinking when its turn ended. Try a simpler "
                    "action like `look`, `inventory`, `go <room>`, `take <item>`, or "
                    "rephrase the question.]"
                )
            elif payloads:
                final_text = "[The world shifts quietly. Try `look` to see what changed.]"
            else:
                final_text = (
                    "[The narrator returned nothing this turn. Try a simpler action: "
                    "`look`, `inventory`, `go <room>`, or `take <item>`.]"
                )

        # Only show the cleaned narration text to the user
        print("\n" + final_text.strip())
        
        # Optionally show images if generated (user-friendly)
        if new_images:
            print(f"\n[New images generated: {len(new_images)} image(s)]")
            
        # CHANGE (chunk D): even without DEBUG, surface the [SUMMARY] + [WARN] lines so the
        # user always sees the per-turn signal. Full per-tool dump still gated by DEBUG env.
        _cli_debug_full = os.environ.get("DEBUG", "").lower() in {"1", "true", "yes"}
        if _cli_debug_full and payloads:
            print("[debug:json] " + json.dumps(payloads, ensure_ascii=False, indent=2))
        for line in _filter_tool_logs_for_display(tool_logs, _cli_debug_full or ADV_VERBOSE_DEBUG):
            print("[debug:tool] " + line)


def do_generate_world_bible(wb_model: str, theme: Optional[str] = None, max_tokens: int = 4000) -> str:
    """Standalone world bible generation for command line usage."""
    success, message, _ = execute_world_bible_generation(StateManager(), wb_model, theme, max_tokens)
    if success:
        print(message)
    return message


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════════

# CHANGE (default-cave shared with browser, May 2026): the in-browser game ships
# `browser_adventure/default_cave.json` — 6 rooms, full puzzle chain, walkthrough,
# carefully designed. The Python engine had a sparser inline default that the
# user could not actually choose. Load the SAME JSON the browser uses so the
# "🕯️ Default Cave Adventure" pick card is wired to a real bible.
def _normalize_browser_chain(wb: Dict[str, Any]) -> None:
    """Convert browser-style chain steps {step, action, gives, unlocks} into the
    engine schema {step, location, requires_item, blocker, result}. Fills in
    missing `location` and `requires_item` by scanning the action text for known
    room / NPC / monster / item names. Idempotent — already-normalized steps are
    left alone."""
    chain = wb.get("solution_chain") or []
    if not chain or not isinstance(chain, list):
        return
    if isinstance(chain[0], dict) and "location" in chain[0]:
        return  # already engine schema

    loc_names = [
        str(l.get("name") or "").strip()
        for l in (wb.get("locations") or [])
        if isinstance(l, dict)
    ]
    loc_names = [n for n in loc_names if n]

    item_names = [
        str(i.get("name") or "").strip()
        for i in (wb.get("key_items") or [])
        if isinstance(i, dict)
    ]
    item_names = [n for n in item_names if n]

    # Where do entities live?
    entity_loc: Dict[str, str] = {}
    for e in (wb.get("npcs") or []) + (wb.get("monsters") or []):
        if isinstance(e, dict):
            nm = str(e.get("name") or "").strip()
            lc = str(e.get("location") or "").strip()
            if nm and lc:
                entity_loc[nm.lower()] = lc

    # Where can an item be obtained? Pull a real room name out of item_locations.
    item_loc: Dict[str, str] = {}
    for item, desc in (wb.get("item_locations") or {}).items():
        s = str(desc)
        for ln in loc_names:
            if ln and ln in s:
                item_loc[item.lower()] = ln
                break

    new_chain: List[Dict[str, Any]] = []
    for i, step in enumerate(chain):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or step.get("result") or "")
        gives = step.get("gives")
        unlocks = step.get("unlocks")
        action_l = action.lower()

        # Derive location: room name in action, NPC/monster in action, item-source room, fallback to start.
        loc: Optional[str] = None
        for ln in loc_names:
            if ln and ln.lower() in action_l:
                loc = ln
                break
        if not loc:
            for nm, ll in entity_loc.items():
                if nm and nm in action_l:
                    loc = ll
                    break
        if not loc and isinstance(gives, str) and gives.lower() in item_loc:
            loc = item_loc[gives.lower()]
        if not loc and loc_names:
            loc = loc_names[0]

        # Derive requires_item: any key_item name in the action OTHER than `gives`.
        gives_l = gives.lower() if isinstance(gives, str) else ""
        req: Optional[str] = None
        for it in item_names:
            if it and it.lower() != gives_l and it.lower() in action_l:
                req = it
                break

        new_chain.append({
            "step": step.get("step", i + 1),
            "location": loc,
            "requires_item": req,
            "blocker": str(unlocks or ""),
            "result": (str(gives) if gives else action),
        })

    wb["solution_chain"] = new_chain


def load_default_world_bible_if_needed() -> bool:
    """If WORLD_BIBLE is unset, load browser_adventure/default_cave.json and
    normalize it to the engine's schema. Returns True if a default was loaded.
    """
    global WORLD_BIBLE
    if isinstance(WORLD_BIBLE, dict) and WORLD_BIBLE:
        return False
    candidates = [
        os.path.join(_SCRIPT_DIR, "..", "browser_adventure", "default_cave.json"),
        os.path.join(os.path.dirname(_SCRIPT_DIR), "browser_adventure", "default_cave.json"),
    ]
    for raw in candidates:
        path = os.path.normpath(raw)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                wb = json.load(f)
        except Exception as e:
            print(f"[default_cave] failed to load {path}: {e}")
            continue
        # Browser → engine schema:
        #   puzzle_chain → solution_chain (key rename)
        #   chain steps {step, action, gives, unlocks} → {step, location, requires_item, blocker, result}
        if "puzzle_chain" in wb and "solution_chain" not in wb:
            wb["solution_chain"] = wb.pop("puzzle_chain")
        _normalize_browser_chain(wb)
        # win_condition string → struct (chunk G).
        try:
            _parse_win_condition(wb)
        except Exception:
            pass
        # Run mechanical auto-repair so the loaded bible passes our validators.
        try:
            repairs = auto_repair_world_bible(wb)
            if repairs:
                print(f"[default_cave] auto-repair applied {len(repairs)} fix(es):")
                for r in repairs[:6]:
                    print(f"  - {r}")
        except Exception:
            pass
        WORLD_BIBLE = wb
        rooms = len(wb.get("locations", []) or [])
        items = len(wb.get("key_items", []) or [])
        chain_steps = len(wb.get("solution_chain", []) or [])
        print(
            f"[default_cave] Loaded {path} ({rooms} rooms, {items} key items, {chain_steps} chain steps)"
        )
        return True
    print(
        "[default_cave] No browser_adventure/default_cave.json found; engine will "
        "use its inline default_plan only when generation is requested without a model."
    )
    return False


def launch_gradio_ui(state_mgr: StateManager, llm: LLMEngine, image_gen: Optional[MfluxImageGenerator]) -> None:
    """Simple Gradio UI that shows images, inventory, health, map, and a debug console."""

    # CHANGE (default-cave shared with browser): if no world bible is set yet
    # (fresh launch, no save loaded), pull in browser_adventure/default_cave.json
    # so the "Default Cave Adventure" pick card is immediately playable. Saved
    # games and freshly-generated bibles still take precedence.
    load_default_world_bible_if_needed()

    # Simple closure-based UI state (avoid Gradio State issues)
    ui_data: Dict[str, Any] = {
        "narration": "",
        "images": [],      # list of image file paths
        "debug": [],       # list of debug strings
    }

    def _append_debug(s: str):
        print(s)
        return s

    # POPULATE GALLERY with any existing images (from pregeneration or previous sessions)
    populate_gallery_from_state(state_mgr, ui_data)

    # Helper: autosave current game if there is meaningful progress
    def _autosave_if_active() -> Optional[str]:
        try:
            # Consider game "active" if any narration, non-empty inventory, or explored map beyond Start
            active = bool(
                (ui_data.get("narration") or "").strip()
                or state_mgr.state.inventory
                or (len(state_mgr.state.known_map or {}) > 1)
                or state_mgr.state.images_generated > 0
            )
            if not active:
                return None
            # Persist UI extras before saving
            state_mgr._current_narration = ui_data.get("narration", "")
            state_mgr._current_debug = ui_data.get("debug", [])
            ts = datetime.now().strftime("%m%d_%H%M")
            label = f"AutoSave_{ts}"
            path = state_mgr.save(label=label)
            ui_data["debug"].append(f"[autosave] Saved current game → {os.path.basename(path)}")
            return path
        except Exception as e:
            ui_data["debug"].append(f"[autosave] Skipped (error: {e})")
            return None
    
    # Generate opening scene only if an LLM is selected; otherwise show instructions.
    # CHANGE (UI v2): the welcome message now points the user back to the Setup tab
    # since there's no LLM dropdown on the Play tab anymore.
    if getattr(llm, "model_id", "").strip():
        opening_text, opening_images = start_story(state_mgr, llm, image_gen)
        ui_data["narration"] = (
            opening_text
            + "\n\nType an action below and press **Send**. "
              "Try `look around`, `inventory`, `go cave mouth`, `take torch`, or talk to the hermit.\n"
        )
        ui_data["images"].extend(opening_images)
    else:
        ui_data["narration"] = (
            "Welcome to JMR's LLM Adventure!\n\n"
            "To begin:\n"
            "  1. Click the **🎲 Setup** tab above.\n"
            "  2. In Step 1, pick your LLM and click **Load / Reload LLM**.\n"
            "  3. In Step 2, choose 🕯️ Default Cave (or generate a new world / load a save).\n"
            "  4. In Step 3, click **▶ Start Adventure** — that switches you back here.\n\n"
            "Once an LLM is loaded the opening scene will appear automatically."
        )

    def do_send(user_text: str):
        if not user_text.strip():
            return (
                ui_data["narration"],  # Return plain text narration
                ui_data["images"],
                get_debug_view(),  # Return debug data for UI
                str(state_mgr.state.health),
                ", ".join(state_mgr.state.inventory) or "(empty)",
                state_mgr.describe_map(),
                None,
                get_llm_context_view(),
                state_mgr.state.location,
            )

        # CHANGE (chunk I): try the deterministic engine fast-path FIRST. Movement,
        # take/drop, look, examine, inventory, map, help, wait — all handled here without
        # touching the LLM. Saves tokens, never desyncs state, never produces garbage.
        lower_text = user_text.strip().lower()
        local = try_local_command(state_mgr, user_text)
        if local is not None:
            local_text, local_debug = local

            # CHANGE (play robustness fix): a fast-path action is a real turn — tick
            # active timers/chains, run post-turn bookkeeping (turn counter + chain
            # progress), and check the win condition. Otherwise picking up the final
            # treasure or walking to the win room would silently fail to end the game.
            timer_events_local = state_mgr.process_timers_and_chains()
            if timer_events_local:
                local_text += "\n\n" + " ".join(timer_events_local)
                ui_data["debug"].extend([f"[timer] {e}" for e in timer_events_local])

            _post_turn_bookkeeping(state_mgr)

            win_msg_local = check_win_condition(state_mgr)
            if win_msg_local:
                local_text += f"\n\n{win_msg_local}"
                try:
                    local_text += "\n\n" + format_session_postmortem(state_mgr)
                except Exception as _pm_e:
                    ui_data["debug"].append(f"[postmortem] failed ({_pm_e})")
                ui_data["debug"].append("[game] Win condition met (via fast-path)!")

            if state_mgr.state.health <= 0:
                local_text += "\n\n**You have fallen. Your adventure ends here.**\nType 'restart' to begin a new adventure, or 'load' to restore a saved game."
                try:
                    local_text += "\n\n" + format_session_postmortem(state_mgr)
                except Exception as _pm_e:
                    ui_data["debug"].append(f"[postmortem] failed ({_pm_e})")
                ui_data["debug"].append("[game] Player died (via fast-path)")

            ui_data["narration"] += f"\n> {user_text}\n\n{local_text}\n"
            for ln in local_debug:
                ui_data["debug"].append(ln)
            latest_image = get_current_room_image(state_mgr)
            return (
                ui_data["narration"],
                ui_data["images"],
                get_debug_view(),
                str(state_mgr.state.health),
                ", ".join(state_mgr.state.inventory) or "(empty)",
                state_mgr.describe_map(),
                latest_image,
                get_llm_context_view(),
                state_mgr.state.location,
            )

        # Block actions if player is dead
        if state_mgr.state.health <= 0 and lower_text not in ("restart", "load", "help"):
            ui_data["narration"] += f"\n> {user_text}\n\n**You have fallen.** Type 'restart' to begin anew or 'load' to restore a save.\n"
            return (
                ui_data["narration"],
                ui_data["images"],
                get_debug_view(),
                str(state_mgr.state.health),
                ", ".join(state_mgr.state.inventory) or "(empty)",
                state_mgr.describe_map(),
                None,
                get_llm_context_view(),
                state_mgr.state.location,
            )

        # Build prompts and call LLM
        if not getattr(llm, "model_id", "").strip():
            # LLM not selected yet; don't call the backend
            ui_data["debug"].append("[llm] Skipped: no model selected")
            return (
                ui_data["narration"] + "\nPlease load an LLM first (use the dropdown and Load/Reload LLM).",
                ui_data["images"],
                get_debug_view(),  # Return debug data for UI
                str(state_mgr.state.health),
                ", ".join(state_mgr.state.inventory) or "(empty)",
                state_mgr.describe_map(),
                None,
                get_llm_context_view(),
                state_mgr.state.location,
            )
        # Process any active timers and chain reactions before LLM turn
        timer_events = state_mgr.process_timers_and_chains()
        if timer_events:
            # Add timer events to narration for continuity
            timer_narration = " ".join(timer_events)
            ui_data["narration"] += f"\n\n{timer_narration}"
            ui_data["debug"].extend([f"[timer] {event}" for event in timer_events])

        system_prompt = get_system_instructions()
        user_prompt = build_user_prompt(state_mgr, user_text)
        llm_text = llm.generate(system_prompt, user_prompt)
        
        # Add raw LLM response to debug only
        ui_data["debug"].append(f"[llm_raw] {llm_text}")

        # Track how many images existed before this turn to identify new ones
        _prev_image_count = len(state_mgr.state.last_images)

        final_text, new_images, payloads, tool_logs = apply_llm_directives(state_mgr, llm_text, image_gen)

        # CHANGE (silent-LLM bugfix, May 2026): when the model emits only an unclosed
        # <think> reasoning block (it ran out of token budget before getting to prose),
        # strip_llm_thinking_blocks legitimately reduces the response to "". The UI
        # would then show the player's command echoed but no narration, looking like
        # a hung engine. Surface a clear, actionable fallback so the player always
        # sees something and knows what went wrong.
        if not (final_text or "").strip():
            raw = (llm_text or "")
            raw_lower = raw.lower()
            raw_len = len(raw)
            if "<think>" in raw_lower and "</think>" not in raw_lower:
                final_text = (
                    "[The narrator is still thinking when its turn ended. The model used "
                    "its full reasoning budget without finishing a reply. Try a simpler "
                    "action: `look`, `inventory`, `go <room>`, `take <item>`, or rephrase "
                    "the question more concretely.]"
                )
                ui_data["debug"].append(
                    f"[silent] unclosed <think> tag — model exceeded reasoning budget ({raw_len} raw chars)"
                )
            elif payloads:
                final_text = "[The world shifts quietly. Try `look` to see what changed.]"
                ui_data["debug"].append(
                    f"[silent] narration empty but {len(payloads)} payload(s) applied"
                )
            else:
                final_text = (
                    "[The narrator returned nothing this turn. Try a simpler action: "
                    "`look`, `inventory`, `go <room>`, or `take <item>`.]"
                )
                ui_data["debug"].append(
                    f"[silent] LLM produced no usable narration or JSON ({raw_len} raw chars)"
                )

        # CHANGE (chunk J): capture warnings emitted this turn so build_user_prompt can
        # surface them to the LLM next turn. Also capture for the chunk K post-mortem tally.
        _turn_warns = [
            ln for ln in tool_logs
            if ln.startswith("[WARN]") or ln.startswith("[WARNING]")
        ]
        if _turn_warns:
            state_mgr.state.engine_notes_for_next_turn = _turn_warns[:6]
            try:
                state_mgr.state.warnings_history.extend(_turn_warns)
            except Exception:
                pass
        # CHANGE (chunk K + play robustness fix): track turns + chain steps via shared helper
        # so fast-path turns count too.
        _post_turn_bookkeeping(state_mgr)

        # BACKUP: If no images were generated and current room has no image, generate one
        # This ensures new games/rooms always get an image even if LLM doesn't request one
        # Uses the narrative context so the image matches what was just described
        if not new_images and state_mgr.state.location and image_gen:
            if not state_mgr.state.has_room_image(state_mgr.state.location):
                # Pass the narrative text so the image matches the description
                backup_image = generate_room_image_if_needed(
                    state_mgr, image_gen, state_mgr.state.location,
                    narrative_context=final_text
                )
                if backup_image:
                    new_images.append(backup_image)
                    ui_data["debug"].append(f"[backup] Generated missing room image for: {state_mgr.state.location}")

        # Death check
        if state_mgr.state.health <= 0:
            final_text += "\n\n**You have fallen. Your adventure ends here.**\nType 'restart' to begin a new adventure, or 'load' to restore a saved game."
            # CHANGE (chunk K): emit post-mortem on death
            try:
                final_text += "\n\n" + format_session_postmortem(state_mgr)
            except Exception as _pm_e:
                ui_data["debug"].append(f"[postmortem] failed ({_pm_e})")
            ui_data["debug"].append("[game] Player died (health <= 0)")

        # Win condition check
        win_msg = check_win_condition(state_mgr)
        if win_msg:
            final_text += f"\n\n{win_msg}"
            # CHANGE (chunk K): emit post-mortem on victory
            try:
                final_text += "\n\n" + format_session_postmortem(state_mgr)
            except Exception as _pm_e:
                ui_data["debug"].append(f"[postmortem] failed ({_pm_e})")
            ui_data["debug"].append("[game] Win condition met!")

        # Save conversation history (keep last 5 turns for context)
        # Include tool results in conversation history
        tool_feedback = []
        if tool_logs:
            # Extract key feedback for LLM awareness
            for log in tool_logs:
                if "not present" in log or "failed" in log.lower() or "skip" in log.lower():
                    tool_feedback.append(log)
                elif any(x in log for x in ["tool.move_to", "tool.add_item", "tool.place_item", "tool.room_take"]):
                    tool_feedback.append(log)
        
        state_mgr.state.recent_history.append({
            "player": user_text,
            "response": final_text[:300],  # Keep first 300 chars to avoid bloat
            "actions_taken": tool_feedback if tool_feedback else None  # All tool feedback for full context
        })
        if len(state_mgr.state.recent_history) > 5:
            state_mgr.state.recent_history.pop(0)  # Remove oldest
        # Log conversation memory status
        try:
            ui_data["debug"].append(
                f"[memory] recent={len(state_mgr.state.recent_history)} story_context={'yes' if (state_mgr.state.story_context or '').strip() else 'no'}"
            )
        except Exception:
            pass
        
        # CHANGE (chunk D): default to minimum-but-rich debug — show only the [SUMMARY] line
        # plus any [WARN]/[WARNING]. Set ADV_VERBOSE_DEBUG=1 to see every per-tool entry.
        if not payloads:
            ui_data["debug"].append("[WARN] [json] none — LLM produced no parseable directives")
        for line in _filter_tool_logs_for_display(tool_logs, ADV_VERBOSE_DEBUG):
            ui_data["debug"].append(f"[tool] {line}")
        if new_images:
            # Show image filenames for gallery
            try:
                _names = [os.path.basename(p) for p in new_images if isinstance(p, str)]
            except Exception:
                _names = new_images
            ui_data["debug"].append(f"[images] {', '.join(_names)}")

        # Update UI state - only show cleaned narration text
        ui_data["narration"] += f"\n> {user_text}\n\n{final_text}\n"
        # TRIVIAL: Deduplicate gallery paths while preserving order
        if new_images:
            seen_paths = set(ui_data.get("images", []))
            for p in new_images:
                if isinstance(p, str) and p not in seen_paths:
                    ui_data["images"].append(p)
                    seen_paths.add(p)

                # CHANGE: Simple priority - show what was generated/requested this turn
        # The LLM controls this by what it requests in JSON images array
        latest_image = None
        try:
            _turn_added = state_mgr.state.last_images[_prev_image_count:]
            # If images were added this turn, show the last one (LLM's focus)
            if _turn_added:
                # The last image added is likely the most relevant (LLM requested it last)
                for _entry in reversed(_turn_added):
                    _p = _entry.get("path")
                    if _p and os.path.exists(_p):
                        latest_image = _p
                        _type = _entry.get("type", "image")
                        _subj = _entry.get("subject", "")
                        ui_data["debug"].append(f"[display] Turn-generated {_type}: {_subj} → {os.path.basename(_p)}")
                        break
        except Exception:
            pass

        # CHANGE (DISPLAY ONLY): If LLM requested an item image and we have it cached,
        # prefer showing that cached item image this turn (no new generation/state changes).
        if not latest_image and payloads:
            try:
                requested_texts: List[str] = []
                for _d in payloads:
                    _imgs = _d.get("images") if isinstance(_d, dict) else None
                    if isinstance(_imgs, list):
                        for _p in _imgs:
                            if isinstance(_p, str) and _p.strip():
                                requested_texts.append(_p.strip().lower())
                
                ui_data["debug"].append(f"[display_debug] requested_texts: {requested_texts}")
                
                if requested_texts:
                    # Prefer exact subject name matches (rooms/items) using helper
                    for req in requested_texts:
                        candidate = get_image_for_subject(state_mgr, req)
                        if candidate:
                            latest_image = candidate
                            ui_data["debug"].append(f"[display] Showing requested subject: {req} → {os.path.basename(candidate)}")
                            break
                    # If no direct subject match, try partial subject/prompt/filename matches
                    if not latest_image:
                        for _img in state_mgr.state.last_images:
                            if _img.get("type") in ("item", "room"):
                                _subject = str(_img.get("subject", ""))
                                _prompt = str(_img.get("prompt", ""))
                                _path = _img.get("path")
                                if _path and os.path.exists(_path):
                                    _filename = os.path.basename(_path).lower()
                                    _sub_l = _subject.lower()
                                    _pr_l = _prompt.lower()
                                    for req in requested_texts:
                                        if (_sub_l in req) or (req in _sub_l) or (req == _filename) or (req in _pr_l):
                                            latest_image = _path
                                            ui_data["debug"].append(f"[display] Showing requested (fuzzy): {_subject} → {os.path.basename(_path)}")
                                            break
                                    if latest_image:
                                        break
                    if not latest_image:
                        ui_data["debug"].append(f"[display_debug] No matching cached item found for requests: {requested_texts}")
            except Exception as e:
                ui_data["debug"].append(f"[display_debug] exception: {e}")
        


        if not latest_image:
            # Align with narration/location
            latest_image = get_current_room_image(state_mgr)
            if latest_image:
                ui_data["debug"].append(f"[display] Showing room image: {state_mgr.state.location} → {os.path.basename(latest_image)}")
            else:
                latest_image = get_inventory_item_image(state_mgr)
                if latest_image:
                    # CHANGE (DEBUG ONLY): include which inventory item matched
                    try:
                        _item_matched = None
                        for _img in state_mgr.state.last_images:
                            if _img.get("type") == "item" and _img.get("path") == latest_image:
                                _item_matched = _img.get("subject")
                                break
                        if _item_matched:
                            ui_data["debug"].append(f"[display] Showing inventory item image: {_item_matched} → {os.path.basename(latest_image)}")
                        else:
                            ui_data["debug"].append(f"[display] Showing inventory item image: {os.path.basename(latest_image)}")
                    except Exception:
                        ui_data["debug"].append(f"[display] Showing inventory item image: {os.path.basename(latest_image)}")
                else:
                    if ui_data["images"]:
                        latest_image = ui_data["images"][-1]
                        ui_data["debug"].append(f"[display] Fallback to latest gallery image: {os.path.basename(latest_image)}")
                    else:
                        ui_data["debug"].append(f"[display] No image found for room '{state_mgr.state.location}' or inventory items")

        # CHANGE: Latest-turn gallery removed per request; single Latest Image panel remains
        return (
            ui_data["narration"],  # Return plain text narration
            ui_data["images"],
            get_debug_view(),
            str(state_mgr.state.health),
            ", ".join(state_mgr.state.inventory) or "(empty)",
            state_mgr.describe_map(),
            latest_image,
            get_llm_context_view(),  # Update dashboard
            state_mgr.state.location,
        )

    def do_reload_llm(model_name: str, max_tokens: int):
        # Reload just the LLM
        nonlocal llm
        # CHANGE: skip reload if model unchanged to avoid no-op reinitialization (compare full path / choice id).
        if (
            model_name
            and getattr(llm, "model_path", None) == model_name
            and getattr(llm, "max_new_tokens", None) == max_tokens
        ):
            _append_debug("[reload] LLM unchanged; skipping")
            return f"✅ LLM unchanged: {model_name}"
        if model_name:
            # Unload previous model to free memory
            if llm and (getattr(llm, "model", None) is not None or getattr(llm, "_uses_ollama", False)):
                llm._unload()
            # CHANGE: ollama:<tag> uses Ollama; otherwise MLX folder under MLX_MODELS_DIR.
            if llm_choice_is_ollama(model_name):
                model_path = model_name
            else:
                model_path = os.path.join(MLX_MODELS_DIR, model_name)
            llm = LLMEngine(model_path=model_path, max_new_tokens=max_tokens, temperature=0.7)
            _append_debug(f"[reload] LLM -> {model_name} ({max_tokens} tokens)")
            return f"✅ LLM loaded: {model_name} ({max_tokens} tokens)"
        return "❌ No LLM model selected"

    def do_reload_diffuser(diffuser_id: str):
        # Reload just the MFLUX image generator with proper cleanup
        nonlocal image_gen

        # TRIVIAL GUARD: skip reload if model unchanged to avoid unnecessary pipeline reinit
        if diffuser_id and image_gen and getattr(image_gen, "model_id", None) == diffuser_id:
            _append_debug("[reload] MFLUX model unchanged; skipping")
            return f"✅ MFLUX model unchanged: {diffuser_id}"

        # Clean up the old model first
        if image_gen:
            image_gen.cleanup()

        if diffuser_id:
            # Map directory name to MFLUX model name
            # "FLUX.1-dev" -> "dev", "FLUX2-klein-9B-mlx-8bit" -> "flux2-klein-9b"
            mflux_name = diffuser_id
            if diffuser_id.startswith("FLUX.1-"):
                mflux_name = diffuser_id.replace("FLUX.1-", "").lower()
            elif "klein-9B" in diffuser_id or "klein-9b" in diffuser_id:
                mflux_name = "flux2-klein-9b"
            elif "klein-4B" in diffuser_id or "klein-4b" in diffuser_id:
                mflux_name = "flux2-klein-4b"
            local_path = os.path.join(diffusion_models_root(), diffuser_id)
            if not os.path.isdir(local_path):
                local_path = None
            image_gen = MfluxImageGenerator(model_name=mflux_name, local_path=local_path)
            _append_debug(f"[reload] MFLUX -> {diffuser_id}")
            return f"✅ MFLUX model loaded: {diffuser_id}"
        else:
            image_gen = None
            _append_debug("[reload] MFLUX -> disabled")
            return "✅ Image generation disabled"
    
    def do_generate_world_bible(selected_model: str, current_theme: str = "", max_tokens: int = 4000):
        """Generate world bible using the model string passed from the dropdown."""
        model_used = (selected_model or "").strip()
        if not model_used:
            # Leave button appearance unchanged on failure
            # Returns: wb_status, world_bible_view, gen_bible_btn, narration_box, gallery, latest_image
            return (
                "❌ Select an LLM first from the dropdown, then click Generate World Bible.",
                get_world_bible_view(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )

        # Autosave current run before generating a new world (non-destructive)
        _autosave_if_active()

        # Get the current theme from the input field
        theme_to_use = current_theme.strip() if current_theme and current_theme.strip() else None

        # World bible generation needs MUCH more tokens than normal conversation
        # Use a high fixed limit (6000) instead of the slider value
        world_bible_max_tokens = 6000
        print(f"[world_bible] Using fixed token limit: {world_bible_max_tokens} (ignoring slider value {max_tokens})")

        success, message, world_bible_data = execute_world_bible_generation(state_mgr, model_used, theme_to_use, world_bible_max_tokens)

        if success and world_bible_data:
            # Add to debug console for troubleshooting
            ui_data["debug"].append(f"[world_bible] Generated successfully:")
            ui_data["debug"].append(f"[world_bible] {json.dumps(world_bible_data, ensure_ascii=False, indent=2)}")
            # Clear current game state and UI after generating a new world bible
            try:
                state_mgr.state = GameState(player_name=state_mgr.state.player_name)
                ui_data["narration"] = ""
                ui_data["images"] = []
                ui_data["debug"].append("[world_bible] Cleared current game state for new world")
            except Exception:
                pass

            # AUTO-START: Generate opening narration and image so player doesn't have to type "look around"
            opening_text = ""
            opening_image = None
            try:
                if getattr(llm, "model_id", "").strip():
                    opening_text, opening_images = start_story(state_mgr, llm, image_gen)
                    ui_data["narration"] = opening_text + "\n\n"
                    ui_data["images"].extend(opening_images)
                    if opening_images:
                        opening_image = opening_images[-1]
                    ui_data["debug"].append("[world_bible] Auto-started opening scene")
            except Exception as e:
                ui_data["debug"].append(f"[world_bible] Auto-start failed: {e}")

            # After success, de-emphasize the button variant
            # Return formatted world bible view data for the JSON component
            formatted_world_bible = {
                "theme": world_bible_data.get("global_theme", world_bible_data.get("theme", "Not set")),
                "objectives": world_bible_data.get("objectives", []),
                "npcs": world_bible_data.get("npcs", world_bible_data.get("key_characters", [])),
                "monsters": world_bible_data.get("monsters", []),
                "key_items": world_bible_data.get("key_items", []),
                "locations": world_bible_data.get("locations", []),
                "riddles": world_bible_data.get("riddles", world_bible_data.get("riddles_and_tasks", [])),
                "mechanics": world_bible_data.get("mechanics", []),
                "progression_hints": world_bible_data.get("progression_hints", []),
                "note": "This is STATIC - generated once. The turn LLM uses this for consistency but doesn't modify it."
            }
            # Returns: wb_status, world_bible_view, gen_bible_btn, narration_box, gallery, latest_image
            return (
                f"✅ {message}",
                formatted_world_bible,
                gr.update(variant="secondary"),
                ui_data["narration"],  # Update narration box
                ui_data["images"],  # Update gallery
                opening_image  # Update latest image
            )
        else:
            # Keep standout variant to indicate action still needed / failed
            # Returns: wb_status, world_bible_view, gen_bible_btn, narration_box, gallery, latest_image
            return (
                f"❌ {message} - Try generating again or check your theme prompt.",
                get_world_bible_view(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )
 
    # CHANGE (UI redesign v2, May 2026): the v1 redesign had three usability problems
    # the user called out from screenshots:
    #   1. White side panels — body/html background wasn't being themed.
    #   2. Pick-card descriptions rendered in --text-muted, basically unreadable.
    #   3. Three pick cards always expanded → cluttered; START button below the fold.
    # v2 fixes by (a) painting html+body dark, (b) using --text not --text-muted for
    # body copy, (c) replacing pick cards with a single gr.Radio + conditional panels,
    # and (d) reorganizing into numbered steps with a HUGE start button right after.
    custom_css = """
    :root {
      --bg: #0a0a12; --panel: #14141f; --panel-alt: #1a1a28; --border: #252535;
      --text: #d8d0c4; --text-muted: #8a8898;
      --accent: #c9a550; --accent-dim: #8a7535;
      --info: #6eb5ff; --ok: #7dffb2; --warn: #ffc46e; --err: #ff8a8a;
    }
    /* Paint the WHOLE viewport — fixes the white side panels in v1 */
    html, body {
      background: var(--bg) !important; color: var(--text) !important; min-height: 100vh !important;
      font-family: system-ui, "Segoe UI", Roboto, sans-serif !important;
    }
    .gradio-container, .app, .main, gradio-app {
      background: var(--bg) !important; color: var(--text) !important;
      max-width: 1200px !important; margin: 0 auto !important;
    }
    .gr-box, .gr-form, .gr-panel, .gr-block, .block, .form, .panel, .gr-group, .gr-accordion {
      background: var(--panel) !important;
      border-color: var(--border) !important;
      color: var(--text) !important;
    }
    h1, h2, h3, h4 { color: var(--accent) !important; }
    /* Body copy is full --text, NOT --text-muted (the v1 mistake) */
    .prose, .gr-markdown, .gr-markdown p, .gr-markdown li, .markdown {
      color: var(--text) !important; line-height: 1.55 !important;
    }
    .gr-markdown strong { color: var(--accent) !important; }
    .gr-markdown code { background: var(--panel-alt) !important; color: var(--accent) !important; padding: 1px 5px; border-radius: 3px; }
    /* Form labels stay muted — they're metadata, not content */
    label, .label-wrap label, .gr-form label, span.svelte-1gfkn6j, .label, .gr-input-label {
      color: var(--text-muted) !important;
    }
    input, textarea, select,
    .gr-input, .gr-textarea, .gr-text-input, .gr-dropdown, .gr-textbox textarea,
    .gr-textbox input, .gr-number input, .gr-dropdown ul, .gr-dropdown li {
      background: var(--panel) !important;
      color: var(--text) !important;
      border-color: var(--border) !important;
    }
    input:focus, textarea:focus, select:focus { border-color: var(--accent-dim) !important; outline: none !important; }
    button, .gr-button {
      background: var(--border) !important; color: var(--text) !important;
      border: 1px solid var(--border) !important; transition: border-color .15s, color .15s, background .15s !important;
    }
    button:hover:not(:disabled), .gr-button:hover:not(:disabled) {
      border-color: var(--accent) !important; color: var(--accent) !important;
    }
    button.primary, .gr-button-primary, button[class*="primary"] {
      background: #2a3520 !important; border-color: var(--accent-dim) !important; color: var(--accent) !important; font-weight: 600 !important;
    }
    button.primary:hover:not(:disabled), .gr-button-primary:hover:not(:disabled) {
      background: #3a4a2a !important; border-color: var(--accent) !important;
    }
    /* Tabs — clear active state */
    .tab-nav, .tabs, .tab-buttons {
      background: var(--panel) !important; border-color: var(--border) !important;
      border-bottom: 1px solid var(--border) !important;
    }
    .tab-nav button, .tabitem button { background: var(--panel) !important; color: var(--text-muted) !important; border-bottom-color: transparent !important; }
    .tab-nav button.selected, .tabitem button.selected {
      background: var(--bg) !important; color: var(--accent) !important; border-bottom-color: var(--accent) !important;
    }
    /* ── Setup wizard ── */
    .setup-step {
      background: var(--panel) !important;
      border: 1px solid var(--border) !important;
      border-radius: 8px !important;
      padding: 16px 18px !important;
      margin-bottom: 14px !important;
    }
    .setup-step h3, .setup-step h4 { margin-top: 0 !important; }
    /* Adventure-choice radio: stack vertically with bigger hit targets */
    .adventure-radio { padding: 4px 0 !important; }
    .adventure-radio fieldset { gap: 8px !important; }
    .adventure-radio label {
      background: var(--panel-alt) !important;
      border: 1px solid var(--border) !important;
      border-radius: 6px !important;
      padding: 10px 14px !important;
      color: var(--text) !important;
      font-size: 14px !important;
      cursor: pointer !important;
      transition: border-color .15s !important;
    }
    .adventure-radio label:hover { border-color: var(--accent-dim) !important; }
    .adventure-radio input:checked + span,
    .adventure-radio label:has(input:checked) { border-color: var(--accent) !important; color: var(--accent) !important; }
    .start-button button {
      font-size: 18px !important; padding: 18px 30px !important;
      background: #2a3520 !important; border: 2px solid var(--accent) !important; color: var(--accent) !important;
      font-weight: 700 !important;
      letter-spacing: 0.04em !important;
    }
    .start-button button:hover { background: #3a4a2a !important; }
    /* ── Narration: serif parchment feel ── */
    #narration-box textarea {
      font-family: Georgia, "Times New Roman", serif !important;
      font-size: 16px !important; line-height: 1.65 !important;
      background: var(--panel) !important; color: var(--text) !important;
      border-color: var(--border) !important;
    }
    /* Action input: monospace */
    #player-input textarea, #player-input input {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
      font-size: 14px !important;
    }
    .status-row .gr-textbox { border: 1px solid var(--border) !important; border-radius: 6px !important; }
    /* Header */
    .app-header { padding: 12px 4px 14px; border-bottom: 1px solid var(--border); margin-bottom: 14px; }
    .app-header h1 { font-size: 1.2rem !important; color: var(--accent) !important; margin: 0 !important; }
    .app-header .subtitle { color: var(--text-muted) !important; font-size: 0.85rem !important; }
    .gr-accordion .label-wrap { color: var(--accent) !important; }
    /* Image scene */
    .scene-image img { object-fit: contain !important; aspect-ratio: 1 / 1 !important; max-width: 100% !important; height: auto !important; }
    footer { color: var(--text-muted) !important; }
    """

    with gr.Blocks(title="JMR's LLM Adventure (MLX + MFLUX)", css=custom_css, theme=gr.themes.Base()) as app:
        # ── Header (always visible) ─────────────────────────────────────────
        gr.HTML(
            "<div class='app-header'>"
            "<h1>JMR's LLM Adventure</h1>"
            "<div class='subtitle'>Local LLM as game master · MLX or Ollama · MFLUX scene art</div>"
            "</div>"
        )

        # Helper for the saved-games dropdown (used by both tabs)
        def _list_saved_games_local():
            try:
                files = [f for f in os.listdir(SAVE_DIR) if f.startswith("Adv_") and (f.endswith('.pkl') or f.endswith('.tkl'))]
            except Exception:
                files = []
            return sorted(files, reverse=True)

        # Constants for the adventure-choice radio (must match keys exactly so the
        # show/hide handler can route correctly).
        _ADV_DEFAULT = "🕯️  Default Cave Adventure  —  Starfire Gem quest, ready to play"
        _ADV_GENERATE = "✨  Generate New Adventure  —  LLM creates a brand-new world from a theme"
        _ADV_LOAD = "💾  Load Saved Adventure  —  resume a previous game"

        with gr.Tabs() as tabs:
            # ═══════════════════ TAB 1: SETUP ═══════════════════
            with gr.TabItem("🎲 Setup", id="setup"):
                # ── Step 1: Load your LLM ──────────────────────────────────
                with gr.Group(elem_classes=["setup-step"]):
                    gr.Markdown("### Step 1 — Load your LLM")
                    gr.Markdown(
                        "Choose an MLX folder name (from `~/MLX_Models/`) or an `ollama:<tag>`. "
                        "This model narrates every turn and emits the JSON tool calls. "
                        "Click **Load / Reload LLM** to bring it up; the status line below confirms."
                    )
                    mlx_models = list_mlx_models() if check_mlx_available() else []
                    ollama_models = list_ollama_model_choices()
                    choices = ollama_models + mlx_models if (ollama_models or mlx_models) else ["(select later)"]
                    default_choice = choices[0]
                    with gr.Row():
                        llm_drop = gr.Dropdown(
                            choices, label="LLM — MLX folder or ollama:<tag>",
                            value=default_choice, scale=4,
                        )
                        reload_llm_btn = gr.Button("Load / Reload LLM", variant="primary", scale=1)
                    llm_status = gr.Markdown()

                # ── Step 2: Choose your adventure ──────────────────────────
                with gr.Group(elem_classes=["setup-step"]):
                    gr.Markdown("### Step 2 — Choose your adventure")
                    adventure_choice = gr.Radio(
                        choices=[_ADV_DEFAULT, _ADV_GENERATE, _ADV_LOAD],
                        value=_ADV_DEFAULT,
                        label="",
                        elem_classes=["adventure-radio"],
                        show_label=False,
                    )

                    # Sub-panel: default (visible by default)
                    _wb_now = _get_world_bible()
                    _default_loaded = bool(_wb_now and _wb_now.get("locations"))
                    _default_room_count = len((_wb_now or {}).get("locations", []) or [])
                    with gr.Group(visible=True) as default_panel:
                        gr.Markdown(
                            "**Starfire Gem quest** — the same canonical 6-room adventure the in-browser "
                            "edition ships in `browser_adventure/default_cave.json`. "
                            f"Currently loaded: **{'yes' if _default_loaded else 'no'}** "
                            f"({_default_room_count} rooms). "
                            "Hermit at Cave Mouth, river troll on the underground river, ancient forge, "
                            "stone dragon in the vault — bring the Starfire Gem back to Cave Mouth to win. "
                            "No generation needed; click **Start Adventure** in step 3."
                        )
                        reload_default_btn = gr.Button("Reload Default Cave", variant="secondary", size="sm")
                        default_status = gr.Markdown()

                    # Sub-panel: generate (hidden until selected)
                    with gr.Group(visible=False) as generate_panel:
                        gr.Markdown(
                            "Pick a preset theme or write your own below. The LLM will spend "
                            "~30–90s generating a 5-room world with NPCs, items, puzzles, and a "
                            "solution chain. The validator + auto-repair cleans up any gaps. "
                            "Click **Generate World Bible** when ready, then **Start Adventure** in step 3."
                        )
                        theme_dropdown = gr.Dropdown(
                            choices=list(PRESET_THEMES.keys()),
                            label="Theme preset",
                            value="Tolkien Cave Adventure",
                        )
                        theme_input = gr.Textbox(
                            label="Theme details (edit or type your own)",
                            value=PRESET_THEMES.get("Tolkien Cave Adventure", ""),
                            lines=3,
                            placeholder="Describe the adventure theme, characters, art style …",
                        )
                        with gr.Row():
                            apply_theme_btn = gr.Button("Apply Theme", variant="secondary", scale=1)
                            gen_bible_btn = gr.Button("Generate World Bible", variant="primary", scale=2)
                        theme_status = gr.Markdown()
                        wb_status = gr.Markdown()

                    # Sub-panel: load (hidden until selected)
                    with gr.Group(visible=False) as load_panel:
                        gr.Markdown(
                            "Resume a previous game (auto-saves and named saves both appear). "
                            "After loading, click **Start Adventure** in step 3."
                        )
                        saved_choices = _list_saved_games_local() or ["(none)"]
                        load_dropdown = gr.Dropdown(
                            saved_choices,
                            label="Saved games",
                            value=(saved_choices[0] if saved_choices else "(none)"),
                        )
                        load_btn = gr.Button("Load Saved Game", variant="primary")
                        load_status = gr.Markdown()

                # ── Step 3: Begin (the prominent button) ───────────────────
                with gr.Group(elem_classes=["setup-step"]):
                    gr.Markdown("### Step 3 — Begin")
                    gr.Markdown(
                        "Once your LLM is loaded (step 1) and your adventure is chosen (step 2), "
                        "press the button below. You will switch to the **Play** tab where "
                        "narration, scene images, and your action input live."
                    )
                    start_adventure_btn = gr.Button(
                        "▶  START ADVENTURE",
                        variant="primary",
                        size="lg",
                        elem_classes=["start-button"],
                    )

                # ── Optional: image model + advanced settings ──────────────
                with gr.Accordion("⚙️  Optional — image model & advanced settings", open=False):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown(
                                "**Scene art (MFLUX)** — generates a picture for each room or "
                                "key item. Optional; the game plays fine without it."
                            )
                            diffuser_choices = combined_diffuser_choices(include_skip=True)
                            current_diffuser = image_gen.model_id if image_gen else (diffuser_choices[0] if diffuser_choices else "")
                            if current_diffuser and current_diffuser not in diffuser_choices:
                                diffuser_choices = [current_diffuser] + diffuser_choices
                            diff_drop = gr.Dropdown(diffuser_choices, label="Image model (MFLUX)", value=current_diffuser)
                            reload_diff_btn = gr.Button("Load / Reload Image Model", variant="secondary", size="sm")
                            diff_status = gr.Markdown()
                        with gr.Column():
                            gr.Markdown(
                                "**Tuning** — `max tokens` caps the LLM response length per turn; "
                                "advanced directives unlock timer events and chain reactions for larger models."
                            )
                            max_tokens_slider = gr.Slider(
                                minimum=250, maximum=2500, value=2500, step=50,
                                label="LLM max tokens (response length)",
                            )
                            advanced_directives_cb = gr.Checkbox(
                                value=ADVANCED_DIRECTIVES,
                                label="Advanced directives (timers, conditionals — for larger models)",
                            )

                # Adventure-choice radio: show the matching sub-panel, hide the others.
                def _on_adventure_choice(choice: str):
                    return (
                        gr.update(visible=(choice == _ADV_DEFAULT)),
                        gr.update(visible=(choice == _ADV_GENERATE)),
                        gr.update(visible=(choice == _ADV_LOAD)),
                    )
                adventure_choice.change(
                    fn=_on_adventure_choice,
                    inputs=[adventure_choice],
                    outputs=[default_panel, generate_panel, load_panel],
                )

            # ═══════════════════ TAB 2: PLAY ═══════════════════
            with gr.TabItem("🎮 Play", id="play"):
                # ── Action bar (top of play tab) ──
                with gr.Row():
                    back_to_setup_btn = gr.Button("← Setup", size="sm", scale=1)
                    save_btn = gr.Button("💾 Save", variant="primary", size="sm", scale=1)
                    restart_btn = gr.Button("🆕 New Game (same world)", size="sm", scale=2)
                with gr.Row():
                    game_name_input = gr.Textbox(label="", placeholder="Save as: My_Adventure", scale=3, show_label=False)
                save_status = gr.Markdown()
                restart_status = gr.Markdown()

                # ── Narration (the centerpiece) ──
                narration_box = gr.Textbox(
                    value=ui_data["narration"],
                    label="Story",
                    lines=14, max_lines=24,
                    interactive=False,
                    elem_id="narration-box",
                )

                # ── Mid row: scene image + status ──
                with gr.Row():
                    with gr.Column(scale=1):
                        latest_image = gr.Image(
                            label="Scene",
                            height=320,
                            value=(get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr) or (ui_data["images"][-1] if ui_data["images"] else None)),
                            elem_classes=["scene-image"],
                        )
                    with gr.Column(scale=1):
                        with gr.Row(elem_classes=["status-row"]):
                            location = gr.Textbox(label="Location", value=str(state_mgr.state.location), interactive=False)
                            health = gr.Textbox(label="Health", value=str(state_mgr.state.health), interactive=False)
                        inventory = gr.Textbox(label="Inventory", value=", ".join(state_mgr.state.inventory) or "(empty)", interactive=False)
                        # CHANGE (visited-map UI fix, May 2026):
                        # - Map now defaults to *visited only* (the user pointed out
                        #   "Known" should mean rooms the player has actually been to).
                        # - 4 lines tall by default → action input box no longer pushed
                        #   down by a long world-bible map.
                        # - Toggle button cycles between visited / full views; user
                        #   can peek at the full world bible map without polluting the
                        #   default play view.
                        map_box = gr.Textbox(
                            label="Known map (visited only)",
                            value=state_mgr.describe_map(),
                            lines=4, max_lines=12,
                            interactive=False,
                        )
                        with gr.Row():
                            map_mode_state = gr.State("visited")  # "visited" | "full"
                            map_toggle_btn = gr.Button("🗺  Show full map", size="sm")

                # ── Action input ──
                with gr.Row():
                    user_box = gr.Textbox(
                        label="What do you do?",
                        placeholder="look around · go north · take key · talk to wizard …",
                        lines=1, scale=4,
                        elem_id="player-input",
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                # ── Optional / advanced (collapsed by default) ──
                with gr.Accordion("🖼 Image gallery & generation", open=False):
                    gallery = gr.Gallery(label="Image gallery", height=200, columns=3, value=ui_data["images"])
                    with gr.Row():
                        show_names_btn = gr.Button("Show image names", size="sm")
                        pregenerate_level = gr.Radio(["some", "most", "all"], value="some", label="Coverage")
                        generate_images_btn = gr.Button(
                            ("Regenerate" if (ui_data.get("images") or state_mgr.state.last_images) else "Generate Images"),
                            variant="secondary", size="sm",
                        )
                    gallery_filenames = gr.Textbox(label="Gallery image filenames", interactive=False, lines=3, visible=False)
                    generate_images_status = gr.Markdown()

                with gr.Accordion("🔍 Debug console", open=False):
                    def get_debug_view():
                        return {"debug_logs": ui_data.get("debug", [])}
                    debug_md = gr.JSON(label="Debug logs", value=get_debug_view())

                with gr.Accordion("📚 World bible (static design)", open=False):
                    def get_world_bible_view():
                        try:
                            wb = _get_world_bible()
                            if not wb:
                                return {"status": "No World Bible generated yet. Use Setup → Generate New Adventure."}
                            return {
                                "theme": wb.get("global_theme", wb.get("theme", "Not set")),
                                "objectives": wb.get("objectives", []),
                                "npcs": wb.get("npcs", wb.get("key_characters", [])),
                                "monsters": wb.get("monsters", []),
                                "key_items": wb.get("key_items", []),
                                "locations": wb.get("locations", []),
                                "riddles": wb.get("riddles", wb.get("riddles_and_tasks", [])),
                                "mechanics": wb.get("mechanics", []),
                                "progression_hints": wb.get("progression_hints", []),
                                "win_condition": wb.get("win_condition"),
                                "solution_chain": wb.get("solution_chain", []),
                            }
                        except Exception as e:
                            return {"error": f"World Bible view error: {e}"}
                    world_bible_view = gr.JSON(label="World Bible", value=get_world_bible_view())

                with gr.Accordion("🤖 LLM context (what the AI sees)", open=False):
                    def get_llm_context_view():
                        try:
                            sample_input = "[Next player input will go here]"
                            full_prompt = build_user_prompt(state_mgr, sample_input)
                            sections = full_prompt.split("\n---\n")
                            game_state_json = sections[0].replace("Game State (read-only; update via JSON directives only):\n", "") if sections else ""
                            try:
                                json_end = game_state_json.find("\nWorld context:")
                                if json_end > 0:
                                    json_part = game_state_json[:json_end]
                                    context_part = game_state_json[json_end:]
                                else:
                                    json_part = game_state_json
                                    context_part = ""
                                parsed_state = json.loads(json_part)
                            except Exception:
                                parsed_state = {"parse_error": "Could not parse game state"}
                                context_part = game_state_json
                            return {
                                "game_state": parsed_state,
                                "world_bible_context": context_part.strip() if context_part else "(No world bible context)",
                            }
                        except Exception as e:
                            return {"error": f"Context view error: {e}"}
                    world_bible_display = gr.JSON(label="Exact context sent to LLM", value=get_llm_context_view())

                with gr.Accordion("🎮 GameState (dynamic memory)", open=False):
                    def get_game_state_view():
                        try:
                            state = state_mgr.state
                            return {
                                "player": {
                                    "name": state.player_name,
                                    "location": state.location,
                                    "health": state.health,
                                    "inventory": state.inventory,
                                },
                                "world": {
                                    "known_map": state.known_map,
                                    "room_items": state.room_items,
                                    "notes": state.notes,
                                },
                                "conversation": {
                                    "recent_history": state.recent_history,
                                    "story_context": state.story_context,
                                },
                                "session": {
                                    "session_turns": state.session_turns,
                                    "chain_steps_completed": state.chain_steps_completed,
                                    "warnings_history_count": len(state.warnings_history),
                                },
                                "game_flags": state.game_flags,
                                "images": {
                                    "rooms_with_images": state.rooms_with_images,
                                    "items_with_images": state.items_with_images,
                                    "total_generated": state.images_generated,
                                    "total_reused": state.images_reused,
                                    "last_images_count": len(state.last_images),
                                },
                            }
                        except Exception as e:
                            return {"error": f"GameState view error: {e}"}
                    game_state_view = gr.JSON(label="Game state", value=get_game_state_view())

        # ── Tab switching + Start handler (UI v2) ────────────────────────────
        def go_to_setup():
            return gr.Tabs(selected="setup")

        def do_start_adventure():
            """When the user clicks ▶ Start Adventure on the Setup tab:
            1. If the LLM is loaded but no narration exists yet → call start_story
               so the opening scene renders. This handles the case where the user
               launched the app with no LLM, then loaded one mid-session.
            2. Switch to the Play tab regardless.
            Returns: tabs update, narration, gallery, latest_image, location, health, inventory, map.
            """
            # 1) Generate opening if needed.
            try:
                narration_now = (ui_data.get("narration") or "").strip()
                # Treat the welcome message as "no narration" so a fresh user lands
                # in the actual opening, not the help text.
                is_welcome_only = "Welcome to JMR's LLM Adventure" in narration_now
                if (
                    getattr(llm, "model_id", "").strip()
                    and (not narration_now or is_welcome_only)
                ):
                    opening_text, opening_images = start_story(state_mgr, llm, image_gen)
                    ui_data["narration"] = (
                        opening_text
                        + "\n\nType an action below and press **Send**. "
                          "Try `look around`, `inventory`, `go cave mouth`, `take torch`, or talk to the NPCs.\n"
                    )
                    ui_data["images"].extend(opening_images)
                    ui_data["debug"].append("[start] Opening scene generated on Start Adventure click")
            except Exception as e:
                ui_data["debug"].append(f"[start] Failed to generate opening: {e}")

            # 2) Build outputs for the Play tab.
            latest_img = (
                get_current_room_image(state_mgr)
                or get_inventory_item_image(state_mgr)
                or (ui_data["images"][-1] if ui_data["images"] else None)
            )
            return (
                gr.Tabs(selected="play"),
                ui_data["narration"],
                ui_data["images"],
                latest_img,
                str(state_mgr.state.location or ""),
                str(state_mgr.state.health),
                ", ".join(state_mgr.state.inventory) or "(empty)",
                state_mgr.describe_map(),
            )

        start_adventure_btn.click(
            fn=do_start_adventure,
            inputs=[],
            outputs=[tabs, narration_box, gallery, latest_image, location, health, inventory, map_box],
        )
        back_to_setup_btn.click(fn=go_to_setup, outputs=[tabs])

        # CHANGE (visited-map UI fix, May 2026): toggle the map between visited-only
        # and full views without leaving the Play tab. Updates the textbox label so
        # it's obvious which mode you're in.
        def _toggle_map(current_mode: str):
            if current_mode == "visited":
                return (
                    "full",
                    gr.update(
                        value=state_mgr.describe_full_map(),
                        label="Known map (full — ✓ visited, ? unseen)",
                        lines=8,
                    ),
                    gr.update(value="📍  Visited only"),
                )
            return (
                "visited",
                gr.update(
                    value=state_mgr.describe_map(),
                    label="Known map (visited only)",
                    lines=4,
                ),
                gr.update(value="🗺  Show full map"),
            )

        map_toggle_btn.click(
            fn=_toggle_map,
            inputs=[map_mode_state],
            outputs=[map_mode_state, map_box, map_toggle_btn],
        )

        # SIMPLE: Show deduped gallery names, using subject names when available
        # Track toggle state for image names
        _image_names_visible = {"value": False}
        
        def toggle_image_names():
            """Toggle visibility of image names and return updated text + visibility + button label"""
            _image_names_visible["value"] = not _image_names_visible["value"]
            
            if not _image_names_visible["value"]:
                # Hide the textbox
                return gr.update(visible=False), gr.update(value="Show Image Names")
            
            # Show the textbox with filenames
            try:
                # Build mapping from path -> subject (room/item) if known
                path_to_subject = {}
                for img in state_mgr.state.last_images:
                    p = img.get("path")
                    if isinstance(p, str):
                        subj = img.get("subject") or ""
                        typ = img.get("type") or ""
                        if subj:
                            label = f"{('Item' if typ=='item' else 'Room')}: {subj}" if typ in ("item", "room") else subj
                            path_to_subject[p] = label
                # Deduplicate ui gallery paths preserving order
                seen = set()
                deduped = []
                for p in ui_data.get("images", []):
                    if isinstance(p, str) and p not in seen:
                        seen.add(p)
                        deduped.append(p)
                if not deduped:
                    return gr.update(value="No images in gallery", visible=True), gr.update(value="Hide Image Names")
                # Render numbered list with simple names when known, else basename
                lines = []
                for idx, p in enumerate(deduped, start=1):
                    simple = path_to_subject.get(p) or os.path.basename(p)
                    lines.append(f"{idx}. {simple}")
                return gr.update(value="\n".join(lines), visible=True), gr.update(value="Hide Image Names")
            except Exception:
                # Safe fallback to basenames only
                paths = [p for p in ui_data.get("images", []) if isinstance(p, str)]
                if not paths:
                    return gr.update(value="No images in gallery", visible=True), gr.update(value="Hide Image Names")
                deduped = []
                seen = set()
                for p in paths:
                    if p not in seen:
                        seen.add(p)
                        deduped.append(p)
                return gr.update(value="\n".join(f"{i+1}. {os.path.basename(p)}" for i, p in enumerate(deduped)), visible=True), gr.update(value="Hide Image Names")
        
        send_btn.click(
            fn=do_send,
            inputs=[user_box],
            outputs=[narration_box, gallery, debug_md, health, inventory, map_box, latest_image, world_bible_display, location],
        )
        # Clear the input after sending
        send_btn.click(lambda: "", outputs=[user_box])
        user_box.submit(
            fn=do_send,
            inputs=[user_box],
            outputs=[narration_box, gallery, debug_md, health, inventory, map_box, latest_image, world_bible_display, location],
        )
        # Clear the input after pressing Enter
        user_box.submit(lambda: "", outputs=[user_box])
        
        # Wire the show names button
        show_names_btn.click(
            fn=toggle_image_names,
            outputs=[gallery_filenames, show_names_btn],
        )
        
        # Note: Removed auto-update of filenames when gallery changes to preserve toggle state

        # Update GameState view after each turn
        def refresh_game_state():
            return get_game_state_view()
        
        send_btn.click(
            fn=refresh_game_state,
            inputs=[],
            outputs=[game_state_view],
        )
        user_box.submit(
            fn=refresh_game_state,
            inputs=[],
            outputs=[game_state_view],
        )
 
        # Update button handlers to change color after first use
        def do_reload_llm_with_color(model_name: str, max_tokens: int):
            result = do_reload_llm(model_name, max_tokens)
            # Return status and update button to grey
            return result, gr.Button(variant="secondary")

        def do_reload_diffuser_with_color(diffuser_id: str):
            result = do_reload_diffuser(diffuser_id)
            # Return status and update button to grey
            return result, gr.Button(variant="secondary")

        reload_llm_btn.click(
            fn=do_reload_llm_with_color,
            inputs=[llm_drop, max_tokens_slider],
            outputs=[llm_status, reload_llm_btn],
        )
        reload_diff_btn.click(
            fn=do_reload_diffuser_with_color,
            inputs=[diff_drop],
            outputs=[diff_status, reload_diff_btn],
        )
        gen_bible_btn.click(
            fn=do_generate_world_bible,
            inputs=[llm_drop, theme_input, max_tokens_slider],
            # Auto-start story after world bible generation - updates narration, gallery, and image
            outputs=[wb_status, world_bible_view, gen_bible_btn, narration_box, gallery, latest_image],
        )
        
        # Update World Bible view when it changes
        def refresh_world_bible():
            return get_world_bible_view()
        
        gen_bible_btn.click(
            fn=refresh_world_bible,
            inputs=[],
            outputs=[world_bible_view],
        )
        def do_save_game_wrapper(custom_label: str):
            # allow naming the game when saving
            # If no custom name provided, add timestamp for uniqueness
            if not custom_label or not custom_label.strip():
                timestamp = datetime.now().strftime("%m%d_%H%M")
                label = f"{state_mgr.state.player_name or 'Adventure'}_{timestamp}"
            else:
                label = custom_label.strip()
            # Capture current UI extras so they are persisted with save
            # Stored on the state manager so persistence layer can access
            try:
                state_mgr._current_narration = ui_data.get("narration", "")
                state_mgr._current_debug = ui_data.get("debug", [])
            except Exception:
                state_mgr._current_narration = ""
                state_mgr._current_debug = []
            path = state_mgr.save(label=label)
            return f"Saved archive: {os.path.basename(path)}"
        save_btn.click(
            fn=do_save_game_wrapper,
            inputs=[game_name_input],
            outputs=[save_status],
        )
        def do_restart_same_bible():
            """
            Start a new game session while keeping the current WORLD_BIBLE intact.
            This resets ONLY the dynamic state (map/inventory/notes/images/recent conversation)
            and replays the opening scene. It's equivalent to replaying from the start.
            """
            try:
                # CHANGE (chunk K): emit post-mortem for the session being abandoned.
                try:
                    pm = format_session_postmortem(state_mgr)
                    if (state_mgr.state.session_turns or 0) > 0:
                        ui_data["narration"] += "\n\n" + pm + "\n"
                        ui_data["debug"].append("[postmortem] Emitted before restart")
                except Exception as _pm_e:
                    ui_data["debug"].append(f"[postmortem] failed before restart ({_pm_e})")

                # Autosave current run before resetting dynamic state
                _autosave_if_active()

                # Preserve existing image caches so we REUSE images (no re-generation)
                prev_last_images = list(state_mgr.state.last_images)
                prev_rooms_with_images = list(state_mgr.state.rooms_with_images)
                prev_items_with_images = list(state_mgr.state.items_with_images)

                # Reset dynamic state to a fresh GameState but keep player name
                player_name = state_mgr.state.player_name
                state_mgr.state = GameState(player_name=player_name)

                # Restore image caches into the fresh state for reuse
                state_mgr.state.last_images = prev_last_images
                state_mgr.state.rooms_with_images = prev_rooms_with_images
                state_mgr.state.items_with_images = prev_items_with_images

                # Clear UI caches
                # IMPORTANT: Do NOT clear the gallery; keeping it avoids slow re-generation
                ui_data["debug"].append("[restart] New game started with existing world bible")

                # Replay opening scene using existing LLM/image settings
                opening_text, opening_images = start_story(state_mgr, llm, image_gen)
                ui_data["narration"] = (opening_text + "\n\nType an action and press Send. Use the controls below to reload the LLM or MFLUX model.\n")
                # Add any newly generated opening images to the existing gallery
                ui_data["images"].extend(opening_images)

                # Refresh UI widgets
                # CHANGE (TRIVIAL): Fallback to last gallery image if no room/item image is available
                latest_img = (get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr) or (ui_data["images"][-1] if ui_data["images"] else None))
                return (
                    "✅ New game started",
                    ui_data["narration"],  # Return plain text narration
                    ui_data["images"],
                    get_debug_view(),  # Return debug data for UI
                    str(state_mgr.state.health),
                    ", ".join(state_mgr.state.inventory) or "(empty)",
                    state_mgr.describe_map(),
                    latest_img,
                    get_llm_context_view(),
                )
            except Exception as e:
                return (f"❌ Restart failed: {e}", ui_data.get("narration", ""), ui_data.get("images", []), get_debug_view(), str(state_mgr.state.health), ", ".join(state_mgr.state.inventory) or "(empty)", state_mgr.describe_map(), None, get_llm_context_view())  # CHANGE: structured JSON for gr.JSON
        restart_btn.click(
            fn=do_restart_same_bible,
            inputs=[],
            outputs=[restart_status, narration_box, gallery, debug_md, health, inventory, map_box, latest_image, world_bible_display],
        )
        def do_apply_theme(theme_text: str):
            # update WORLD_BIBLE.global_theme to enforce consistent style
            try:
                global WORLD_BIBLE
                if not theme_text.strip():
                    return "❌ Provide a theme first", theme_text
                if WORLD_BIBLE is None:
                    WORLD_BIBLE = {"global_theme": theme_text.strip()}
                else:
                    WORLD_BIBLE["global_theme"] = theme_text.strip()
                return f"✅ Theme set: {theme_text.strip()}", theme_text.strip()
            except Exception as e:
                return f"❌ Theme update failed: {e}", theme_text
        # When theme dropdown changes, update the text box with the preset
        def on_theme_dropdown_change(selected_theme):
            preset_text = PRESET_THEMES.get(selected_theme, "")
            return preset_text
        
        theme_dropdown.change(
            fn=on_theme_dropdown_change,
            inputs=[theme_dropdown],
            outputs=[theme_input],
        )
        
        apply_theme_btn.click(
            fn=do_apply_theme,
            inputs=[theme_input],
            outputs=[theme_status, theme_input],
        )

        # CHANGE (default-cave shared with browser): reload default_cave.json on demand
        # — useful if the user wants to discard a generated/loaded bible and play the
        # canonical Starfire Gem quest the browser ships.
        def do_reload_default_cave():
            global WORLD_BIBLE
            try:
                # Force-reset so loader actually reloads even if a bible is set.
                WORLD_BIBLE = None
                loaded = load_default_world_bible_if_needed()
                if not loaded:
                    return ("❌ Could not find browser_adventure/default_cave.json",
                            get_world_bible_view())
                # Reset dynamic state so seed/start_story re-seeds from the new bible.
                player_name = state_mgr.state.player_name
                state_mgr.state = GameState(player_name=player_name)
                wb = _get_world_bible() or {}
                rooms = len(wb.get("locations", []) or [])
                items = len(wb.get("key_items", []) or [])
                msg = f"✅ Loaded default cave ({rooms} rooms, {items} key items). Click ▶ Start Adventure."
                ui_data["debug"].append(f"[default_cave] reloaded ({rooms} rooms, {items} items)")
                return msg, get_world_bible_view()
            except Exception as e:
                return f"❌ Reload failed: {e}", get_world_bible_view()

        reload_default_btn.click(
            fn=do_reload_default_cave,
            inputs=[],
            outputs=[default_status, world_bible_view],
        )

        def do_toggle_advanced(val: bool):
            global ADVANCED_DIRECTIVES
            ADVANCED_DIRECTIVES = val
            ui_data["debug"].append(f"[config] Advanced directives: {'ON' if val else 'OFF'}")

        advanced_directives_cb.change(
            fn=do_toggle_advanced,
            inputs=[advanced_directives_cb],
        )

        def do_pregenerate(level: str):
            # Pre-generate images for speed: some/most/all
            global WORLD_BIBLE
            try:
                # Determine targets
                rooms = list(state_mgr.state.known_map.keys())
                items = sorted({i for v in state_mgr.state.room_items.values() for i in v})
                # CHANGE: include World Bible rooms and key items so later gameplay reuses images
                try:
                    wb = _get_world_bible() or {}
                    # Locations may be list of dicts with 'name' or list of strings
                    wb_locations = wb.get("locations", []) or []
                    for loc in wb_locations:
                        if isinstance(loc, dict):
                            nm = loc.get("name")
                        else:
                            nm = str(loc)
                        if nm and nm not in rooms:
                            rooms.append(nm)
                    # Key items may be list of dicts with 'name' or list of strings
                    wb_items = wb.get("key_items", []) or []
                    for it in wb_items:
                        if isinstance(it, dict):
                            nm = it.get("name")
                        else:
                            nm = str(it)
                        if nm:
                            items.append(nm)
                    # de-duplicate while preserving order
                    seen_r = set()
                    rooms = [r for r in rooms if not (r in seen_r or seen_r.add(r))]
                    seen_i = set()
                    items = [i for i in items if not (i in seen_i or seen_i.add(i))]
                except Exception:
                    pass
                # Scale by level
                if level == "some":
                    rooms_sel = rooms[:3]
                    items_sel = items[:3]
                elif level == "most":
                    rooms_sel = rooms[:8]
                    items_sel = items[:8]
                else:
                    rooms_sel = rooms
                    items_sel = items
                
                made = 0
                # CHANGE: Track generated vs reused counts for clear debug/status reporting
                start_generated = state_mgr.state.images_generated
                start_reused = state_mgr.state.images_reused
                
                # PREGENERATION PHASE - Populate Image Cache Before Game Starts
                # ==============================================================
                # This creates images BEFORE gameplay to avoid generation delays
                # These use the EXACT SAME tracking system as dynamic images!
                
                # Log active models to make runs auditable
                try:
                    ui_data["debug"].append(f"[tool] Models: LLM={getattr(llm, 'model_id', 'None')}, MFLUX={(getattr(image_gen, 'model_id', None) or 'None')}")
                except Exception:
                    pass
                # Generate room overviews - directly linked to map
                if image_gen:
                    for r in rooms_sel:
                        # Use helper that handles reuse checking and theming
                        out = generate_room_image_if_needed(state_mgr, image_gen, r, ui_data["debug"])
                        if out:
                            made += 1
                            ui_data["debug"].append(f"[tool] Pre-gen saved: {os.path.basename(out)} — room: {r}")
                            
                # Generate item close-ups - directly linked to items  
                if image_gen:
                    for it in items_sel:
                        # Use helper that handles reuse checking and theming
                        out = generate_item_image_if_needed(state_mgr, image_gen, it, ui_data["debug"])
                        if out:
                            made += 1
                            ui_data["debug"].append(f"[tool] Pre-gen saved: {os.path.basename(out)} — item: {it}")
                
                # Summarize counts
                newly_generated = state_mgr.state.images_generated - start_generated
                newly_reused = state_mgr.state.images_reused - start_reused
                ui_data["debug"].append(f"[tool] Pre-gen summary: generated {newly_generated}, reused {newly_reused} (level={level})")
                ui_data["debug"].append(f"[tool] Image tracking: {len(state_mgr.state.rooms_with_images)} rooms, {len(state_mgr.state.items_with_images)} items")
                
                # POPULATE GALLERY with all existing images including newly pregenerated ones
                populate_gallery_from_state(state_mgr, ui_data)
                # Update the Latest Image thumbnail to current room if available (reflect fresh cache)
                latest_img = (get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr))
                
                # CHANGE: Also return updated debug console as structured JSON so gr.JSON can render
                return f"✅ Generated {made} images (new: {newly_generated}, reused: {newly_reused}, level={level})", ui_data["images"], get_debug_view(), latest_img
            except Exception as e:
                return f"❌ Generate failed: {e}", [], get_debug_view(), None  # CHANGE: structured JSON for gr.JSON

        def do_generate_images(level: str):
            """
            Generate images for ALL world bible rooms and items based on level.
            Always includes world bible content - doesn't just regenerate existing.
            """
            # ALWAYS use full pregenerate logic - it handles both new and existing
            # This ensures all world bible rooms/items get images, not just already-tracked ones
            status, images, debug, latest = do_pregenerate(level)

            # Update button text based on whether images were generated
            has_images = bool(ui_data.get("images") or state_mgr.state.last_images)
            button_text = "Regenerate" if has_images else "Generate Images"

            return status, images, debug, latest, button_text

        generate_images_btn.click(
            fn=do_generate_images,
            inputs=[pregenerate_level],
            outputs=[generate_images_status, gallery, debug_md, latest_image, generate_images_btn],
        )
        def do_load_game_dropdown(file_name: str):
            try:
                if not file_name or file_name == "(none)":
                    # Return current UI state unchanged except status
                    return (
                        "❌ No saved game selected",
                        ui_data["images"],
                        ui_data["narration"],  # Return plain text narration
                        get_debug_view(),  # Return debug data for UI
                        str(state_mgr.state.health),
                        ", ".join(state_mgr.state.inventory) or "(empty)",
                        state_mgr.describe_map(),
                        get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr),
                        get_llm_context_view(),
                    )

                path = os.path.join(SAVE_DIR, file_name)
                # CHANGE (TRIVIAL): Clear current UI before loading new game
                ui_data["narration"] = ""
                ui_data["images"] = []
                ui_data["debug"] = []
                state_mgr.load(path)

                # POPULATE GALLERY with loaded images
                populate_gallery_from_state(state_mgr, ui_data)

                # Prefer saved narration/debug if present in the save; otherwise reconstruct
                saved_narr = getattr(state_mgr, "_loaded_narration", None)
                saved_debug = getattr(state_mgr, "_loaded_debug", None)
                if isinstance(saved_narr, str) and saved_narr.strip():
                    narration_value = saved_narr
                else:
                    # Build a concise narration recap from recent history and map
                    recap_lines: List[str] = []
                    recap_lines.append(f"Loaded: {file_name}")
                    if state_mgr.state.recent_history:
                        recap_lines.append("\nRecent conversation:")
                        for turn in state_mgr.state.recent_history:
                            player_txt = turn.get("player", "").strip()
                            resp_txt = (turn.get("response", "") or "").strip()
                            if player_txt:
                                recap_lines.append(f"> {player_txt}")
                            if resp_txt:
                                recap_lines.append(resp_txt)
                    # Always include a fresh map recap
                    map_text = state_mgr.describe_map()
                    if map_text:
                        recap_lines.append("\nKnown Map:")
                        recap_lines.append(map_text)
                    narration_value = "\n".join([ln for ln in recap_lines if ln is not None])
                ui_data["narration"] = narration_value

                # Update debug info: prefer saved debug if present
                if isinstance(saved_debug, list) and saved_debug:
                    ui_data["debug"] = saved_debug
                ui_data["debug"].append(f"[load] Loaded {file_name}")

                # Choose a sensible latest image (current room, else a recent item)
                latest_img = (get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr))
                # Ensure map_text is always defined
                safe_map_text = map_text if 'map_text' in locals() else state_mgr.describe_map()

                return (
                    f"✅ Loaded: {file_name}",
                    ui_data["images"],
                    narration_value,  # Return plain text narration
                    get_debug_view(),  # Return debug data for UI
                    str(state_mgr.state.health),
                    ", ".join(state_mgr.state.inventory) or "(empty)",
                    safe_map_text,
                    latest_img,
                    get_llm_context_view(),
                    state_mgr.state.location,
                )
            except Exception as e:
                # Provide safe fallbacks and include a latest-turn gallery best guess
                return (
                    f"❌ Load failed: {e}",
                    ui_data.get("images", []),
                    ui_data.get("narration", ""),  # Return plain text narration
                    get_debug_view(),  # Return debug data for UI
                    str(state_mgr.state.health),
                    ", ".join(state_mgr.state.inventory) or "(empty)",
                    state_mgr.describe_map(),
                    get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr),
                    get_llm_context_view(),
                )
        load_btn.click(
            fn=do_load_game_dropdown,
            inputs=[load_dropdown],
            outputs=[load_status, gallery, narration_box, debug_md, health, inventory, map_box, latest_image, world_bible_display, location],
        )

    # CHANGE: Use GRADIO_SERVER_PORT if provided; default to 7860 to avoid occupied-port crash
    try:
        _env_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    except Exception:
        _env_port = 7860
    app.launch(server_name="127.0.0.1", server_port=_env_port, share=True, show_error=True)


def main(argv: Optional[List[str]] = None) -> int:
    ensure_directories_exist()

    parser = argparse.ArgumentParser(description="LLM Adventure — MLX-LM + MFLUX on Apple Silicon")
    parser.add_argument("--player", type=str, default="Adventurer", help="Player name")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM: MLX folder name under MLX_Models, or ollama:<tag> (e.g. ollama:qwen3.6:27b-coding-mxfp8)",
    )
    parser.add_argument("--max_tokens", type=int, default=600, help="Max new tokens per turn")
    parser.add_argument("--no-images", action="store_true", help="Disable image generation entirely")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode instead of Gradio UI")
    args = parser.parse_args(argv)

    state_mgr = StateManager(GameState(player_name=args.player))

    # No interactive console prompts; UI handles model selection/reload
    model_name = args.model or ""

    # MFLUX image generation (FLUX.2-klein-9B on Apple Silicon)
    if args.no_images:
        image_gen = None
        print("Images: disabled (--no-images flag)")
    else:
        image_gen = MfluxImageGenerator(model_name="flux2-klein-9b")
        print("Images: enabled via MFLUX (FLUX.2-klein-9B)")

    # Build full model path for MLX, or pass through ollama:<tag> for Ollama.
    if model_name:
        if llm_choice_is_ollama(model_name):
            model_path = model_name
            print(f"Using LLM (Ollama): {ollama_model_from_choice(model_name)}")
        else:
            model_path = os.path.join(MLX_MODELS_DIR, model_name) if not os.path.isdir(model_name) else model_name
            print(f"Using LLM (MLX): {model_name}")
    else:
        model_path = ""
        print("Using LLM (MLX): (select in UI)")

    llm = LLMEngine(model_path=model_path, max_new_tokens=args.max_tokens, temperature=0.7)

    # World bible is loaded from .pkl files now, not separate files
    global WORLD_BIBLE
    WORLD_BIBLE = None
    print("\nWelcome to the Minimal LLM Adventure!")
    print("The world is driven by your chosen LLM. Images use MFLUX (FLUX.2-klein-9B).")

    if args.cli:
        # Run in command-line mode
        interactive_loop(state_mgr, llm, image_gen)
        return 0
    else:
        # Launch the Gradio UI (you can still use reloads to change LLMs)
        launch_gradio_ui(state_mgr, llm, image_gen)
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    raise SystemExit(main())
