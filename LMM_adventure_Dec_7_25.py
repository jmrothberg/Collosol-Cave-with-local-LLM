# JMR's LLM Adventure Game Engine
# December 4, 2025
#
# This is a fully LLM-driven adventure game where the Large Language Model
# acts as the game master, storyteller, and world engine. The LLM has complete
# creative control while using structured JSON to maintain consistent game state.
#
# DEPENDENCIES:
# - Ollama (for LLM models): https://ollama.ai
# - Gradio (for web UI): pip install gradio
# - PyTorch (for image generation): pip install torch torchvision torchaudio
# - Diffusers (for AI image generation): pip install diffusers
# - Optional: diffusion_manager.py handles all diffusion operations (local and remote)
# - Optional: Local diffusion models in ~/Diffusion_Models/ or /data/Diffusion_Models/
#   If neither remote nor local models available, images will be disabled (game still works)
#
# The LLM is instructed via system prompt to use these JSON tools, making this
# a powerful general-purpose adventure engine that can run ANY adventure the
# LLM can imagine, while maintaining proper game state through function calls.

# CRITICAL: Set CUDA environment BEFORE any torch imports!
import os
# Add CUDA to library path - this was missing!
cuda_lib_path = '/usr/local/cuda/lib64'
current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
if cuda_lib_path not in current_ld_path:
    os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib_path}:{current_ld_path}".rstrip(':')

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import json
import os
import pickle
import sys
import time
import zipfile
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
import re
import gradio as gr

import torch
import ollama
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline, StableDiffusionXLPipeline, ZImagePipeline

# Import diffusion manager (local generation only)
from diffusion_manager import ImageGenerator

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


SAVE_DIR = os.path.join(os.getcwd(), "Adventure_Game_Saved")
ART_DIR = os.path.join(os.getcwd(), "Adventure_Art")
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
        theme_suffix = ((WORLD_BIBLE or {}).get("global_theme") if 'WORLD_BIBLE' in globals() else None)
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
    
    if not check_ollama_running():
        return False, "Ollama not running. Cannot generate world bible.", None

    # Handle auto model selection
    if model_id == "(auto)":
        heavy_model_id = select_llm_interactively()
    else:
        heavy_model_id = model_id

    if not heavy_model_id:
        return False, "No heavy model selected for world bible generation.", None

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


def generate_themed_image(image_gen: "ImageGenerator", base_prompt: str, theme_suffix: Optional[str] = None) -> Optional[str]:
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


def generate_room_image_if_needed(state_mgr: "StateManager", image_gen: Optional["ImageGenerator"], room_name: str, debug_lines: Optional[List[str]] = None, narrative_context: Optional[str] = None) -> Optional[str]:
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
            if 'WORLD_BIBLE' in globals() and WORLD_BIBLE:
                for loc in WORLD_BIBLE.get("locations", []):
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


def generate_item_image_if_needed(state_mgr: "StateManager", image_gen: Optional["ImageGenerator"], item_name: str, debug_lines: Optional[List[str]] = None) -> Optional[str]:
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
    try:
        wb = WORLD_BIBLE or {}
    except Exception:
        wb = {}
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
        lines: List[str] = []
        for room, info in self.state.known_map.items():
            exits_str = ", ".join(info.get("exits", [])) or "None"
            notes = info.get("notes", "")
            note_str = f" | notes: {notes}" if notes else ""
            # Items are not displayed in the map view
            items_part = ""
            here = " (current)" if room == self.state.location else ""
            lines.append(f"- {room}{here} -> exits: {exits_str}{note_str}{items_part}")
        return "\n".join(lines) if lines else "(map is empty)"

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


class LLMEngine:
    def __init__(self, model_id: str, max_new_tokens: int = 1500, temperature: float = 0.7) -> None:
        """
        Initialize the LLM engine.
        
        Args:
            model_id: Ollama model name
            max_new_tokens: Maximum tokens to generate (1500 allows for detailed narration + complete JSON)
            temperature: Creativity setting (0.7 is balanced)
        """
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            # Use Ollama's system field plus a single combined prompt
            generate_params: Dict[str, Any] = {
                "model": self.model_id,
                "prompt": user_prompt,
                "system": system_prompt,
                "options": {
                    "num_predict": int(self.max_new_tokens),
                    "temperature": float(self.temperature),
                },
            }
            response = ollama.generate(**generate_params)
            return response.get("response", "").strip()
        except Exception as e:
            return f"(LLM error: {e})"


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


def generate_world_bible(state_mgr: StateManager, heavy_model_id: Optional[str], theme: Optional[str] = None, max_tokens: int = 4000) -> Optional[Dict[str, Any]]:
    """
    SIMPLE world bible generation - just call LLM and return result.
    Returns None if generation fails or produces invalid results.
    """
    # Ensure max_tokens is valid
    if max_tokens is None or not isinstance(max_tokens, int) or max_tokens < 250:
        max_tokens = 600  # Use safe default

    # Default cave adventure - well-designed and always available
    default_plan = {
        "objectives": [
            "Light your way through the darkness",
            "Gain passage across the underground river",
            "Unlock the ancient temple",
            "Claim the legendary treasure"
        ],
        "theme": "Dark fantasy cave with glowing crystals, ancient stone architecture, misty atmosphere",
        "locations": [
            {"name": "Entrance", "description": "rocky cave mouth with supplies left by previous explorers"},
            {"name": "Dark Passage", "description": "pitch black tunnel that requires light"},
            {"name": "Underground River", "description": "rushing water with narrow ledge"},
            {"name": "Ancient Temple", "description": "locked stone doors with riddle inscription"},
            {"name": "Treasury", "description": "golden vault guarded by stone sentinel"}
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
        "win_condition": "Obtain the golden treasure and return to entrance"
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
                if 'WORLD_BIBLE' in globals() and WORLD_BIBLE and isinstance(WORLD_BIBLE, dict):
                    enforced_theme = WORLD_BIBLE.get("global_theme")
            except Exception:
                pass

        # If still no theme, use default
        if not enforced_theme or not enforced_theme.strip():
            enforced_theme = DEFAULT_IMAGE_THEME
        
        prompt = f"""Create a COMPLETE, SOLVABLE adventure game outline for: {snapshot}

FRAMEWORK REQUIREMENTS - Every element must serve the story:
- CONSISTENT STORY ARC: Create a logical progression where each objective builds toward the win condition
- ALL ITEMS HAVE PURPOSE: Every key_item must be essential for solving puzzles, defeating monsters, or achieving objectives
- ALL NPCs HAVE REASON: Every NPC must provide crucial information, items, or access that advances the story
- INTERCONNECTED ELEMENTS: Items, NPCs, riddles, and mechanics must form a cohesive puzzle where each piece is necessary
- WINNABLE PROGRESSION: The player must be able to logically discover and use all elements to complete the game

FLEXIBLE BUT COHERENT: Be creative with the theme and setting, but ensure every element contributes to the narrative goal.



THEME REQUIREMENT:
- Use this EXACT art/world theme for the entire game and set JSON field "global_theme" to  this string:
{enforced_theme}

Include these fields:
- objectives: 4 main goals (in logical order of completion)
- theme: visual style (one paragraph)  
- locations: 5 rooms with name and description
- npcs: 2-3 characters with name, location, personality, and what they provide (hint/item/access)
- monsters: 2-3 enemies with name, location, difficulty, and weakness (what defeats them)
- riddles: 2 puzzles with location, hint, solution approach, and reward
- key_items: 6-8 items with name and purpose (include weapons, tools, keys, healing items)
- item_locations: where each key_item is found or obtained
- progression_hints: 3-4 hints about the intended solution path
- mechanics: 3-5 cause-effect rules with action, effect, and location (mechanics should enhance gameplay and support objectives, not create arbitrary obstacles)
- main_arc: story summary (2-3 sentences)
- win_condition: what specifically completes the game

Example structure (but be creative):
- Player needs torch (from entrance) to see in dark cave
- Dark cave has key (guarded by riddle)
- Key opens temple where sword is found
- Sword defeats guardian blocking treasure
- Treasure contains artifact that completes quest

Respond with JSON only. Do not include any explanatory text, markdown formatting, or anything else - just the raw JSON object."""

        # DEBUG: Show full prompt being sent to LLM
        print(f"[world_bible] PROMPT BEING SENT TO {heavy_model_id} ({len(prompt)} chars):")
        print("=" * 80)
        print(prompt)
        print("=" * 80)
        print(f"[world_bible] Calling {heavy_model_id} with num_predict={max_tokens}")

        response = ollama.generate(
            model=heavy_model_id,
            prompt=prompt,
            system="You are a game designer. Output ONLY a single valid JSON object. No markdown fences, no commentary, no text outside the JSON.",
            format="json",
            options={"temperature": 0.7, "num_predict": max_tokens}
        )

        # DEBUG: Show full Ollama response object
        print(f"[world_bible] FULL OLLAMA RESPONSE OBJECT: {response}")
        print(f"[world_bible] Response type: {type(response)}")

        # DEBUG: Show raw LLM response before any processing
        raw_response = response.get("response", "")
        print(f"[world_bible] RAW LLM RESPONSE ({len(raw_response)} chars):")
        print("=" * 80)
        print(raw_response)
        print("=" * 80)

        # Extract JSON from response - be very forgiving of LLM mistakes
        text = raw_response

        # Remove markdown if present
        text = text.replace("```json", "").replace("```", "").strip()

        # DEBUG: Show full raw JSON response
        print(f"[world_bible] FULL RAW RESPONSE FROM LLM ({len(text)} chars):")
        print("=" * 80)
        print(text)
        print("=" * 80)

        plan = None

        # Strategy 1: Try to extract complete JSON objects using balanced braces
        # NOTE: extract_json_from_text returns already-parsed dicts, not strings
        candidates = extract_json_from_text(text)
        if candidates:
            plan = candidates[0]  # Already parsed by extract_json_from_text
            print(f"[world_bible] Successfully parsed JSON candidate (from {len(candidates)} candidates)")

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

        # If all parsing failed, return None - let user retry
        if plan is None:
            print("[world_bible] No valid JSON parsed from response")
            return None

        # NORMALIZE THEME KEY: prefer single canonical key "global_theme" (from UI)
        try:
            # Always set to enforced theme from UI to avoid drift
            plan["global_theme"] = enforced_theme
        except Exception:
            pass
        
        # Validate the generated world bible
        is_valid, issues = validate_world_bible(plan)
        
        if is_valid:
            print(f"[world_bible] Successfully generated valid {len(str(plan))} char plan")
            return plan
        else:
            print(f"[world_bible] Generated plan has issues:")
            for issue in issues:
                print(f"  - {issue}")
            print(f"[world_bible] Plan rejected - generation failed")
            return None
            
    except Exception as e:
        print(f"[world_bible] Generation failed ({e})")

    return None

    # Simple, direct prompt that any OpenAI model should handle easily
    system = "You are a helpful assistant that creates adventure game plans in JSON format."
    
    user = (
        f"Create a story outline for this adventure game: {json.dumps(snapshot)}\n\n"
        "Return a JSON outline with these fields (be concise but descriptive):\n"
        "- objectives: 3-4 main goals\n"
        "- theme: visual style description (one paragraph)\n"
        "- key_npcs: 2-3 important characters with name, role, brief description\n"
        "- locations: 5-6 important rooms with name and visual description\n"
        "- key_items: 5-7 story items with name and purpose\n"
        "- main_arc: story summary (2-3 sentences)\n\n"
        "Example:\n"
        "{\n"
        '  "objectives": ["Find the lost artifact", "Defeat the guardian", "Unlock the ancient portal", "Escape the realm"],\n'
        '  "theme": "Dark fantasy cave system with bioluminescent crystals casting ethereal blue-green light, ancient stone architecture with mysterious runes, misty atmosphere with shadows dancing on wet walls",\n'
        '  "key_npcs": [\n'
        '    {"name": "Elder Sage", "role": "guide", "description": "ancient keeper of cave lore"},\n'
        '    {"name": "Stone Guardian", "role": "boss", "description": "massive golem protecting the artifact"}\n'
        '  ],\n'
        '  "locations": [\n'
        '    {"name": "Entrance Cavern", "description": "vast opening with stalactites and echoing darkness"},\n'
        '    {"name": "Crystal Gallery", "description": "chamber filled with glowing blue crystals"},\n'
        '    {"name": "Underground Lake", "description": "dark waters reflecting crystal light"},\n'
        '    {"name": "Ancient Temple", "description": "carved stone halls with mysterious altars"},\n'
        '    {"name": "Guardian Chamber", "description": "massive circular arena with stone pillars"}\n'
        '  ],\n'
        '  "key_items": [\n'
        '    {"name": "torch", "purpose": "light dark passages"},\n'
        '    {"name": "crystal key", "purpose": "unlock temple doors"},\n'
        '    {"name": "ancient map", "purpose": "reveal hidden paths"},\n'
        '    {"name": "magic sword", "purpose": "defeat guardian"},\n'
        '    {"name": "artifact", "purpose": "complete the quest"}\n'
        '  ],\n'
        '  "main_arc": "An explorer ventures into an ancient cave system seeking a legendary artifact. They must navigate treacherous passages, solve ancient puzzles, and face the Stone Guardian to claim their prize and escape."\n'
        "}\n\n"
        "Create a balanced outline - detailed enough for visualization but concise for quick processing. JSON only:"
    )

    plan = None  # Initialize at function scope

    try:
        # If no explicit model provided, fail fast with friendly behavior
        if not heavy_model_id:
            raise RuntimeError("No LLM model selected. Use the LLM dropdown first.")
        
        print(f"[world_bible] Attempting generation with model: {heavy_model_id}")
        print(f"[world_bible] System prompt length: {len(system)} chars")
        print(f"[world_bible] User prompt length: {len(user)} chars")
        print(f"[world_bible] SYSTEM PROMPT: {system}")
        print(f"[world_bible] USER PROMPT: {user}")
        
        # Use EXACT same pattern as main LLM engine that works
        try:
            print(f"[world_bible] Using same pattern as main LLM engine")
            generate_params: Dict[str, Any] = {
                "model": heavy_model_id,
                "prompt": user,
                "system": system,
                "options": {
                    "num_predict": max_tokens,  # Use slider value for consistency
                    "temperature": 0.6,
                },
            }
            response = ollama.generate(**generate_params)
            text = response.get("response", "").strip()
            print(f"[world_bible] Raw response length: {len(text)} chars")
            print(f"[world_bible] Raw response preview: {text[:300]}...")
            
            # Extract JSON from the response using balanced brace extraction
            # First try to remove markdown code fences if present
            cleaned_text = text
            if "```json" in text:
                cleaned_text = text.replace("```json", "").replace("```", "")
            elif "```" in text:
                cleaned_text = text.replace("```", "")
            
            # NOTE: extract_json_from_text returns already-parsed dicts
            candidates = extract_json_from_text(cleaned_text)
            if candidates:
                plan = candidates[0]
                print(f"[world_bible] Successfully parsed JSON candidate (from {len(candidates)} candidates)")
            
            if plan is None:
                # Try one more time with the original text
                candidates = extract_json_from_text(text)
                if candidates:
                    plan = candidates[0]
                    print(f"[world_bible] Successfully parsed JSON candidate (from original)")
            
            if plan is None:
                raise ValueError("No valid JSON found in response")
            
            # SUCCESS! We have a plan, skip to the post-processing
            
        except ollama.ResponseError as e:
            print(f"[world_bible] Ollama ResponseError in JSON mode:")
            print(f"[world_bible] Status code: {e.status_code}")
            print(f"[world_bible] Error message: {e.error}")
            print(f"[world_bible] Full error details: {str(e)}")
            raise e
        except Exception as e:
            print(f"[world_bible] Other error in JSON mode: {e}")
            print(f"[world_bible] Error type: {type(e).__name__}")
            raise e
    except Exception as e_json:
        # Only go to fallback if we don't already have a plan
        if plan is not None:
            print(f"[world_bible] Plan already generated successfully, skipping fallback")
        else:
            print(f"[world_bible] strict JSON mode failed: {e_json}")
            print(f"[world_bible] Error type: {type(e_json).__name__}")
        # Fallback: no format flag, then attempt to extract JSON blob
        if plan is None:
            try:
                print(f"[world_bible] Trying fallback mode without format=json")
                try:
                    response = ollama.generate(
                        model=heavy_model_id,
                        prompt=user,
                        system=system,
                        options={"temperature": 0.6, "num_predict": 24200},
                    )
                    text = response.get("response", "") if isinstance(response, dict) else str(response)
                    print(f"[world_bible] Fallback response length: {len(text)} chars")
                    print(f"[world_bible] Fallback response preview: {text[:200]}...")
                    print(f"[world_bible] FULL FALLBACK RAW RESPONSE:")
                    print("=" * 80)
                    print(text)
                    print("=" * 80)
                except ollama.ResponseError as e:
                    print(f"[world_bible] Ollama ResponseError in fallback mode:")
                    print(f"[world_bible] Status code: {e.status_code}")
                    print(f"[world_bible] Error message: {e.error}")
                    print(f"[world_bible] Full error details: {str(e)}")
                    raise e
                except Exception as e:
                    print(f"[world_bible] Other error in fallback mode: {e}")
                    print(f"[world_bible] Error type: {type(e).__name__}")
                    raise e
                # Extract JSON-looking segment using balanced brace extraction
                # NOTE: extract_json_from_text returns already-parsed dicts
                candidates = extract_json_from_text(text)
                print(f"[world_bible] Found {len(candidates)} JSON-like candidates")
                if candidates:
                    plan = candidates[0]
                    print(f"[world_bible] Successfully parsed candidate")
                if plan is None:
                    raise ValueError("No valid JSON found in LLM output")
            except Exception as e_free:
                print(f"[world_bible] free-form fallback failed: {e_free}")
                print(f"[world_bible] Fallback error type: {type(e_free).__name__}")
                print(f"[world_bible] Using hardcoded fallback plan")
                # Last resort: build a tiny default plan so the game can continue
                plan = {
                    "objectives": ["Explore the dungeon", "Find a key artifact", "Reach the hidden sanctuary"],
                    "acts": [{"title": "Emergence", "beats": ["Light a torch", "Map 3 rooms", "Acquire first tool"]}],
                    "key_characters": [{"name": "Guide", "role": "mentor", "notes": "Offers early tips."}],
                    "locations": [{"room": snapshot.get("start", "Start"), "theme": "gloomy caverns", "notes": "first steps"}],
                    "riddles_and_tasks": [{"room": snapshot.get("start", "Start"), "task": "Open sealed door", "requirements": ["Rusty Key"], "hint": "Listen for hollow walls"}],
                    "loot_progression": ["Lantern -> Rope -> Key -> Relic"],
                    "fail_states": ["Health hits zero", "Drop key item into chasm"],
                    "dynamic_rules": ["New exits may appear after key events"],
                    # ensure a global theme exists for consistent imagery
                    "global_theme": "Colossal cave fantasy with moody lighting"
                }
    # At this point, plan should be set (either from main try, fallback, or hardcoded)
    if plan is None:
        print(f"[world_bible] WARNING: plan is still None after all attempts, using emergency fallback")
        plan = {
            "objectives": ["Explore the mysterious caves", "Find the hidden treasure", "Defeat any guardians", "Escape safely"],
            "theme": "Dark fantasy cave with glowing crystals, ancient stone architecture, misty atmosphere with dancing shadows",
            "key_npcs": [
                {"name": "Cave Guide", "role": "helper", "description": "mysterious figure who knows the caves"},
                {"name": "Guardian", "role": "boss", "description": "ancient protector of the treasure"}
            ],
            "locations": [
                {"name": "Entrance", "description": "rocky cave mouth with cool air flowing out"},
                {"name": "Crystal Chamber", "description": "cavern filled with glowing crystals"},
                {"name": "Underground River", "description": "rushing water through stone channels"},
                {"name": "Treasury", "description": "ancient vault with stone pedestals"},
                {"name": "Exit Tunnel", "description": "narrow passage leading to daylight"}
            ],
            "key_items": [
                {"name": "torch", "purpose": "light the way"},
                {"name": "rope", "purpose": "cross chasms"},
                {"name": "key", "purpose": "unlock doors"},
                {"name": "map", "purpose": "navigate maze"},
                {"name": "treasure", "purpose": "quest goal"}
            ],
            "main_arc": "An adventurer enters mysterious caves seeking legendary treasure. They must navigate dangerous passages and overcome the ancient guardian to claim their prize."
        }
    
    # Ensure a global theme exists even if model omitted it
    try:
        if isinstance(plan, dict) and not plan.get("global_theme") and not plan.get("theme"):
            first_loc = (plan.get("locations") or [{}])[0]
            inferred = first_loc.get("theme") or "Colossal cave fantasy with moody lighting"
            plan["global_theme"] = inferred
    except Exception:
        pass
    # Pre-generate image placeholders for early content (optional, non-critical)
    try:
        # Get early rooms from state
        early_rooms = list(state_mgr.state.known_map.keys())[:3]
        
        # Get critical items from loot_progression (new format) or riddles_and_tasks (old format)
        critical_items: List[str] = []
        
        # Try new concise format (key_items)
        if "key_items" in plan and isinstance(plan["key_items"], list):
            for item in plan["key_items"][:5]:
                if isinstance(item, str):
                    critical_items.append(item.strip())
        # Try comprehensive format (loot_progression) 
        elif "loot_progression" in plan and isinstance(plan["loot_progression"], list):
            for item in plan["loot_progression"][:5]:
                if isinstance(item, str):
                    # Extract item name (before any dash description)
                    item_name = item.split(" - ")[0].strip()
                    if item_name:
                        critical_items.append(item_name)
        
        # Fallback to old format if needed
        elif "riddles_and_tasks" in plan:
            for task in plan.get("riddles_and_tasks", [])[:5]:
                reqs = task.get("requirements") or []
                for r in reqs:
                    if isinstance(r, str) and r not in critical_items:
                        critical_items.append(r)
                if len(critical_items) >= 3:
                    break
    except Exception as e:
        # This is non-critical, just log and continue
        pass
    print(f"[world_bible] SUCCESS! Generated world bible with {len(str(plan))} characters")
    return plan


# Full-featured system instructions for capable local LLMs
SYSTEM_INSTRUCTIONS = (
    "You are a text adventure narrator.\n\n"

    "YOUR OUTPUT FORMAT (follow this EXACTLY every turn):\n"
    "1. Write 2-5 sentences of narration describing what happens and what the player sees.\n"
    "2. After the narration, write a JSON block inside ```json``` fences.\n"
    "3. The JSON block MUST have this structure:\n"
    '   ```json\n'
    '   {"state_updates": { ... }, "images": ["short image prompt"]}\n'
    '   ```\n\n'

    "BEFORE YOU RESPOND, CHECK THE GAME STATE:\n"
    "- location: where player is now\n"
    "- inventory: what player carries\n"
    "- current_room_items: what is in this room\n"
    "- current_exits: where the player can go from here\n"
    "- story_context: what happened before (YOUR MEMORY from previous turns)\n"
    "- game_flags: puzzle states, timers, conditions you set\n\n"

    "AVAILABLE TOOLS (put these inside state_updates):\n"
    "  move_to: \"Room Name\"              - move player to a room\n"
    "  connect: [[\"RoomA\", \"RoomB\"]]     - link rooms bidirectionally\n"
    "  place_items: [\"item\"]              - put item in current room for player to find\n"
    "  room_take: [\"item\"]               - player picks up item from room into inventory\n"
    "  add_items: [\"item\"]               - give item directly to player inventory\n"
    "  remove_items: [\"item\"]            - remove item from inventory\n"
    "  change_health: -5 or +10           - damage or heal the player\n"
    "  set_context: \"summary\"            - save what happened (YOUR MEMORY for next turn)\n"
    "  set_flag: {\"name\": \"x\", \"value\": true} - track puzzle/event states\n"
    "  timer_event: {\"name\": \"poison\", \"duration\": 3, \"action\": \"take_damage\", \"value\": 5}\n"
    "  conditional_action: {\"condition\": \"has_key\", \"action\": \"unlock\", \"fallback\": \"door is locked\"}\n\n"

    "IMAGE PROMPTS (put inside images array):\n"
    "  images: [\"short visual description of the scene\"]\n"
    "  ALWAYS include at least one image prompt that matches your narration.\n\n"

    "EXAMPLE 1 - Player enters a new room:\n"
    "The passage opens into a pitch-black chamber. You hear dripping water echoing off distant walls. Without light, you can barely see your own hand.\n\n"
    "```json\n"
    "{\"state_updates\": {"
    "\"move_to\": \"Dark Chamber\", "
    "\"connect\": [[\"Entrance Hall\", \"Dark Chamber\"]], "
    "\"place_items\": [\"Old Torch\"], "
    "\"set_context\": \"Player entered dark room. Old torch on ground. Needs light to explore.\"}, "
    "\"images\": [\"pitch-black cavern chamber, faint water dripping, barely visible shadows\"]}\n"
    "```\n\n"

    "EXAMPLE 2 - Player uses an item:\n"
    "You insert the rusty key into the lock. It turns with a satisfying click and the ancient door swings open, revealing a golden glow beyond.\n\n"
    "```json\n"
    "{\"state_updates\": {"
    "\"remove_items\": [\"Rusty Key\"], "
    "\"set_flag\": {\"name\": \"treasury_unlocked\", \"value\": true}, "
    "\"move_to\": \"Treasury\", "
    "\"connect\": [[\"Locked Corridor\", \"Treasury\"]], "
    "\"set_context\": \"Used key to unlock treasury. Key consumed. Treasury now accessible.\"}, "
    "\"images\": [\"ancient treasury door swinging open, golden light spilling through\"]}\n"
    "```\n\n"

    "RULES:\n"
    "1. ALWAYS check inventory before letting player use items\n"
    "2. ALWAYS check current_room_items before room_take\n"
    "3. ALWAYS use connect when the player moves to a new room\n"
    "4. ALWAYS update set_context with a summary of what happened this turn\n"
    "5. ALWAYS include at least one image prompt in the images array\n"
    "6. JSON goes AFTER narration text, inside ```json``` fences\n"
    "7. Describe what the player SEES - the environment, objects, atmosphere\n"
)


def start_story(state_mgr: StateManager, llm: "LLMEngine", image_gen: Optional["ImageGenerator"], theme: Optional[str] = None) -> Tuple[str, List[str]]:
    """Ask the LLM to produce an opening scene and seed initial state via JSON.

    Returns: (opening_narration, generated_image_paths)
    """
    # Prefer theme from world bible when available
    try:
        chosen_theme = theme or ((WORLD_BIBLE or {}).get("global_theme") if 'WORLD_BIBLE' in globals() else None) or "An atmospheric fantasy in caverns and ruins beneath an ancient forest."
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
        if WORLD_BIBLE:
            # Build a richer context from world bible
            hints = []
            
            # Objectives
            if objs := WORLD_BIBLE.get("objectives", []):
                hints.append(f"objectives: {'; '.join(objs[:3])}")
            
            # NPCs in current or nearby locations
            if npcs := WORLD_BIBLE.get("npcs", WORLD_BIBLE.get("key_characters", [])):
                npc_info = [f"{n.get('name', 'Someone')} ({n.get('personality', n.get('role', 'NPC'))})" for n in npcs[:2]]
                if npc_info:
                    hints.append(f"NPCs: {', '.join(npc_info)}")
            
            # Monsters that might be encountered
            if monsters := WORLD_BIBLE.get("monsters", []):
                monster_info = [f"{m.get('name', 'creature')} ({m.get('difficulty', 'unknown')})" for m in monsters[:2]]
                if monster_info:
                    hints.append(f"monsters: {', '.join(monster_info)}")
            
            # Current health status
            if state_mgr.state.health < 50:
                hints.append(f"player health: {state_mgr.state.health}/100 (wounded)")
            elif state_mgr.state.health < 100:
                hints.append(f"player health: {state_mgr.state.health}/100")
            
            # Locations and their descriptions (CRITICAL for consistency)
            if locations := WORLD_BIBLE.get("locations", []):
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

    llm_text = llm.generate(SYSTEM_INSTRUCTIONS + wb_hint, kickoff)
    final_text, new_images, payloads, tool_logs = apply_llm_directives(state_mgr, llm_text, image_gen)
    # If LLM failed to seed a location, ensure we have at least 'Start'.
    if not state_mgr.state.location:
        state_mgr.move_to("Start")
    
    # ENSURE opening room image: If no images were generated, generate one directly
    # Pass narrative context so image matches what was described
    if not new_images and state_mgr.state.location and image_gen:
        # Generate image directly for starting room using the narrative for context
        start_image = generate_room_image_if_needed(
            state_mgr, image_gen, state_mgr.state.location,
            narrative_context=final_text
        )
        if start_image:
            new_images.append(start_image)
    
    return final_text or "Your journey begins...", new_images


def apply_llm_directives(state_mgr: StateManager, text: str, image_gen: Optional[ImageGenerator]) -> Tuple[str, List[str], List[Dict[str, Any]], List[str]]:
    image_paths: List[str] = []
    cleaned_text = text
    debug_lines: List[str] = []
    json_payloads: List[Dict[str, Any]] = []

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

    # FLEXIBLE DIRECTIVE FALLBACK SYSTEM
    # If no JSON was found but text contains directive-like patterns, try to extract them
    if not json_payloads:
        fallback_directives = extract_fallback_directives(cleaned_text)
        if fallback_directives:
            json_payloads.append(fallback_directives)
            debug_lines.append(f"Extracted fallback directives: {list(fallback_directives.keys())}")

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
            
            # Room connections (can happen anytime)
            if isinstance(updates.get("connect"), list):
                for pair in updates["connect"]:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        state_mgr.connect_rooms(str(pair[0]), str(pair[1]))
                        debug_lines.append(f"tool.connect_rooms: {pair[0]} <-> {pair[1]}")
            
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
                    debug_lines.append(f"tool.room_take: {item} (FAILED - item not in room)")
                    # Don't auto-correct - let LLM learn from its mistakes
            
            # Now move the player (after all room actions are complete)
            if isinstance(updates.get("move_to"), str):
                state_mgr.move_to(updates["move_to"]) 
                debug_lines.append(f"tool.move_to: {updates['move_to']}")
                last_room_moved_to = updates["move_to"]  # Track the move
            
            # Inventory updates (can happen anytime)
            for item in updates.get("add_items", []) or []:
                state_mgr.add_item(str(item))
                debug_lines.append(f"tool.add_item: {item}")
            for item in updates.get("remove_items", []) or []:
                state_mgr.remove_item(str(item))
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
                    # Execute the conditional action (simple mapping for now)
                    if action.startswith("unlock_"):
                        target = action[7:]
                        debug_lines.append(f"tool.conditional_action: {condition} met -> {action}")
                    elif action.startswith("reveal_"):
                        target = action[7:]
                        debug_lines.append(f"tool.conditional_action: {condition} met -> {action}")
                    else:
                        debug_lines.append(f"tool.conditional_action: {condition} met -> {action} (custom)")
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

            # Images - both LLM-requested and auto-generated for new rooms and items
            img_prompts = directives.get("images", []) if isinstance(directives, dict) else []
            # Track processed prompts to avoid duplicates
            img_prompts_lower = [p.lower() for p in img_prompts if isinstance(p, str)]
            
            # AUTO-GENERATE IMAGE FOR NEW ROOMS (Critical Reuse Point #1)
            # ============================================================
            # This handles room images when player moves to a new location
            # REUSE VERIFICATION:
            # - Pregenerated "Start" -> has_room_image("Start") = True -> Skip generation
            # - First visit to "Cave" -> has_room_image("Cave") = False -> Generate
            # - Return to "Cave" -> has_room_image("Cave") = True -> Skip generation
            if image_gen and isinstance(updates.get("move_to"), str):
                new_room = updates["move_to"]
                # SIMPLE CHECK: use our direct tracking system
                if not state_mgr.state.has_room_image(new_room):
                    # Generate an atmospheric overview for this room
                    auto_prompt = f"atmospheric fantasy {new_room.lower()}, detailed environment, adventure game location"
                    # Skip if LLM already requested this type of image
                    if not any(new_room.lower() in p for p in img_prompts_lower):
                        img_prompts = list(img_prompts) + [auto_prompt]
                        debug_lines.append(f"Auto-generating image for room: {new_room}")
                    else:
                        debug_lines.append(f"Skip auto-gen for room {new_room} - LLM already requested it")
                else:
                    state_mgr.state.images_reused += 1  # Track successful reuse!
                    # Find the actual image filename for debugging
                    existing_image = "unknown"
                    for img_data in state_mgr.state.last_images:
                        if img_data.get("type") == "room" and img_data.get("subject") == new_room:
                            existing_image = os.path.basename(img_data.get("path", "unknown"))
                            break
                    debug_lines.append(f"REUSING existing image for room: {new_room} → {existing_image} (saved generation)")
            
            # Auto-generate images for important items when placed or taken
            items_to_image = []
            
            # Check for newly placed items
            for item in updates.get("place_items", []) or []:
                item_str = str(item)
                # Check if this item has been imaged before
                item_already_imaged = any(
                    item_str.lower() in img.get("prompt", "").lower()
                    for img in state_mgr.state.last_images
                )
                if not item_already_imaged and image_gen:
                    # Skip if LLM already requested this image
                    if not any(item_str.lower() in p for p in img_prompts_lower):
                        items_to_image.append(item_str)
                        debug_lines.append(f"Auto-generating image for new item: {item_str}")
                    else:
                        debug_lines.append(f"Skip auto-gen for item {item_str} - LLM already requested it")
            
            # Check for items being taken (room_take)
            for item in updates.get("room_take", []) or []:
                item_str = str(item)
                # Important items being picked up should get a close-up
                item_already_imaged = any(
                    item_str.lower() in img.get("prompt", "").lower()
                    for img in state_mgr.state.last_images
                )
                if not item_already_imaged and image_gen:
                    # Skip if LLM already requested this image
                    if not any(item_str.lower() in p for p in img_prompts_lower):
                        items_to_image.append(item_str)
                        debug_lines.append(f"Auto-generating image for picked up item: {item_str}")
                    else:
                        debug_lines.append(f"Skip auto-gen for picked up item {item_str} - LLM already requested it")
            
            # Add item image prompts
            for item in items_to_image:
                item_prompt = f"detailed close-up of {item}, fantasy adventure game item, magical artifact"
                img_prompts = list(img_prompts) + [item_prompt]
            
            # Generate all collected image prompts
            if image_gen and isinstance(img_prompts, list):
                # Always apply global theme for consistency across images
                theme_suffix = get_theme_suffix()
                # CHANGE: add theme origin to debug for analysis (WORLD_BIBLE vs DEFAULT)
                try:
                    has_wb_theme = bool((WORLD_BIBLE or {}).get("global_theme"))
                except Exception:
                    has_wb_theme = False
                debug_lines.append(f"Theme origin: {'WORLD_BIBLE' if has_wb_theme else 'DEFAULT'}")
                # CHANGE: filter LLM-requested prompts if we already have pregenerated images for the same subject
                try:
                    move_room = updates.get("move_to") if isinstance(updates.get("move_to"), str) else None
                    placed_items = [str(x) for x in (updates.get("place_items") or [])]
                    taken_items = [str(x) for x in (updates.get("room_take") or [])]
                    known_items = set(placed_items + taken_items)
                    # TRIVIAL BOOST: also consider inventory, items visible in rooms, and world-bible key items
                    try:
                        # Live items
                        for v in state_mgr.state.room_items.values():
                            for it in v:
                                known_items.add(str(it))
                        for it in state_mgr.state.inventory:
                            known_items.add(str(it))
                        # World bible key items (names only)
                        wb = WORLD_BIBLE or {}
                        for it in (wb.get("key_items") or []):
                            nm = (it.get("name") if isinstance(it, dict) else str(it))
                            if nm:
                                known_items.add(str(nm))
                    except Exception:
                        pass
                    def _has_image_for(text: str) -> bool:
                        """
                        LLM-REQUESTED IMAGE FILTERING (Critical Reuse Point #2)
                        =======================================================
                        When LLM requests images via JSON, we check if we already have them.
                        This prevents regeneration of existing content.
                        
                        REUSE EXAMPLES:
                        - LLM requests image for "Start" -> _has_image_for("Start") = True -> Skip
                        - LLM requests image for "Torch" -> _has_image_for("Torch") = True -> Skip
                        - Works for BOTH pregenerated AND dynamically generated images!
                        """
                        try:
                            # Check if it's a room name (case-insensitive exact match)
                            t = str(text).strip().lower()
                            if any((t == str(r).strip().lower()) for r in state_mgr.state.rooms_with_images):
                                state_mgr.state.images_reused += 1
                                return True
                            # Check if it's an item name (case-insensitive exact match)  
                            if any((t == str(i).strip().lower()) for i in state_mgr.state.items_with_images):
                                state_mgr.state.images_reused += 1
                                return True
                            return False
                        except Exception:
                            return False
                    # Determine target room (new move if present, otherwise current location)
                    try:
                        target_room = move_room or state_mgr.state.location
                    except Exception:
                        target_room = move_room
                    has_room_image = bool(target_room and _has_image_for(str(target_room)))
                    skipped_room_count = 0
                    skipped_env_count = 0
                    skipped_item_count = 0
                    filtered_prompts: List[str] = []
                    for p in img_prompts:
                        if not isinstance(p, str) or not p.strip():
                            continue
                        lc = p.lower()
                        # Skip if it's a room prompt and we already have an image for that room
                        if move_room and move_room.lower() in lc and _has_image_for(move_room):
                            debug_lines.append(f"Reuse existing image for room: {move_room}")
                            skipped_room_count += 1
                            continue
                        # If we already have a room image, skip ALL environment prompts unless it's a close-up
                        if has_room_image:
                            is_closeup = ("close-up" in lc) or ("close up" in lc) or ("closeup" in lc)
                            if not is_closeup:
                                debug_lines.append(f"Reuse existing room image for: {target_room} (skipping env prompt)")
                                skipped_env_count += 1
                                continue
                        # Skip if it's an item prompt and we already have an image for that item
                        skip_item = False
                        for it in known_items:
                            if it.lower() in lc and _has_image_for(it):
                                # CHANGE (DEBUG ONLY): include the cached filename when reusing an item image
                                try:
                                    _added_detail = False
                                    for _img in state_mgr.state.last_images:
                                        if _img.get("type") == "item" and str(_img.get("subject")).lower() == str(it).lower():
                                            _p = _img.get("path")
                                            if _p and os.path.exists(_p):
                                                debug_lines.append(f"Reuse existing image for item: {it} → {os.path.basename(_p)}")
                                                _added_detail = True
                                                break
                                    if not _added_detail:
                                        debug_lines.append(f"Reuse existing image for item: {it}")
                                except Exception:
                                    debug_lines.append(f"Reuse existing image for item: {it}")
                                skipped_item_count += 1
                                skip_item = True
                                break
                        if skip_item:
                            continue
                        filtered_prompts.append(p)
                except Exception:
                    filtered_prompts = [p for p in img_prompts if isinstance(p, str) and p.strip()]
                # CHANGE: one-line summary of image filtering
                try:
                    debug_lines.append(
                        f"Image filter: skipped room={locals().get('skipped_room_count', 0)}, env={locals().get('skipped_env_count', 0)}, item={locals().get('skipped_item_count', 0)}, to_generate={len(filtered_prompts)}"
                    )
                except Exception:
                    pass
                # CHANGE: diffuser summary (once per batch)
                try:
                    ptype = type(getattr(image_gen, "_pipeline", None)).__name__ if getattr(image_gen, "_pipeline", None) is not None else "(lazy)"
                    est_steps = 4 if ptype == "FluxPipeline" else 10
                    debug_lines.append(f"Diffuser: model={image_gen.model_id or '(none)'}, device={image_gen.device}, pipeline={ptype}, steps≈{est_steps}")
                except Exception:
                    pass
                # TRIVIAL FIX: prevent duplicate room images in the same turn
                # Track which room subjects have already had an image generated this batch
                generated_rooms_this_batch: Set[str] = set()
                for p in filtered_prompts:
                    if isinstance(p, str) and p.strip():
                        final_prompt = p.strip()
                        # TRIVIAL GUARD: if LLM passed an existing filename, reuse it instead of regenerating
                        try:
                            base = os.path.basename(final_prompt)
                            if base.lower().endswith(('.png', '.jpg', '.jpeg')):
                                candidate_paths = [
                                    os.path.join(ART_DIR, base),
                                    os.path.join(IN_GAME_IMG_DIR, base),
                                    final_prompt if os.path.isabs(final_prompt) else None,
                                ]
                                candidate_paths = [p0 for p0 in candidate_paths if p0]
                                existing = next((cp for cp in candidate_paths if os.path.exists(cp)), None)
                                if existing:
                                    if existing not in image_paths:
                                        image_paths.append(existing)
                                    debug_lines.append(f"[reuse] Using existing image file: {os.path.basename(existing)}")
                                    # Also mark the room subject as generated if identifiable to avoid duplication this turn
                                    try:
                                        original_prompt_lower = p.strip().lower()
                                        for room_name in state_mgr.state.known_map.keys():
                                            if room_name.lower() in original_prompt_lower:
                                                generated_rooms_this_batch.add(room_name)
                                                break
                                    except Exception:
                                        pass
                                    continue
                                # If the filename doesn't exist locally, trivially reuse context-linked images
                                # Prefer current room, then a recent inventory item, to avoid wasteful regeneration
                                try:
                                    room_path = get_current_room_image(state_mgr)
                                    if room_path:
                                        if room_path not in image_paths:
                                            image_paths.append(room_path)
                                        debug_lines.append(f"[reuse] Using current room image for filename hint: {os.path.basename(room_path)}")
                                        continue
                                    item_path = get_inventory_item_image(state_mgr)
                                    if item_path:
                                        if item_path not in image_paths:
                                            image_paths.append(item_path)
                                        debug_lines.append(f"[reuse] Using inventory item image for filename hint: {os.path.basename(item_path)}")
                                        continue
                                except Exception:
                                    pass
                        except Exception:
                            pass
                        # Identify target room before spending diffuser time; skip if already done this turn
                        try:
                            original_prompt_lower = p.strip().lower()
                            target_room_for_prompt: Optional[str] = None
                            for room_name in state_mgr.state.known_map.keys():
                                if room_name.lower() in original_prompt_lower:
                                    target_room_for_prompt = room_name
                                    break
                            if target_room_for_prompt and target_room_for_prompt in generated_rooms_this_batch:
                                debug_lines.append(f"Skip duplicate room image this turn: {target_room_for_prompt}")
                                continue
                        except Exception:
                            pass
                        # Identify target item before spending diffuser time
                        try:
                            original_prompt_lower = p.strip().lower()
                            target_item_for_prompt: Optional[str] = None
                            all_items_pre = set(state_mgr.state.inventory)
                            for room_items in state_mgr.state.room_items.values():
                                all_items_pre.update(room_items)
                            for item_name in all_items_pre:
                                if isinstance(item_name, str) and item_name.lower() in original_prompt_lower:
                                    target_item_for_prompt = item_name
                                    break
                            # CHANGE: If not in placed items, check if it's in world bible key_items (LLM knows about these)
                            if not target_item_for_prompt:
                                try:
                                    wb_items = (WORLD_BIBLE or {}).get("key_items", []) or []
                                    for it in wb_items:
                                        nm = (it.get("name") if isinstance(it, dict) else str(it))
                                        if isinstance(nm, str) and nm.lower() in original_prompt_lower:
                                            target_item_for_prompt = nm
                                            debug_lines.append(f"[track] Item from world bible: {nm}")
                                            break
                                except Exception:
                                    pass
                        except Exception:
                            target_item_for_prompt = None
                        # CHANGE: If LLM explicitly requested this image in JSON, honor it even if we can't identify the subject
                        # This respects the LLM's narrative focus
                        if not target_room_for_prompt and not target_item_for_prompt:
                            # Check if this prompt came from the LLM's JSON images array
                            if p in img_prompts:
                                debug_lines.append(f"[track] LLM-requested image (narrative focus): {p}")
                                # Continue to generation since LLM explicitly wants this
                            else:
                                debug_lines.append("[skip] Image prompt not linked to known room or item; generation skipped")
                                continue
                        # CHANGE: explicitly enforce the exact world art style entered by user
                        if theme_suffix:
                            if theme_suffix.lower() not in final_prompt.lower():
                                final_prompt = f"{final_prompt}. World art style: {theme_suffix}."
                            final_prompt = f"{final_prompt} Strictly adhere to this style."
                        debug_lines.append(f"Generating image: {final_prompt}")
                        # CHANGE (TRIVIAL): Log concise image prompt info for LLM-first debugging (no logic change)
                        try:
                            _src = "LLM" if p in img_prompts else "AUTO"
                            _p_prev = (final_prompt[:120] + "...") if len(final_prompt) > 120 else final_prompt
                            _t_prev = ((theme_suffix[:80] + "...") if theme_suffix and len(theme_suffix) > 80 else (theme_suffix or ""))
                            debug_lines.append(f"[image_prompt] src={_src} len={len(final_prompt)} prompt=\"{_p_prev}\" theme=\"{_t_prev}\"")
                        except Exception:
                            pass
                        out_path = image_gen.generate(final_prompt)
                        if out_path:
                            # SMART TRACKING: figure out what this image is for and track it properly
                            original_prompt = p.strip().lower()
                            
                            # Check if this is a room image (simple like inventory)
                            room_identified = False
                            for room_name in state_mgr.state.known_map.keys():
                                if room_name.lower() in original_prompt:
                                    state_mgr.state.add_room_image(room_name, out_path, p.strip())
                                    debug_lines.append(f"[track] ROOM IMAGE: '{room_name}' → {os.path.basename(out_path)}")
                                    # Prevent further room images for same subject within this turn
                                    try:
                                        generated_rooms_this_batch.add(room_name)
                                    except Exception:
                                        pass
                                    room_identified = True
                                    # Record for display only when identified
                                    image_paths.append(out_path)
                                    # CHANGE: include filename and brief description for debug console
                                    try:
                                        fname = os.path.basename(out_path)
                                    except Exception:
                                        fname = out_path
                                    short_desc = (final_prompt[:80] + "...") if len(final_prompt) > 80 else final_prompt
                                    debug_lines.append(f"Image saved: {fname} — {short_desc}")
                                    break
                            
                            # Check if this is an item image (if not identified as room)
                            if not room_identified:
                                item_identified = False
                                # Check current room items and inventory
                                all_items = set(state_mgr.state.inventory)
                                for room_items in state_mgr.state.room_items.values():
                                    all_items.update(room_items)
                                
                                for item_name in all_items:
                                    if item_name.lower() in original_prompt:
                                        state_mgr.state.add_item_image(item_name, out_path, p.strip())
                                        debug_lines.append(f"[track] ITEM IMAGE: '{item_name}' → {os.path.basename(out_path)}")
                                        item_identified = True
                                        # Record for display only when identified
                                        image_paths.append(out_path)
                                        # CHANGE: include filename and brief description for debug console
                                        try:
                                            fname = os.path.basename(out_path)
                                        except Exception:
                                            fname = out_path
                                        short_desc = (final_prompt[:80] + "...") if len(final_prompt) > 80 else final_prompt
                                        debug_lines.append(f"Image saved: {fname} — {short_desc}")
                                        break
                                
                                # If we can't identify it, still track it the old way
                                if not item_identified:
                                    # CHANGE: Do not track or display unidentified images
                                    debug_lines.append("[skip] Generated image did not match a room or item; discarded from tracking")
                        else:
                            debug_lines.append(f"Image generation failed for: {p.strip()}")
                # CHANGE: image cache size after generation
                try:
                    debug_lines.append(f"Images cached: {len(state_mgr.state.last_images)}")
                except Exception:
                    pass
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
    return cleaned_text, image_paths, json_payloads, debug_lines


def check_ollama_running() -> bool:
    try:
        _ = ollama.list()
        return True
    except Exception as e:
        print(f"Error: Ollama is not running or not accessible: {e}")
        print("Please start Ollama with 'ollama serve' in a terminal")
        return False


def list_ollama_models() -> List[str]:
    try:
        response = ollama.list()
        models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
        names: List[str] = []
        for m in models:
            name = m.get("model") or m.get("name") or ""
            if name:
                names.append(name)
        return names
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        return []


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
    root = diffusion_models_root()
    if not os.path.isdir(root):
        return []
    try:
        names = [n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)) and not n.startswith('.')]
        return sorted(names)
    except Exception:
        return []


def curated_hf_diffusers() -> List[str]:
    # Only include models that are actually available locally
    return []


def saved_diffusers() -> List[str]:
    # Merge local folders with curated HF IDs, deduplicated while preserving order
    local_models = list_local_diffusers()
    hf_models = curated_hf_diffusers()

    # Optional device/memory note (mirrors original intent)
    if torch.cuda.is_available():
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            total_memory = 8
        if total_memory >= 26:
            print("Added FLUX models for high-memory CUDA")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Added FLUX models optimized for Mac MPS")

    combined: List[str] = []
    seen = set()
    for name in (local_models + hf_models):
        if name not in seen:
            seen.add(name)
            combined.append(name)
    return combined if combined else ["No diffusion models found"]


def combined_diffuser_choices(include_skip: bool = True) -> List[str]:
    combined: List[str] = saved_diffusers()
    # Don't add any blank or "(none)" option - just show available models
    return combined


def select_llm_interactively() -> str:
    if not check_ollama_running():
        raise SystemExit(1)
    models = list_ollama_models()
    if not models:
        print("No Ollama models found. Install one with: ollama pull llama3.1:8b or similar")
        raise SystemExit(1)
    print("Select an Ollama model (same list as GUI):")
    # Sort like Colossal Cave: common first, then rest
    common = ['llama3.1', 'llama3', 'llama2', 'mistral', 'phi4', 'phi3', 'gemma3', 'gemma2', 'qwen3', 'qwen2.5']
    ordered: List[str] = []
    for c in common:
        for m in models:
            if m.startswith(c) and m not in ordered:
                ordered.append(m)
    for m in models:
        if m not in ordered:
            ordered.append(m)
    models = ordered
    for idx, name in enumerate(models, 1):
        print(f"  {idx}. {name}")
    choice = input("Enter a number (default 1): ").strip() or "1"
    try:
        idx = max(1, min(len(models), int(choice))) - 1
    except Exception:
        idx = 0
    return models[idx]


def maybe_select_diffuser_interactively() -> Optional[str]:
    print("Image generation is optional. Select a diffuser model or press Enter to skip:")
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


def build_user_prompt(state_mgr: StateManager, player_input: str) -> str:
    state = state_mgr.state
    # Enrich context so the LLM can decide importance without engine heuristics
    current_items = state.room_items.get(state.location, [])
    current_exits = state.known_map.get(state.location, {}).get("exits", [])
    visited_rooms = list(state.known_map.keys())
    
    # Enhanced context with better structure and prioritization
    context = {
        # CORE GAME STATE (always include)
        "location": state.location,
        "health": state.health,
        "inventory": state.inventory,

        # NAVIGATION & EXPLORATION (high priority)
        "current_exits": current_exits,
        "current_room_items": current_items,
        "known_map": state.known_map,
        "visited_rooms": visited_rooms,

        # NARRATIVE & MEMORY (medium priority)
        "story_context": state.story_context,
        "recent_conversation": state.recent_history[-3:] if len(state.recent_history) > 3 else state.recent_history,  # Limit for token efficiency
        "notes": state.notes[-5:] if len(state.notes) > 5 else state.notes,  # Most recent notes

        # ADVANCED GAME STATE (contextual)
        "game_flags": state.game_flags,
        "player_name": state.player_name,

        # WORLD BUILDING (when available)
        "world_theme": get_theme_suffix(),
        "win_condition": (WORLD_BIBLE.get("win_condition") if 'WORLD_BIBLE' in globals() and WORLD_BIBLE else None),

        # CREATIVE AIDS (for image generation)
        "rooms_with_images": state.rooms_with_images,
        "items_with_images": state.items_with_images,

        # DEBUG INFO (only when needed)
        "debug_info": {
            "total_images": len(state.last_images),
            "images_reused": state.images_reused,
            "active_timers": [k for k, v in state.game_flags.items() if isinstance(v, dict) and v.get("type") == "timer"],
            "active_chains": [k for k, v in state.game_flags.items() if isinstance(v, dict) and v.get("type") == "chain_reaction"]
        }
    }
    # Include world bible context to guide gameplay
    bible_line = ""
    try:
        if WORLD_BIBLE:
            # WORLD BIBLE → CUES (compact, relevant excerpts)
            # We DO NOT send the entire world bible each turn. Instead, we derive
            # short, situational cues that are directly relevant to the current
            # location and gameplay. This keeps the prompt small while preserving
            # design intent.
            cues = []

            # Include location description if available (for narrative consistency)
            if locations := WORLD_BIBLE.get("locations", []):
                for loc in locations:
                    if isinstance(loc, dict) and loc.get("name", "").lower() == state.location.lower():
                        desc = loc.get("description")
                        if desc:
                            cues.append(f"Location description: {desc}")
                            break  # Only include the current location's description

            # Check for NPCs in current location
            if npcs := WORLD_BIBLE.get("npcs", WORLD_BIBLE.get("key_characters", [])):
                for npc in npcs:
                    if npc.get("location", "").lower() == state.location.lower():
                        provides = npc.get('provides', '')
                        cues.append(f"NPC here: {npc.get('name')} - {npc.get('personality', 'mysterious')}")
                        if provides:
                            cues.append(f"  (can provide: {provides})")
            
            # Check for monsters in current location
            if monsters := WORLD_BIBLE.get("monsters", []):
                for monster in monsters:
                    if monster.get("location", "").lower() == state.location.lower():
                        weakness = monster.get('weakness', '')
                        cues.append(f"Monster here: {monster.get('name')} ({monster.get('difficulty', 'dangerous')})")
                        if weakness and any(item in state.inventory for item in ['torch', 'sword', 'key']):
                            cues.append(f"  (weakness: {weakness})")
            
            # Check for riddles in current location
            if riddles := WORLD_BIBLE.get("riddles", []):
                for riddle in riddles:
                    if riddle.get("location", "").lower() == state.location.lower():
                        cues.append(f"Puzzle: {riddle.get('hint', 'something mysterious')}")
                        if riddle.get('reward'):
                            cues.append(f"  (solving grants: {riddle.get('reward')})")
            
            # Check for mechanics in current location
            if mechanics := WORLD_BIBLE.get("mechanics", []):
                for mech in mechanics:
                    if mech.get("location", "").lower() == state.location.lower():
                        cues.append(f"Mechanic: {mech.get('action')} -> {mech.get('effect')}")
            
            # Check for items that should be in this location
            if item_locs := WORLD_BIBLE.get("item_locations", {}):
                for item, loc_desc in item_locs.items():
                    if state.location.lower() in loc_desc.lower() and item not in state.inventory:
                        # Check if item should be available based on game logic
                        item_info = next((i for i in WORLD_BIBLE.get("key_items", []) if i.get("name") == item), {})
                        if item_info:
                            cues.append(f"Item available: {item} - {item_info.get('purpose', 'useful item')}")
            
            # Add current objective based on inventory/progress
            if objectives := WORLD_BIBLE.get("objectives", []):
                # Simple heuristic: which objective are we on based on items collected
                treasure_items = ['torch', 'rope', 'map', 'key', 'sword', 'treasure']
                items_collected = sum(1 for item in treasure_items if item in state.inventory)
                current_obj_idx = min(items_collected, len(objectives) - 1)
                cues.append(f"Current goal: {objectives[current_obj_idx]}")
            
            # SIMPLE IMAGE TRACKING (belongs in game state context)
            # Direct connection to existing map/item structures  
            if state.rooms_with_images:
                cues.append(f"Rooms with images: {', '.join(state.rooms_with_images[:5])}")
            if state.items_with_images:
                cues.append(f"Items with images: {', '.join(state.items_with_images[:5])}")
            
            # Add progression hints if stuck (no items collected recently)
            if len(state.inventory) < 2 and (hints := WORLD_BIBLE.get("progression_hints", [])):
                cues.append(f"Hint: {hints[0]}")
            
            if cues:
                bible_line = f"\nWorld context:\n" + "\n".join(f"  {c}" for c in cues)
    except Exception:
        bible_line = ""

    header = (
        "Game State (read-only; update via JSON directives only):\n" +
        json.dumps(context, ensure_ascii=False, indent=2) +
        bible_line +
        "\n---\n"
    )
    return header + f"Player says: {player_input}"


def interactive_loop(state_mgr: StateManager, llm: LLMEngine, image_gen: Optional[ImageGenerator]) -> None:
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
            print("\nGoodbye!")
            break

        if not user:
            continue
        if user.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        if user.lower() == "help":
            print("Commands: help, save, map, inv, health, images, quit")
            continue
        if user.lower() == "save":
            path = state_mgr.save()
            print(f"Saved: {path}")
            continue
        if user.lower() == "map":
            print(state_mgr.describe_map())
            continue
        if user.lower() == "inv":
            print(", ".join(state_mgr.state.inventory) or "(empty)")
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

        system_prompt = SYSTEM_INSTRUCTIONS
        user_prompt = build_user_prompt(state_mgr, user)
        llm_text = llm.generate(system_prompt, user_prompt)
        final_text, new_images, payloads, tool_logs = apply_llm_directives(state_mgr, llm_text, image_gen)
        
        # Only show the cleaned narration text to the user
        print("\n" + final_text.strip())
        
        # Optionally show images if generated (user-friendly)
        if new_images:
            print(f"\n[New images generated: {len(new_images)} image(s)]")
            
        # Debug output only if DEBUG environment variable is set
        if os.environ.get("DEBUG", "").lower() in {"1", "true", "yes"}:
            if payloads:
                print("[debug:json] " + json.dumps(payloads, ensure_ascii=False, indent=2))
            if tool_logs:
                for line in tool_logs:
                    print("[debug:tool] " + line)


def do_generate_world_bible(wb_model: str, theme: Optional[str] = None, max_tokens: int = 4000) -> str:
    """Standalone world bible generation for command line usage."""
    success, message, _ = execute_world_bible_generation(StateManager(), wb_model, theme, max_tokens)
    if success:
        print(message)
    return message


def launch_gradio_ui(state_mgr: StateManager, llm: LLMEngine, image_gen: Optional[ImageGenerator]) -> None:
    """Simple Gradio UI that shows images, inventory, health, map, and a debug console."""

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
    
    # Generate opening scene only if an LLM is selected; otherwise show instructions
    if getattr(llm, "model_id", "").strip():
        opening_text, opening_images = start_story(state_mgr, llm, image_gen)
        ui_data["narration"] = (opening_text + "\n\nType an action and press Send. Use the controls below to reload the LLM or diffuser.\n")
        ui_data["images"].extend(opening_images)
    else:
        ui_data["narration"] = (
            "Welcome! Select an Ollama model in the dropdown and click 'Load/Reload LLM' to begin.\n"
            "You can also load a diffuser for images (optional)."
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

        system_prompt = SYSTEM_INSTRUCTIONS
        user_prompt = build_user_prompt(state_mgr, user_text)
        llm_text = llm.generate(system_prompt, user_prompt)
        
        # Add raw LLM response to debug only
        ui_data["debug"].append(f"[llm_raw] {llm_text}")

        # Track how many images existed before this turn to identify new ones
        _prev_image_count = len(state_mgr.state.last_images)

        final_text, new_images, payloads, tool_logs = apply_llm_directives(state_mgr, llm_text, image_gen)
        
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
        
        # Add JSON and tool logs to debug console only
        # Payloads already contain parsed JSON
        if payloads:
            # Just log that we have JSON, not the full content (it's in tool logs)
            ui_data["debug"].append(f"[json] processed {len(payloads)} directive(s)")
        else:
            ui_data["debug"].append("[json] none")
            
        if tool_logs:
            for line in tool_logs:
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

    def do_reload_llm(ollama_model: str, max_tokens: int):
        # Reload just the LLM
        nonlocal llm
        # CHANGE: skip reload if model unchanged to avoid no-op reinitialization
        if ollama_model and getattr(llm, "model_id", None) == ollama_model and getattr(llm, "max_new_tokens", None) == max_tokens:
            _append_debug("[reload] LLM unchanged; skipping")
            return f"✅ LLM unchanged: {ollama_model}"
        if ollama_model:
            # Clean up old LLM if needed (Ollama manages its own memory)
            llm = LLMEngine(model_id=ollama_model, max_new_tokens=max_tokens, temperature=0.8)
            _append_debug(f"[reload] LLM -> {ollama_model} ({max_tokens} tokens)")
            return f"✅ LLM loaded: {ollama_model} ({max_tokens} tokens)"
        return "❌ No LLM model selected"

    def do_reload_diffuser(diffuser_id: str):
        # Reload just the diffuser with proper cleanup
        nonlocal image_gen

        # TRIVIAL GUARD: skip reload if model unchanged to avoid unnecessary pipeline reinit
        if diffuser_id and image_gen and getattr(image_gen, "model_id", None) == diffuser_id:
            _append_debug("[reload] Diffuser unchanged; skipping")
            return f"✅ Diffuser unchanged: {diffuser_id}"

        # Clean up the old diffuser first
        if image_gen:
            image_gen.cleanup()

        if diffuser_id:
            # Unified ImageGenerator handles both local and server automatically
            image_gen = ImageGenerator(model_id=diffuser_id, device_preference=None, local_root=diffusion_models_root())
            _append_debug(f"[reload] Diffuser -> {diffuser_id}")
            return f"✅ Diffuser loaded: {diffuser_id}"
        else:
            image_gen = None
            _append_debug("[reload] Diffuser -> disabled")
            return "✅ Diffuser disabled"
    
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
 
    # Note: css parameter may need to be passed differently in newer Gradio versions
    custom_css = "#narration-box textarea { font-size: 150%; }"
    try:
        # Try newer Gradio API first
        app = gr.Blocks(title="JMR's LLM Adventure")
        app.css = custom_css
    except Exception:
        pass
    with gr.Blocks(title="JMR's LLM Adventure") as app:
        gr.Markdown("## JMR's LLM Adventure (Ollama + Diffusers)")
 
        with gr.Row():
            with gr.Column(scale=3):
                # Instructions for new players in collapsible accordion (starts open)
                with gr.Accordion("📖 How to Play", open=True):
                    gr.Markdown("""
                    ### Quick Start Guide
                    
                    **Starting a New Game:**
                    1. **Select an Ollama model** from the dropdown (e.g., gpt-oss:20b)
                    2. **Click Load/Reload LLM** (orange button) to load the model
                    3. **Optional:** Set the Theme (world/art style)
                    4. **Optional:** Pick and LLM and click "Generate World Bible" for a new adventure (or use the default)
                    5. **Optional:** Load a diffuser for visual art (select and click Load/Reload)
                    6. **Start playing!** Type commands in the action box below
                    
                    **Loading a Saved Game:**
                    1. Select from "Saved Games" dropdown
                    2. Click "Load Game" button
                    3. Continue your adventure!
                    
                    **Playing Tips:**
                    - Natural language commands: "look around", "go north", "take sword", "talk to wizard"
                    - The Turn-by-Turn LLM embelishes the story dynamically!
                    - Save your progress anytime with "Save Game"
                    - Images are generated automatically for new locations (or pre-generate)
                    - Pregenerate images to speed up gameplay
                    
                    **Want to know more?**
                    - Check the Debug Console for operational details
                    - View LLM Context to see what the AI is being told
                    - View the World Bible to see the static game design
                    - View the GameState to see the dynamic game data
                   
                    """)
                
                # CHANGE (TRIVIAL): Initialize Latest Image so opening image shows immediately on new game
                latest_image = gr.Image(label="Latest Image", height=768, width=768, value=(get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr) or (ui_data["images"][-1] if ui_data["images"] else None)))
                # CHANGE: Removed secondary latest-turn gallery per request; one latest image panel only
                # CHANGE: Wrap main Image Gallery in a collapsible accordion for easy hiding
                with gr.Accordion("Image Gallery", open=False):
                    # CHANGE (TRIVIAL): Initialize Gallery with any opening images so they display immediately
                    gallery = gr.Gallery(label="Image Gallery", height=200, columns=3, value=ui_data["images"])
                    # Toggle button to show/hide image names
                    show_names_btn = gr.Button("Show Image Names", size="sm")
                    gallery_filenames = gr.Textbox(label="Gallery Image Filenames", interactive=False, lines=3, visible=False)
                user_box = gr.Textbox(label="Your action", placeholder="look around, go north, pick up key ...", lines=1)  # Enter submits
                send_btn = gr.Button("Send")
                with gr.Row():
                    with gr.Column():
                        ollama_models = list_ollama_models() if check_ollama_running() else []
                        choices = ollama_models or ["(select later)"]
                        default_choice = choices[0]
                        llm_drop = gr.Dropdown(choices, label="Ollama Model", value=default_choice)
                        reload_llm_btn = gr.Button("Load/Reload Turn-LLM", variant="primary")
                        llm_status = gr.Markdown()
                        # Save/Load/New Game buttons - grouped below LLM controls
                        with gr.Row():
                            restart_btn = gr.Button("New Game", variant="secondary", size="sm")
                            save_btn = gr.Button("Save", variant="primary", size="sm")
                        with gr.Row():
                            game_name_input = gr.Textbox(label="Save as Name", placeholder="My_Adventure", scale=2)
                        # Saved games dropdown and load
                        def _list_saved_games_local():
                            try:
                                files = [f for f in os.listdir(SAVE_DIR) if f.startswith("Adv_") and (f.endswith('.pkl') or f.endswith('.tkl'))]
                            except Exception:
                                files = []
                            return sorted(files, reverse=True)
                        saved_choices = _list_saved_games_local() or ["(none)"]
                        load_dropdown = gr.Dropdown(saved_choices, label="Saved Games", value=(saved_choices[0] if saved_choices else "(none)"))
                        load_btn = gr.Button("Load Game", variant="secondary", size="sm")
                        save_status = gr.Markdown()
                        load_status = gr.Markdown()
                        restart_status = gr.Markdown()
                    with gr.Column():
                        diffuser_choices = combined_diffuser_choices(include_skip=True)
                        # If no diffuser is currently loaded, default to the first available choice (like LLM dropdown)
                        current_diffuser = image_gen.model_id if image_gen else (diffuser_choices[0] if diffuser_choices else "")
                        if current_diffuser and current_diffuser not in diffuser_choices:
                            diffuser_choices = [current_diffuser] + diffuser_choices
                        diff_drop = gr.Dropdown(diffuser_choices, label="Diffuser (local or HF)", value=current_diffuser)
                        reload_diff_btn = gr.Button("Load/Reload Art Diffuser", variant="primary")  # Start orange
                        diff_status = gr.Markdown()
            with gr.Column(scale=2):
                # Scrollable narration window with box border - plain text display
                narration_box = gr.Textbox(value=ui_data["narration"], label="Narration", lines=10, max_lines=10, interactive=False, elem_id="narration-box")

                # Max tokens slider for LLM response length control
                max_tokens_slider = gr.Slider(minimum=250, maximum=2500, value=2500, step=50, label="LLM Max Tokens (response length)")

                with gr.Row():
                    location = gr.Textbox(label="Location", value=str(state_mgr.state.location), interactive=False)
                    health = gr.Textbox(label="Health", value=str(state_mgr.state.health), interactive=False)
                inventory = gr.Textbox(label="Inventory", value=", ".join(state_mgr.state.inventory), interactive=False)
                # CHANGE: Make Known Map scroll similarly to Narration by matching visible lines
                map_box = gr.Textbox(label="Known Map", value=state_mgr.describe_map(), lines=12, interactive=False)
                
                try:
                    _wb_theme = ((WORLD_BIBLE or {}).get("global_theme") if 'WORLD_BIBLE' in globals() else None)
                except Exception:
                    _wb_theme = None
                # Theme selection - dropdown with presets + custom text option
                theme_dropdown = gr.Dropdown(
                    choices=list(PRESET_THEMES.keys()),
                    label="Theme Preset",
                    value="Tolkien Cave Adventure"
                )
                theme_input = gr.Textbox(
                    label="Theme Details (edit or type custom)",
                    value=PRESET_THEMES.get("Tolkien Cave Adventure", ""),
                    lines=3,
                    placeholder="Describe your adventure theme, characters, and art style..."
                )
                with gr.Row():
                    apply_theme_btn = gr.Button("Apply Theme", variant="secondary")
                    gen_bible_btn = gr.Button("Generate World Bible (uses LLM dropdown)", variant="stop")
                with gr.Row():
                    theme_status = gr.Markdown()
                    wb_status = gr.Markdown()
                
                # CHANGE: Show what the LLM actually sees - the exact prompt we send
                def get_llm_context_view():
                    """Show only what the LLM actually receives (no debug extras)."""
                    try:
                        # Build the same prompt the LLM would see
                        sample_input = "[Next player input will go here]"
                        full_prompt = build_user_prompt(state_mgr, sample_input)
                        
                        # Split into sections for clarity
                        sections = full_prompt.split("\n---\n")
                        game_state_json = sections[0].replace("Game State (read-only; update via JSON directives only):\n", "") if sections else ""
                        
                        # Parse the JSON and world context separately for better display
                        import json
                        try:
                            # Extract JSON part
                            json_end = game_state_json.find("\nWorld context:")
                            if json_end > 0:
                                json_part = game_state_json[:json_end]
                                context_part = game_state_json[json_end:]
                            else:
                                json_part = game_state_json
                                context_part = ""
                            
                            parsed_state = json.loads(json_part)
                        except:
                            parsed_state = {"parse_error": "Could not parse game state"}
                            context_part = game_state_json
                        
                        # CHANGE (TRIVIAL): Show ONLY what is sent to the LLM, with no wrapper key
                        return {
                            "game_state": parsed_state,  # CHANGE: this is exactly the JSON block sent
                            "world_bible_context": context_part.strip() if context_part else "(No world bible context)"  # CHANGE: this is appended verbatim
                        }
                    except Exception as e:
                        return {"error": f"Context view error: {e}"}
                
                # CHANGE: Show LLM Context instead of redundant dashboard
                with gr.Accordion("LLM Context (What the AI Sees)", open=False):
                    world_bible_display = gr.JSON(label="🤖 Exact Context Sent to LLM", value=get_llm_context_view())
                
                # CHANGE: Add separate World Bible view for debugging game structure
                with gr.Accordion("World Bible (Static Game Design)", open=False):
                    def get_world_bible_view():
                        """Display the current world bible for debugging"""
                        try:
                            wb = WORLD_BIBLE if 'WORLD_BIBLE' in globals() and WORLD_BIBLE else None
                            if not wb:
                                return {"status": "No World Bible generated yet. Click 'Generate World Bible' to create one."}
                            
                            # Organize for readability
                            return {
                                "theme": wb.get("global_theme", wb.get("theme", "Not set")),
                                "objectives": wb.get("objectives", []),
                                "npcs": wb.get("npcs", wb.get("key_characters", [])),
                                "monsters": wb.get("monsters", []),
                                "key_items": wb.get("key_items", []),
                                "locations": wb.get("locations", []),
                                "riddles": wb.get("riddles", wb.get("riddles_and_tasks", [])),
                                # CHANGE (TRIVIAL): Show mechanics defined in the world bible
                                "mechanics": wb.get("mechanics", []),
                                "progression_hints": wb.get("progression_hints", []),
                                "note": "This is STATIC - generated once. The turn LLM uses this for consistency but doesn't modify it."
                            }
                        except Exception as e:
                            return {"error": f"World Bible view error: {e}"}
                    
                    world_bible_view = gr.JSON(label="📚 World Bible (Game Design Document)", value=get_world_bible_view())
                
                # CHANGE: Add GameState view to see all dynamic game data
                with gr.Accordion("GameState (Dynamic Game Memory)", open=False):
                    def get_game_state_view():
                        """Display the complete dynamic game state"""
                        try:
                            state = state_mgr.state
                            return {
                                "player": {
                                    "name": state.player_name,
                                    "location": state.location,
                                    "health": state.health,
                                    "inventory": state.inventory
                                },
                                "world": {
                                    "known_map": state.known_map,
                                    "room_items": state.room_items,
                                    "notes": state.notes
                                },
                                "conversation": {
                                    "recent_history": state.recent_history,
                                    "story_context": state.story_context
                                },
                                # CHANGE (TRIVIAL): Expose LLM-defined flags/mechanics for debugging
                                "game_flags": state.game_flags,
                                "images": {
                                    "rooms_with_images": state.rooms_with_images,
                                    "items_with_images": state.items_with_images,
                                    "total_generated": state.images_generated,
                                    "total_reused": state.images_reused,
                                    "last_images_count": len(state.last_images)
                                },
                                "note": "This is DYNAMIC - changes every turn as you play"
                            }
                        except Exception as e:
                            return {"error": f"GameState view error: {e}"}
                    
                    game_state_view = gr.JSON(label="🎮 GameState (Current Playthrough)", value=get_game_state_view())
                
                # Save/Load buttons moved to first column below LLM controls
        
        # Image generation controls - radio buttons with small generate/regenerate button
        with gr.Row():
            pregenerate_level = gr.Radio(["some", "most", "all"], value="some", label="Image Coverage")
            # Set initial button text based on whether images already exist
            initial_button_text = "Regenerate" if (ui_data.get("images") or state_mgr.state.last_images) else "Generate Images"
            generate_images_btn = gr.Button(initial_button_text, variant="secondary", size="sm")
            generate_images_status = gr.Markdown()
 
        with gr.Accordion("Debug Console", open=False):
            # CHANGE: Use JSON like other windows for consistency and copy functionality
            def get_debug_view():
                """Format debug logs as structured data"""
                return {"debug_logs": ui_data.get("debug", [])}
            
            debug_md = gr.JSON(label="Debug Logs", value=get_debug_view())
 
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
        def do_reload_llm_with_color(ollama_model: str, max_tokens: int):
            result = do_reload_llm(ollama_model, max_tokens)
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
                ui_data["narration"] = (opening_text + "\n\nType an action and press Send. Use the controls below to reload the LLM or diffuser.\n")
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
        def do_pregenerate(level: str):
            # Pre-generate images for speed: some/most/all
            global WORLD_BIBLE
            try:
                # Determine targets
                rooms = list(state_mgr.state.known_map.keys())
                items = sorted({i for v in state_mgr.state.room_items.values() for i in v})
                # CHANGE: include World Bible rooms and key items so later gameplay reuses images
                try:
                    wb = WORLD_BIBLE or {}
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
                    ui_data["debug"].append(f"[tool] Models: LLM={getattr(llm, 'model_id', 'None')}, Diffuser={(getattr(image_gen, 'model_id', None) or 'None')}")
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
    global USE_DIFFUSION_SERVER  # Ensure we can modify the global variable

    ensure_directories_exist()

    parser = argparse.ArgumentParser(description="Minimal LLM-centered text adventure with Z-Image")
    parser.add_argument("--player", type=str, default="Adventurer", help="Player name")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name (e.g., llama3.1:8b)")
    parser.add_argument("--max_tokens", type=int, default=600, help="Max new tokens per turn")
    parser.add_argument("--no-server", action="store_true", help="Skip diffusion server startup, use local generation only")
    parser.add_argument("--no-images", action="store_true", help="Disable image generation entirely")
    parser.add_argument("--cli", action="store_true", help="Run in command-line mode instead of Gradio UI")
    args = parser.parse_args(argv)

    state_mgr = StateManager(GameState(player_name=args.player))

    # No interactive console prompts; UI handles model selection/reload
    model_id = args.model or ""

    # Local ImageGenerator for image generation
    if args.no_images:
        image_gen = None
        print("Images: disabled (--no-images flag)")
    else:
        image_gen = ImageGenerator(model_id="Z-Image-Turbo")
        print("Images: enabled via Z-Image-Turbo (local model)")

    if model_id:
        print(f"Using LLM (Ollama): {model_id}")
    else:
        print("Using LLM (Ollama): (select in UI)")
    print("Images: enabled via Z-Image-Turbo (local model)")

    llm = LLMEngine(model_id=model_id, max_new_tokens=args.max_tokens, temperature=0.7)

    # World bible is loaded from .pkl files now, not separate files
    global WORLD_BIBLE
    WORLD_BIBLE = None
    print("\nWelcome to the Minimal LLM Adventure!")
    print("The world is driven by your chosen LLM. Images use Z-Image-Turbo.")

    if args.cli:
        # Run in command-line mode
        interactive_loop(state_mgr, llm, image_gen)
        return 0
    else:
        # Launch the Gradio UI (you can still use reloads to change LLMs)
        launch_gradio_ui(state_mgr, llm, image_gen)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())