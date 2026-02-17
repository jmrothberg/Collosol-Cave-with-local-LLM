# JMR's LLM Adventure Game Engine
# August 19, 2025
#
# This is a fully LLM-driven adventure game where the Large Language Model
# acts as the game master, storyteller, and world engine. The LLM has complete
# creative control while using structured JSON to maintain consistent game state.
#
# The LLM is instructed via system prompt to use these JSON tools, making this
# a powerful general-purpose adventure engine that can run ANY adventure the
# LLM can imagine, while maintaining proper game state through function calls.

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
from diffusers import DiffusionPipeline, FluxPipeline, StableDiffusion3Pipeline

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

# CHANGE: default image theme used only when no Theme is applied in the UI
DEFAULT_IMAGE_THEME = (
    "Studio Ghibli background art, dark fantasy cavern, painterly matte painting, "
    "soft rim light, volumetric fog, by Studio Ghibli"
)


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


def execute_world_bible_generation(state_mgr: "StateManager", model_id: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
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
    result = generate_world_bible(state_mgr, heavy_model_id)
    
    if result:
        WORLD_BIBLE = result
        return True, "World bible generated successfully.", WORLD_BIBLE
    else:
        return False, "Failed to generate world bible.", None


def generate_themed_image(image_gen: "ImageGenerator", base_prompt: str, theme_suffix: Optional[str] = None) -> Optional[str]:
    """
    Generate an image with theme applied consistently.
    Returns the generated image path or None if generation fails.
    """
    if not image_gen:
        return None
    
    if not theme_suffix:
        theme_suffix = get_theme_suffix()
    
    # Apply theme to prompt
    if theme_suffix and theme_suffix.lower() not in base_prompt.lower():
        themed_prompt = f"{base_prompt}. World art style: {theme_suffix}. Strictly adhere to this style."
    else:
        themed_prompt = base_prompt
    
    return image_gen.generate(themed_prompt)


def generate_room_image_if_needed(state_mgr: "StateManager", image_gen: Optional["ImageGenerator"], room_name: str, debug_lines: Optional[List[str]] = None) -> Optional[str]:
    """
    Generate a room image only if one doesn't already exist.
    Returns the image path (new or existing) or None.
    """
    if not image_gen:
        return None
    
    # Check if room already has an image
    if state_mgr.state.has_room_image(room_name):
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
    
    # Generate new room image
    base_prompt = f"atmospheric fantasy {room_name.lower()}, detailed environment, adventure game location"
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
    # CHANGE: Deduplicate file paths while preserving order to avoid duplicate thumbnails
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
        # CHANGE: add world bible reference if available for easier debugging/consistency
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
        # CHANGE: add world bible reference if available for easier debugging/consistency
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

    def describe_map(self) -> str:
        lines: List[str] = []
        for room, info in self.state.known_map.items():
            exits_str = ", ".join(info.get("exits", [])) or "None"
            notes = info.get("notes", "")
            note_str = f" | notes: {notes}" if notes else ""
            # CHANGE (TRIVIAL): Do not list items in the Known Map; items are not part of map display
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


class ImageGenerator:
    def __init__(self, model_id: Optional[str] = None, device_preference: Optional[str] = None, local_root: Optional[str] = None) -> None:
        self.model_id = model_id
        self.device = self._resolve_device(device_preference)
        self.local_root = local_root
        self._pipeline = None  # lazy init
        self._dtype = self._default_dtype()
    
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

    def _default_dtype(self) -> torch.dtype:
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
        # Default path similar to Colossal Cave
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
        torch.set_default_dtype(torch.float32)
        original_from_numpy = torch.from_numpy
        def mps_safe_from_numpy(ndarray):
            tensor = original_from_numpy(ndarray)
            if tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor
        torch.from_numpy = mps_safe_from_numpy  # type: ignore
        original_arange = torch.arange
        def mps_safe_arange(*args, **kwargs):
            if 'dtype' in kwargs and kwargs['dtype'] == torch.float64:
                kwargs['dtype'] = torch.float32
            return original_arange(*args, **kwargs)
        torch.arange = mps_safe_arange  # type: ignore

    def _lazy_init(self) -> None:
        if self._pipeline is not None or not self.model_id:
            return
        model_local_path = self._local_model_path()
        try:
            if model_local_path:
                # Prefer local pipeline types matching model
                base_name_upper = os.path.basename(model_local_path).upper()
                print(f"[image] Loading local diffusion model from: {model_local_path}")
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
                    # CHANGE: Use device_map="balanced" on CUDA and avoid .to();
                    # keep explicit .to() for MPS. This preserves CUDA sharding and fixes Qwen-Image.
                    if self.device.startswith("cuda"):
                        self._pipeline = DiffusionPipeline.from_pretrained(
                            model_local_path,
                            torch_dtype=self._dtype,
                            device_map="balanced",
                        )
                        # IMPORTANT: do not call .to() after device_map
                    else:
                        self._pipeline = DiffusionPipeline.from_pretrained(
                            model_local_path,
                            torch_dtype=self._dtype,
                        )
                        if self.device == "mps":
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

            # Generic fallback via HF id - auto-enable HF downloads for known models
            hf_models = curated_hf_diffusers()
            auto_allow = any(self.model_id.startswith(prefix) for prefix in ["black-forest-labs/", "stabilityai/", "runwayml/"])
            
            if auto_allow or os.environ.get("ALLOW_HF_DOWNLOAD", "0").lower() in {"1", "true", "yes"}:
                if auto_allow:
                    print(f"[image] Auto-enabling HF download for recognized model: {self.model_id}")
                    # Temporarily set the environment variable for this session
                    os.environ["ALLOW_HF_DOWNLOAD"] = "1"
                # CHANGE: On CUDA use device_map="balanced" and skip .to(); on MPS use explicit .to()
                if self.device.startswith("cuda"):
                    self._pipeline = DiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=self._dtype,
                        device_map="balanced",
                    )
                    # IMPORTANT: do not call .to() after device_map
                else:
                    self._pipeline = DiffusionPipeline.from_pretrained(
                        self.model_id,
                        torch_dtype=self._dtype,
                    )
                    if self.device == "mps":
                        self._pipeline = self._pipeline.to(self.device)
            else:
                print(f"[image] Skipping download for '{self.model_id}'. Set ALLOW_HF_DOWNLOAD=1 to enable.")
                self._pipeline = None
        except Exception as e:
            print(f"[image] Failed to initialize pipeline '{self.model_id}': {e}")
            
            # If FLUX failed due to memory, try a lighter model
            if "out of memory" in str(e).lower() and "flux" in self.model_id.lower():
                print("[image] FLUX failed due to memory. Trying lighter SDXL-turbo fallback...")
                try:
                    # Try SDXL-turbo as fallback (much lighter)
                    fallback_model = "stabilityai/sdxl-turbo"
                    self._pipeline = DiffusionPipeline.from_pretrained(
                        fallback_model,
                        torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
                        variant="fp16" if self.device.startswith("cuda") else None,
                        use_safetensors=True,
                    )
                    if self.device.startswith("cuda"):
                        # Use the selected GPU
                        self._pipeline = self._pipeline.to(self.device)
                    print(f"[image] Successfully loaded fallback model: {fallback_model}")
                except Exception as e2:
                    print(f"[image] Fallback also failed: {e2}")
                    self._pipeline = None
            else:
                self._pipeline = None

    def generate(self, prompt: str) -> Optional[str]:
        if not self.model_id:
            print("[image] Image generation disabled (no model selected).")
            return None
        self._lazy_init()
        if self._pipeline is None:
            return None
        try:
            if isinstance(self._pipeline, FluxPipeline):
                image = self._pipeline(
                    prompt,
                    num_inference_steps=4,
                    guidance_scale=0.0,
                ).images[0]
            else:
                # CHANGE: default to 10 steps for non-FLUX pipelines (e.g., Qwen-Image, SD)
                image = self._pipeline(prompt, num_inference_steps=10).images[0]
            ts = datetime.now().strftime("%m%d-%H%M%S")
            safe = prompt[:24].strip().replace(" ", "_").replace("/", "_")
            out_name = f"{safe}_{ts}.png" if safe else f"img_{ts}.png"
            out_path = os.path.join(ART_DIR, out_name)
            image.save(out_path)
            # Also copy into in-game images to persist with saves
            try:
                ig_path = os.path.join(IN_GAME_IMG_DIR, out_name)
                shutil.copyfile(out_path, ig_path)
            except Exception as e:
                print(f"[image] copy to in-game images failed: {e}")
            return out_path
        except Exception as e:
            print(f"[image] Generation error: {e}")
            return None


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


def generate_world_bible(state_mgr: StateManager, heavy_model_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    SIMPLE world bible generation - just call LLM and return result.
    If anything goes wrong, return a default plan. NO COMPLEX NESTING!
    """
    # Default plan that always works - showing interconnected gameplay
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
    
    # If no model, just return default
    if not heavy_model_id:
        print("[world_bible] No model provided, using default plan")
        return default_plan
    
    try:
        # Simple prompt
        snapshot = {
            "location": state_mgr.state.location,
            "rooms": list(state_mgr.state.known_map.keys())[:5]
        }
        
        # ENFORCED THEME from UI (if any): we do NOT hard-wire.
        # If user applied a theme via the Theme input, we carry it forward here.
        try:
            enforced_theme = ((WORLD_BIBLE or {}).get("global_theme"))
        except Exception:
            enforced_theme = None
        if not enforced_theme:
            enforced_theme = DEFAULT_IMAGE_THEME
        
        prompt = f"""Create a COMPLETE, SOLVABLE adventure game outline for: {snapshot}

IMPORTANT: Design an interconnected game where:
- Items are needed to solve puzzles/defeat monsters
- NPCs provide crucial hints or items
- Riddles unlock access to new areas or items
- Everything connects to achieve the objectives
- The game must be winnable with logical progression



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

Respond with JSON only."""

        # Call LLM
        print(f"[world_bible] Calling {heavy_model_id}")
        response = ollama.generate(
            model=heavy_model_id,
            prompt=prompt,
            options={"temperature": 0.7, "num_predict": 3000}
        )
        
        # Extract JSON from response
        text = response.get("response", "")
        
        # Remove markdown if present
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        
        if start >= 0 and end > start:
            json_str = text[start:end]
            plan = json.loads(json_str)
            
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
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"  - {issue}")
                print(f"[world_bible] Using enhanced default plan instead")
                return default_plan
            
    except Exception as e:
        print(f"[world_bible] Generation failed ({e}), using default")
    
    return default_plan


    # Minimal snapshot of current seed for better planning
    snapshot = {
        "start": state_mgr.state.location,
        "known_map_keys": list(state_mgr.state.known_map.keys())[:20],
        "inventory": list(state_mgr.state.inventory)[:10],
    }

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
    
    # Helper function to extract complete JSON objects with balanced braces
    def extract_json_from_text(text):
        start_idx = text.find('{')
        if start_idx == -1:
            return []
        
        candidates = []
        brace_count = 0
        json_start = start_idx
        
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    # Found complete JSON object
                    candidates.append(text[json_start:i+1])
                    # Look for next JSON object
                    remaining = text[i+1:]
                    next_start = remaining.find('{')
                    if next_start != -1:
                        json_start = i + 1 + next_start
                        brace_count = 0
                    else:
                        break
        
        return candidates

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
                    "num_predict": 4000,  # Balanced - detailed enough for images but not too long
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
            
            candidates = extract_json_from_text(cleaned_text)
            for i, c in enumerate(candidates):
                try:
                    plan = json.loads(c)
                    print(f"[world_bible] Successfully parsed JSON candidate {i}")
                    break
                except Exception as parse_err:
                    print(f"[world_bible] Failed to parse candidate {i}: {parse_err}")
                    continue
            
            if plan is None:
                # Try one more time with the original text
                candidates = extract_json_from_text(text)
                for i, c in enumerate(candidates):
                    try:
                        plan = json.loads(c)
                        print(f"[world_bible] Successfully parsed JSON candidate {i} (from original)")
                        break
                    except:
                        continue
            
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
                candidates = extract_json_from_text(text)
                print(f"[world_bible] Found {len(candidates)} JSON-like candidates")
                for i, c in enumerate(candidates):
                    try:
                        plan = json.loads(c)
                        print(f"[world_bible] Successfully parsed candidate {i}")
                        break
                    except Exception as parse_err:
                        print(f"[world_bible] Failed to parse candidate {i}: {parse_err}")
                        continue
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


SYSTEM_INSTRUCTIONS = (
    "You are the narrator for a text adventure. Keep responses 2-5 sentences.\n\n"
    "STORYTELLING PRINCIPLES:\n"
    "- Create tension: foreshadow dangers, build to climaxes\n"
    "- Reward exploration: hide secrets, reward careful examination\n"
    "- Fail forward: setbacks should create new opportunities, not dead ends\n"
    "- Remember earlier events: reference past actions when relevant\n"
    "- Multiple solutions: consider 2+ ways to overcome obstacles\n"
    "- Use story_context to track narrative threads and intentions\n\n"
    "WORLD BIBLE RULES:\n"
    "- Follow the world bible for NPCs, monsters, items, and objectives\n"
    "- Monster weaknesses are ABSOLUTE (torch defeats troll, sword defeats golem)\n"
    "- NPCs remember last 5 turns (check recent_conversation)\n\n"
    "GOAL ALIGNMENT:\n"
    "- Consider win_condition in context; prefer actions that move the player toward it when reasonable.\n\n"
    "MOVEMENT RULES:\n"
    "- When entering NEW rooms from world bible, ALWAYS connect them:\n"
    "  {\"move_to\": \"Underground River\", \"connect\": [[\"Dark Passage\", \"Underground River\"]]}\n"
    "- Check known_map to avoid dead ends\n\n"
    "COMBAT:\n"
    "- Use change_health for damage (negative) or healing (positive)\n"
    "- Respect monster weaknesses from world bible\n\n"
    "JSON TOOLS (append after narration):\n"
    "- move_to: string room name\n"
    "- connect: [[\"RoomA\", \"RoomB\"]] pairs for bidirectional exits\n"
    "- add_items: [\"item\"] to inventory\n"
    "- remove_items: [\"item\"] from inventory\n"
    "- change_health: integer (-5 for damage, +5 for heal)\n"
    "- add_note: \"quest update\"\n"
    "- set_context: \"narrative memory/intentions\" (track story threads)\n"
    "- set_flag: {\"name\": \"flag_name\", \"value\": any} (track puzzles/states/timers)\n"
    "- mechanics: {\"action\": \"block door with stone\", \"effect\": \"door stays open\", \"permanent\": true}\n"
    "- place_items: [\"item\"] in current room\n"
    "- room_take: [\"item\"] from room to inventory\n"
    "- images: [\"room overview\", \"item close-up\"]\n\n"
    "EXAMPLE JSON:\n"
    "```json\n"
    "{\n"
    "  \"state_updates\": {\n"
    "    \"move_to\": \"Glowing Cavern\",\n"
    "    \"place_items\": [\"Glowing Mushroom\"],\n"
    "    \"set_context\": \"Player discovered the hidden cavern. The mushroom's glow hints at deeper magic.\"\n"
    "  },\n"
    "  \"images\": [\"Glowing Cavern overview\", \"Glowing Mushroom close-up\"]\n"
    "}\n"
    "```\n\n"
    "CRITICAL ITEM RULES:\n"
    "- ANY item you mention MUST be in place_items (unless already in room)\n"
    "- To pick up items: use room_take (moves from room to inventory)\n"
    "- NEVER mention items that aren't actually there\n"
    "- If the player tries to take X, include room_take for X only if X is in the current room; otherwise, say where it is and how to reach it.\n"
    "- When you place an item, remember where you put it. Use set_context to track item locations if needed.\n"
    "- When inventing new mechanics, items, or behaviors, record them in set_flag (for data) and set_context (for narrative memory) so future turns can reference them.\n"
    "- For simple cause-effect rules (e.g., 'stone blocks door'), use mechanics: {\"action\": \"what player did\", \"effect\": \"what happens\"}.\n"
    "- New mechanics should: enhance gameplay, align with objectives, be consistent with the world's logic, and create meaningful choices (not arbitrary obstacles).\n\n"
    "IMAGE RULES:\n"
    "- Check rooms_with_images and items_with_images in context\n"
    "- If player asks to see/examine an item: include its name in images array (e.g., 'Ancient Map')\n"
    "- If item already has an image, just use the item name - system will reuse cached image\n"
    "- Only include a ROOM image if a move_to actually happens this turn (or it's a brand new room without an image)\n"
    "- Do NOT include images for rooms that were only discussed but not moved to\n"
    "- The LAST image in array is what displays to player\n"
    "- When requesting new room images: describe what makes it unique (e.g., 'dark cave with glowing crystals and underground stream')\n"
    "- Keep prompts atmospheric and relevant to the narrative\n"
    "- For images: use exact room/item names with spaces (e.g., 'Echo Chamber overview' not 'Echo_Chamber_overview').\n\n"
    "JSON PLACEMENT: Always append JSON AFTER narration, never inside it."
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
        "Start a new adventure. Provide an opening scene (2-5 sentences). "
        "Set an initial location and minimal map with sensible exits. If helpful, place a useful starting item "
        "(e.g., Lantern, Rope, Map). If the scene mentions a notable room or object, request 1 short image prompt.\n\n"
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
    
    # ENSURE opening room image: If no images were generated, inject a move_to to trigger auto-generation
    if not new_images and state_mgr.state.location and image_gen:
        # Re-process with a synthetic move_to to trigger image generation for starting location
        synthetic_json = {"state_updates": {"move_to": state_mgr.state.location}}
        _, start_images, _, _ = apply_llm_directives(state_mgr, f"{final_text}\n```json\n{json.dumps(synthetic_json)}\n```", image_gen)
        new_images.extend(start_images)
    
    return final_text or "Your journey begins...", new_images


def apply_llm_directives(state_mgr: StateManager, text: str, image_gen: Optional[ImageGenerator]) -> Tuple[str, List[str], List[Dict[str, Any]], List[str]]:
    image_paths: List[str] = []
    cleaned_text = text
    debug_lines: List[str] = []
    json_payloads: List[Dict[str, Any]] = []

    # AGGRESSIVE JSON REMOVAL STRATEGY
    # We'll find ANY text that looks like JSON and remove it
    
    # 1) Extract fenced code blocks first
    fenced_pat = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
    fenced_blocks = []
    for m in fenced_pat.finditer(text):
        fenced_blocks.append((m.start(), m.end()))
        block = m.group(1).strip()
        try:
            directives = json.loads(block)
            if isinstance(directives, dict):
                json_payloads.append(directives)
                debug_lines.append(f"Extracted JSON from fenced block")
        except Exception as e:
            debug_lines.append(f"Failed to parse fenced JSON: {str(e)[:50]}")
    
    # Remove all fenced blocks
    if fenced_blocks:
        fenced_blocks.sort(reverse=True)  # Remove from end to start
        for start, end in fenced_blocks:
            text = text[:start] + text[end:]
    
    # 2) Find ALL JSON-like structures (anything starting with { and containing our keys)
    # This includes complete, incomplete, and malformed JSON
    recognized_keys = ["state_updates", "images"]
    
    # Pattern to find any { that might start JSON with our keys
    # This will match from { to end of text/line if JSON is incomplete
    json_like_pattern = re.compile(
        r'\{[^}]*?"(?:' + '|'.join(recognized_keys) + r')"[^}]*(?:\}[\s\S]*?\}|\}|[\s\S]*?$)',
        re.MULTILINE | re.DOTALL
    )
    
    json_spans = []
    for match in json_like_pattern.finditer(text):
        json_spans.append((match.start(), match.end()))
        json_text = match.group(0)
        
        # Try to extract valid JSON by finding balanced braces
        brace_count = 0
        in_string = False
        escape_next = False
        valid_end = -1
        
        for i, char in enumerate(json_text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        valid_end = i + 1
                        break
        
        # If we found valid JSON, try to parse it
        if valid_end > 0:
            try:
                directives = json.loads(json_text[:valid_end])
                if isinstance(directives, dict):
                    json_payloads.append(directives)
                    debug_lines.append(f"Extracted valid JSON object")
            except:
                debug_lines.append(f"Found JSON-like structure but couldn't parse")
        else:
            debug_lines.append(f"Found incomplete JSON fragment")
    
    # Remove all JSON-like structures
    if json_spans:
        json_spans.sort(reverse=True)  # Remove from end to start
        for start, end in json_spans:
            text = text[:start] + text[end:]
    
    # 3) Final cleanup: Remove any remaining fragments that look like JSON
    # This catches any { ... that wasn't caught above
    text = re.sub(r'\{\s*"(?:state_updates|images)"[^}]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\{\s*"(?:state_updates|images)"[^\n]*', '', text)
    
    # Remove incomplete JSON arrays and values
    text = re.sub(r',\s*"images":\s*\[\s*"[^"]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'"images":\s*\[\s*"[^"]*$', '', text, flags=re.MULTILINE)
    
    # Remove trailing commas and incomplete JSON structures
    text = re.sub(r',\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*,\s*"images":\s*$', '', text, flags=re.MULTILINE)
    
    # Also remove any standalone braces or JSON punctuation at line ends
    text = re.sub(r'\s*\{\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\[\s*"[^"]*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\]\s*,?\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s*\}\s*,?\s*$', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    cleaned_text = text

    # 3) Apply directives
    last_room_moved_to = None  # Track room moves for image display
    image_paths = []  # Initialize image paths list
    
    for directives in json_payloads:
        try:
            updates = directives.get("state_updates", {}) if isinstance(directives, dict) else {}
            # Map updates
            if isinstance(updates.get("move_to"), str):
                state_mgr.move_to(updates["move_to"]) 
                debug_lines.append(f"tool.move_to: {updates['move_to']}")
                last_room_moved_to = updates["move_to"]  # Track the move
            if isinstance(updates.get("connect"), list):
                for pair in updates["connect"]:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        state_mgr.connect_rooms(str(pair[0]), str(pair[1]))
                        debug_lines.append(f"tool.connect_rooms: {pair[0]} <-> {pair[1]}")
            # Inventory updates
            for item in updates.get("add_items", []) or []:
                state_mgr.add_item(str(item))
                debug_lines.append(f"tool.add_item: {item}")
            for item in updates.get("remove_items", []) or []:
                state_mgr.remove_item(str(item))
                debug_lines.append(f"tool.remove_item: {item}")
            # Room item placement/take
            for item in updates.get("place_items", []) or []:
                state_mgr.place_item_in_room(str(item))
                debug_lines.append(f"tool.place_item_in_room: {item} @ {state_mgr.state.location}")
            for item in updates.get("room_take", []) or []:
                taken = state_mgr.remove_item_from_room(str(item))
                if taken:
                    state_mgr.add_item(str(item))
                    debug_lines.append(f"tool.room_take: {item}")
                else:
                    debug_lines.append(f"tool.room_take: {item} (FAILED - item not in room)")
                    # Don't auto-correct - let LLM learn from its mistakes
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
            
            # CHANGE (TRIVIAL): Game mechanics - simpler way for weak LLMs to record cause-effect rules
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

            # Images - both LLM-requested and auto-generated for new rooms and items
            img_prompts = directives.get("images", []) if isinstance(directives, dict) else []
            # CHANGE: Track what we've already added to avoid duplicates
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
                    # CHANGE: Only add if LLM hasn't already requested something similar
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
                    # CHANGE: Only auto-gen if LLM hasn't already requested it
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
                    # CHANGE: Only auto-gen if LLM hasn't already requested it
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
    # Short curated list; selectable alongside locals. Downloads happen only if ALLOW_HF_DOWNLOAD=1.
    return [
        "black-forest-labs/FLUX.1-schnell",
        "stabilityai/sdxl-turbo",
    ]


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
    # OPTIMIZATION: Don't include full image prompts - just counts for efficiency
    
    context = {
        "player_name": state.player_name,
        "location": state.location,
        "health": state.health,
        "inventory": state.inventory,
        "known_map": state.known_map,
        "notes": state.notes,
        # Recent conversation history for NPC continuity (all stored turns)
        "recent_conversation": state.recent_history,
        # Story context - LLM's narrative memory
        "story_context": state.story_context,
        # Game flags - LLM-defined states
        "game_flags": state.game_flags,
        # Added LLM aids (read-only)
        "current_room_items": current_items,
        "current_exits": current_exits,
        "visited_rooms": visited_rooms,
        # CHANGE (TRIVIAL): Include game theme every turn for consistent narration & image prompts
        "world_theme": get_theme_suffix(),
        # CHANGE (TRIVIAL): Include overall win condition each turn to keep decisions aligned
        "win_condition": (WORLD_BIBLE.get("win_condition") if 'WORLD_BIBLE' in globals() and WORLD_BIBLE else None),
        # IMAGE REUSE INFO - simple lists for fast LLM processing
        "rooms_with_images": state.rooms_with_images,
        "items_with_images": state.items_with_images,
        # DEBUG: Detailed image mapping for debugging
        "image_debug": {
            "room_images": {img["subject"]: os.path.basename(img["path"]) for img in state.last_images if img.get("type") == "room"},
            "item_images": {img["subject"]: os.path.basename(img["path"]) for img in state.last_images if img.get("type") == "item"},
            "total_images": len(state.last_images),
            "images_reused": state.images_reused
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
            # design intent. As LLMs improve, these cues naturally become more useful.
            #
            # Cue sources:
            # - NPCs: only those in the current room, with personality and what they provide
            # - Monsters: only those in the current room, with difficulty and known weakness
            # - Riddles: only those in the current room, with hint and reward
            # - Item availability: match item_locations entries mentioning current room
            # - Current objective: chosen from objectives based on simple inventory progress
            cues = []
            
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


def do_generate_world_bible(wb_model: str) -> str:
    """Standalone world bible generation for command line usage."""
    success, message, _ = execute_world_bible_generation(StateManager(), wb_model)
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
                get_debug_view(),  # CHANGE: return structured JSON for gr.JSON
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
                get_debug_view(),  # CHANGE: return structured JSON for gr.JSON
                str(state_mgr.state.health),
                ", ".join(state_mgr.state.inventory) or "(empty)",
                state_mgr.describe_map(),
                None,
                get_llm_context_view(),
                state_mgr.state.location,
            )
        system_prompt = SYSTEM_INSTRUCTIONS
        user_prompt = build_user_prompt(state_mgr, user_text)
        llm_text = llm.generate(system_prompt, user_prompt)
        
        # Add raw LLM response to debug only
        ui_data["debug"].append(f"[llm_raw] {llm_text}")

        # Track how many images existed before this turn to identify new ones
        _prev_image_count = len(state_mgr.state.last_images)

        final_text, new_images, payloads, tool_logs = apply_llm_directives(state_mgr, llm_text, image_gen)
        
        # Save conversation history (keep last 5 turns for context)
        # CHANGE: Include tool execution feedback so LLM knows what worked/failed
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
        # CHANGE (TRIVIAL): Log memory window status for LLM-first debugging (no logic change)
        try:
            ui_data["debug"].append(
                f"[memory] recent={len(state_mgr.state.recent_history)} story_context={'yes' if (state_mgr.state.story_context or '').strip() else 'no'}"
            )
        except Exception:
            pass
        
        # Add JSON and tool logs to debug console only
        # CHANGE: Don't duplicate - payloads is already the parsed JSON
        if payloads:
            # Just log that we have JSON, not the full content (it's in tool logs)
            ui_data["debug"].append(f"[json] processed {len(payloads)} directive(s)")
        else:
            ui_data["debug"].append("[json] none")
            
        if tool_logs:
            for line in tool_logs:
                ui_data["debug"].append(f"[tool] {line}")
        if new_images:
            # CHANGE (DEBUG ONLY): list basenames for consistency with gallery filename
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

    def do_reload_llm(ollama_model: str):
        # Reload just the LLM
        nonlocal llm
        # CHANGE: skip reload if model unchanged to avoid no-op reinitialization
        if ollama_model and getattr(llm, "model_id", None) == ollama_model:
            _append_debug("[reload] LLM unchanged; skipping")
            return f"✅ LLM unchanged: {ollama_model}"
        if ollama_model:
            # Clean up old LLM if needed (Ollama manages its own memory)
            # CHANGE (TRIVIAL): allow a bit longer replies to reduce rare truncated JSON
            llm = LLMEngine(model_id=ollama_model, max_new_tokens=2500, temperature=0.8)
            _append_debug(f"[reload] LLM -> {ollama_model}")
            return f"✅ LLM loaded: {ollama_model}"
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
            # Let ImageGenerator auto-select best GPU for multi-GPU systems
            image_gen = ImageGenerator(model_id=diffuser_id, device_preference=None, local_root=diffusion_models_root())
            _append_debug(f"[reload] Diffuser -> {diffuser_id}")
            return f"✅ Diffuser loaded: {diffuser_id}"
        else:
            image_gen = None
            _append_debug("[reload] Diffuser -> disabled")
            return "✅ Diffuser disabled"


    def do_generate_world_bible(selected_model: str):
        """Generate world bible using the model string passed from the dropdown."""
        model_used = (selected_model or "").strip()
        if not model_used:
            # Leave button appearance unchanged on failure
            # CHANGE: Always return structured JSON to the JSON component (avoid None)
            return "❌ Select an LLM first from the dropdown, then click Generate World Bible.", get_world_bible_view(), gr.update()

        # Autosave current run before generating a new world (non-destructive)
        _autosave_if_active()
        
        success, message, world_bible_data = execute_world_bible_generation(state_mgr, model_used)
        
        if success and world_bible_data:
            # Add to debug console for troubleshooting
            ui_data["debug"].append(f"[world_bible] Generated successfully:")
            ui_data["debug"].append(f"[world_bible] {json.dumps(world_bible_data, ensure_ascii=False, indent=2)}")
            # CHANGE (TRIVIAL): Clear current game state and UI after generating a new world bible
            try:
                state_mgr.state = GameState(player_name=state_mgr.state.player_name)
                ui_data["narration"] = ""
                ui_data["images"] = []
                ui_data["debug"].append("[world_bible] Cleared current game state for new world")
            except Exception:
                pass
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
            return f"✅ {message}", formatted_world_bible, gr.update(variant="secondary")
        else:
            # Keep standout variant to indicate action still needed / failed
            # CHANGE: Always return structured JSON to the JSON component (avoid None)
            return f"❌ {message}", get_world_bible_view(), gr.update()
 
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
                latest_image = gr.Image(label="Latest Image", height=300, value=(get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr) or (ui_data["images"][-1] if ui_data["images"] else None)))
                # CHANGE: Removed secondary latest-turn gallery per request; one latest image panel only
                # CHANGE: Wrap main Image Gallery in a collapsible accordion for easy hiding
                with gr.Accordion("Image Gallery", open=False):
                    # CHANGE (TRIVIAL): Initialize Gallery with any opening images so they display immediately
                    gallery = gr.Gallery(label="Image Gallery", height=200, columns=3, value=ui_data["images"])
                    # SIMPLE FIX: Button to show all filenames
                    show_names_btn = gr.Button("Show Image Names", size="sm")
                    gallery_filenames = gr.Textbox(label="Gallery Image Filenames", interactive=False, lines=3)
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
                narration_box = gr.Textbox(value=ui_data["narration"], label="Narration", lines=10, max_lines=10, interactive=False)
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
                theme_input = gr.Textbox(
                    label="Global Theme (world/art style)",
                    value=(_wb_theme or ""),
                    placeholder=DEFAULT_IMAGE_THEME,
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
                
                with gr.Row():
                    # NEW: Start a fresh run using the current world bible (replay from start)
                    restart_btn = gr.Button("New Game (same World Bible)", variant="secondary")
                    restart_status = gr.Markdown()
                with gr.Row():
                    # Allow naming the game when saving
                    game_name_input = gr.Textbox(label="Save as Name", placeholder="My_Adventure")
                    save_btn = gr.Button("Save Game (.pkl)", variant="primary")
                    save_status = gr.Markdown()
                with gr.Row():
                    # Replace drop-file with a saved-games dropdown for convenience
                    def _list_saved_games_local():
                        try:
                            files = [f for f in os.listdir(SAVE_DIR) if f.startswith("Adv_") and (f.endswith('.pkl') or f.endswith('.tkl'))]
                        except Exception:
                            files = []
                        return sorted(files, reverse=True)
                    saved_choices = _list_saved_games_local() or ["(none)"]
                    load_dropdown = gr.Dropdown(saved_choices, label="Saved Games", value=(saved_choices[0] if saved_choices else "(none)"))
                    load_btn = gr.Button("Load Game", variant="secondary")
                    load_status = gr.Markdown()
        
        # Image generation controls (pre-generate vs re-generate)
        with gr.Row():
            # CHANGE (TRIVIAL): Rename to 'Generate images' and share between both actions
            pregenerate_level = gr.Radio(["some", "most", "all"], value="some", label="Generate images")
            pregenerate_btn = gr.Button("PREGENERATE", variant="secondary")
            pregenerate_status = gr.Markdown()
        
        # TRIVIAL: Add a button to regenerate images for current room and nearby items
        with gr.Row():
            regenerate_btn = gr.Button("REGENERATE", variant="secondary")
            regenerate_status = gr.Markdown()
 
        with gr.Accordion("Debug Console", open=False):
            # CHANGE: Use JSON like other windows for consistency and copy functionality
            def get_debug_view():
                """Format debug logs as structured data"""
                return {"debug_logs": ui_data.get("debug", [])}
            
            debug_md = gr.JSON(label="Debug Logs", value=get_debug_view())
 
        # SIMPLE: Show deduped gallery names, using subject names when available
        def show_all_filenames():
            try:
                # Build mapping from path -> subject (room/item) if known
                path_to_subject = {}
                for img in state_mgr.state.last_images:
                    p = img.get("path")
                    if isinstance(p, str):
                        subj = img.get("subject") or ""
                        typ = img.get("type") or ""
                        if subj:
                            # Prefer "Item: Name" or "Room: Name" labels for clarity
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
                    return "No images in gallery"
                # Render numbered list with simple names when known, else basename
                lines = []
                for idx, p in enumerate(deduped, start=1):
                    simple = path_to_subject.get(p) or os.path.basename(p)
                    lines.append(f"{idx}. {simple}")
                return "\n".join(lines)
            except Exception:
                # Safe fallback to basenames only
                paths = [p for p in ui_data.get("images", []) if isinstance(p, str)]
                if not paths:
                    return "No images in gallery"
                deduped = []
                seen = set()
                for p in paths:
                    if p not in seen:
                        seen.add(p)
                        deduped.append(p)
                return "\n".join(f"{i+1}. {os.path.basename(p)}" for i, p in enumerate(deduped))
        
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
            fn=show_all_filenames,
            outputs=[gallery_filenames],
        )
        
        # Also auto-update filenames when gallery changes
        send_btn.click(
            fn=show_all_filenames,
            outputs=[gallery_filenames],
        )
        user_box.submit(
            fn=show_all_filenames,
            outputs=[gallery_filenames],
        )
        
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
        def do_reload_llm_with_color(ollama_model: str):
            result = do_reload_llm(ollama_model)
            # Return status and update button to grey
            return result, gr.Button(variant="secondary")
        
        def do_reload_diffuser_with_color(diffuser_id: str):
            result = do_reload_diffuser(diffuser_id)
            # Return status and update button to grey
            return result, gr.Button(variant="secondary")
        
        reload_llm_btn.click(
            fn=do_reload_llm_with_color,
            inputs=[llm_drop],
            outputs=[llm_status, reload_llm_btn],
        )
        reload_diff_btn.click(
            fn=do_reload_diffuser_with_color,
            inputs=[diff_drop],
            outputs=[diff_status, reload_diff_btn],
        )
        gen_bible_btn.click(
            fn=do_generate_world_bible,
            inputs=[llm_drop],
            # CHANGE: also update the button variant after successful generation
            outputs=[wb_status, world_bible_view, gen_bible_btn],
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
                    get_debug_view(),  # CHANGE: return structured JSON for gr.JSON
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
                    return "❌ Provide a theme first"
                if WORLD_BIBLE is None:
                    WORLD_BIBLE = {"global_theme": theme_text.strip()}
                else:
                    WORLD_BIBLE["global_theme"] = theme_text.strip()
                return f"✅ Theme set: {theme_text.strip()}"
            except Exception as e:
                return f"❌ Theme update failed: {e}"
        apply_theme_btn.click(
            fn=do_apply_theme,
            inputs=[theme_input],
            outputs=[theme_status],
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
                return f"✅ Pre-generated {made} images (new: {newly_generated}, reused: {newly_reused}, level={level})", ui_data["images"], get_debug_view(), latest_img
            except Exception as e:
                return f"❌ Pre-generate failed: {e}", [], get_debug_view(), None  # CHANGE: structured JSON for gr.JSON
        pregenerate_btn.click(
            fn=do_pregenerate,
            inputs=[pregenerate_level],
            # CHANGE: Update Debug Console when pregenerating so logging is visible without sending a turn
            # CHANGE: Also refresh the Latest Image thumbnail
            outputs=[pregenerate_status, gallery, debug_md, latest_image],
        )
        
        def do_regenerate_current():
            """Force-regenerate images that already exist (no expansion of scope).
            - Replaces images for tracked rooms/items only (rooms_with_images/items_with_images)
            - Does NOT delete files; only clears tracking so fresh images are generated.
            """
            try:
                if not image_gen:
                    return "❌ No diffuser loaded", ui_data["images"], get_debug_view(), (get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr))
                # Targets: only those already tracked as having images
                rooms_sel = list(state_mgr.state.rooms_with_images)
                items_sel = list(state_mgr.state.items_with_images)
                # Helper to clear tracking for a subject
                def _clear_tracking(subject: str, typ: str) -> None:
                    try:
                        state_mgr.state.last_images = [img for img in state_mgr.state.last_images if not (img.get("type") == typ and img.get("subject") == subject)]
                        if typ == "room":
                            state_mgr.state.rooms_with_images = [r for r in state_mgr.state.rooms_with_images if r != subject]
                        elif typ == "item":
                            state_mgr.state.items_with_images = [i for i in state_mgr.state.items_with_images if i != subject]
                    except Exception:
                        pass
                # Clear and regenerate rooms
                new_room_count = 0
                first_room_img = None
                for r in rooms_sel:
                    _clear_tracking(r, "room")
                    p = generate_room_image_if_needed(state_mgr, image_gen, r, ui_data["debug"])
                    if p:
                        ui_data["images"].append(p)
                        new_room_count += 1
                        if not first_room_img:
                            first_room_img = p
                # Clear and regenerate items
                new_item_count = 0
                for it in items_sel:
                    _clear_tracking(it, "item")
                    p = generate_item_image_if_needed(state_mgr, image_gen, it, ui_data["debug"])
                    if p:
                        ui_data["images"].append(p)
                        new_item_count += 1
                latest_img = first_room_img or get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr)
                status = f"✅ Regenerated images: rooms={new_room_count}, items={new_item_count}"
                return status, ui_data["images"], get_debug_view(), latest_img
            except Exception as e:
                return f"❌ Regenerate failed: {e}", ui_data["images"], get_debug_view(), (get_current_room_image(state_mgr) or get_inventory_item_image(state_mgr))
        regenerate_btn.click(
            fn=do_regenerate_current,
            inputs=[],
            outputs=[regenerate_status, gallery, debug_md, latest_image],
        )
        def do_load_game_dropdown(file_name: str):
            try:
                if not file_name or file_name == "(none)":
                    # Return current UI state unchanged except status
                    return (
                        "❌ No saved game selected",
                        ui_data["images"],
                        ui_data["narration"],  # Return plain text narration
                        get_debug_view(),  # CHANGE: return structured JSON for gr.JSON
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
                    get_debug_view(),  # CHANGE: return structured JSON for gr.JSON
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
                    get_debug_view(),  # CHANGE: return structured JSON for gr.JSON
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

    parser = argparse.ArgumentParser(description="Minimal LLM-centered text adventure")
    parser.add_argument("--player", type=str, default="Adventurer", help="Player name")
    parser.add_argument("--model", type=str, default=None, help="Ollama model name (e.g., llama3.1:8b)")
    parser.add_argument("--device", type=str, default=None, help="Preferred device for images: cuda|mps|cpu")
    parser.add_argument("--max_tokens", type=int, default=1500, help="Max new tokens per turn")
    parser.add_argument("--diffuser", type=str, default=None, help="Diffusers model id (optional)")
    args = parser.parse_args(argv)

    state_mgr = StateManager(GameState(player_name=args.player))

    # No interactive console prompts; UI handles model selection/reload
    model_id = args.model or ""
    image_model_id = args.diffuser

    image_gen = ImageGenerator(model_id=image_model_id, device_preference=args.device, local_root=diffusion_models_root()) if image_model_id else None

    if model_id:
        print(f"Using LLM (Ollama): {model_id}")
    else:
        print("Using LLM (Ollama): (select in UI)")
    if image_gen and image_gen.model_id:
        print(f"Images: enabled via '{image_gen.model_id}' on {image_gen.device}")
    else:
        print("Images: disabled")

    llm = LLMEngine(model_id=model_id, max_new_tokens=args.max_tokens, temperature=0.7)

    # World bible is loaded from .pkl files now, not separate files
    global WORLD_BIBLE
    WORLD_BIBLE = None
    print("\nWelcome to the Minimal LLM Adventure!")
    print("The world is driven by your chosen LLM. Keep inputs concise for best results.")

    # Always launch the Gradio UI (you can still use reloads to change LLMs)
    launch_gradio_ui(state_mgr, llm, image_gen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())