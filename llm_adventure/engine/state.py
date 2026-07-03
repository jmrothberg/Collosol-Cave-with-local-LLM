"""GameState and StateManager."""

import json
import os
import pickle
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from config.paths import IN_GAME_IMG_DIR, SAVE_DIR
from engine.bible_store import get_world_bible, set_world_bible


@dataclass
class GameState:
    player_name: str = "Adventurer"
    location: str = "Start"
    health: int = 100
    inventory: List[str] = field(default_factory=list)
    room_items: Dict[str, List[str]] = field(default_factory=dict)
    known_map: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "Start": {"exits": [], "notes": "A nondescript starting place."}
    })
    notes: List[str] = field(default_factory=list)
    recent_history: List[Dict[str, str]] = field(default_factory=list)
    story_context: str = ""
    game_flags: Dict[str, Any] = field(default_factory=dict)
    engine_notes_for_next_turn: List[str] = field(default_factory=list)
    session_turns: int = 0
    warnings_history: List[str] = field(default_factory=list)
    chain_steps_completed: List[int] = field(default_factory=list)
    last_described_room: str = ""
    visited_rooms: List[str] = field(default_factory=list)
    last_images: List[Dict[str, str]] = field(default_factory=list)
    rooms_with_images: List[str] = field(default_factory=list)
    items_with_images: List[str] = field(default_factory=list)
    recent_actions: List[str] = field(default_factory=list)
    images_generated: int = 0
    images_reused: int = 0
    turns_played: int = 0

    def add_room_image(self, room_name: str, image_path: str, prompt: str) -> None:
        self.last_images.append({"prompt": prompt, "path": image_path, "type": "room", "subject": room_name})
        if room_name not in self.rooms_with_images:
            self.rooms_with_images.append(room_name)
        self.images_generated += 1

    def add_item_image(self, item_name: str, image_path: str, prompt: str) -> None:
        self.last_images.append({"prompt": prompt, "path": image_path, "type": "item", "subject": item_name})
        if item_name not in self.items_with_images:
            self.items_with_images.append(item_name)
        self.images_generated += 1

    def has_room_image(self, room_name: str) -> bool:
        return room_name in self.rooms_with_images

    def has_item_image(self, item_name: str) -> bool:
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

    def move_to(self, new_location: str) -> None:
        if not new_location:
            return
        if new_location not in self.state.known_map:
            self.state.known_map[new_location] = {"exits": [], "notes": ""}
        self.state.location = new_location
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
        self.state.health = max(-10, min(200, self.state.health + int(delta)))

    def add_note(self, note: str) -> None:
        if note:
            self.state.notes.append(note)

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

    def get_shortest_path(self, start_room: Optional[str] = None, target_room: str = "") -> List[str]:
        start = start_room or self.state.location
        if start == target_room:
            return [start]
        visited = set()
        queue = [(start, [start])]
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
        return []

    def describe_map(self) -> str:
        visited = list(self.state.visited_rooms or [])
        if not visited and self.state.location:
            visited = [self.state.location]
        if not visited:
            return "(map is empty — explore to discover rooms)"
        visited_set = set(visited)
        lines: List[str] = []
        for room in visited:
            info = self.state.known_map.get(room) or {}
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

    def save(self, label: Optional[str] = None) -> str:
        timestamp = datetime.now().strftime("%b_%d_%H_%M_%S")
        label = label or f"{self.state.player_name}_{timestamp}"
        pkl_name = f"Adv_{label}.pkl"
        pkl_path = os.path.join(SAVE_DIR, pkl_name)
        try:
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
            save_data = {
                "game_state": self.state,
                "world_bible": get_world_bible(),
                "images": image_data,
                "save_version": "2.0",
            }
            with open(pkl_path, "wb") as f:
                pickle.dump(save_data, f)
        except Exception as e:
            print(f"[save] pkl creation failed: {e}")
        return pkl_path

    def load(self, file_path: str) -> None:
        if file_path.lower().endswith(".pkl") and os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict) and "save_version" in data:
                self.state = data["game_state"]
                set_world_bible(data.get("world_bible"))
                image_data = data.get("images", {})
                for filename, img_bytes in image_data.items():
                    try:
                        img_path = os.path.join(IN_GAME_IMG_DIR, filename)
                        with open(img_path, "wb") as img_f:
                            img_f.write(img_bytes)
                    except Exception:
                        pass
                return
            raise Exception("Unsupported save format - please create a new save")
        if file_path.lower().endswith(".tkl") and os.path.isfile(file_path):
            with zipfile.ZipFile(file_path, "r") as zf:
                tmp_dir = os.path.join(SAVE_DIR, f"tmp_{int(time.time())}")
                os.makedirs(tmp_dir, exist_ok=True)
                zf.extractall(tmp_dir)
                json_files = [p for p in os.listdir(tmp_dir) if p.endswith(".json") and p.startswith("Adv_")]
                if json_files:
                    with open(os.path.join(tmp_dir, json_files[0]), "r", encoding="utf-8") as f:
                        self.state = GameState.from_json(f.read())
                wb_files = [p for p in os.listdir(tmp_dir) if p.endswith(".world_bible.json")]
                if wb_files:
                    with open(os.path.join(tmp_dir, wb_files[0]), "r", encoding="utf-8") as f:
                        set_world_bible(json.load(f))
                img_src = os.path.join(tmp_dir, "in_game_images")
                if os.path.isdir(img_src):
                    for fn in os.listdir(img_src):
                        try:
                            shutil.copyfile(os.path.join(img_src, fn), os.path.join(IN_GAME_IMG_DIR, fn))
                        except Exception:
                            pass
                try:
                    shutil.rmtree(tmp_dir)
                except Exception:
                    pass
            return
        with open(file_path, "r", encoding="utf-8") as f:
            self.state = GameState.from_json(f.read())
