## JMR's LLM Adventure — Short Guide

### What this is
An LLM-driven, turn-based text adventure with optional image generation. You type actions; a local LLM (via Ollama) writes the story and issues tool calls. The engine applies those calls to update game state, show images, and keep the world consistent.

### How a turn works
1. You enter an action (e.g., "open chest", "go north", "examine crystal key").
2. The turn LLM returns narration plus a small JSON block of directives.
3. The engine parses the JSON and executes tools (state updates and image requests).
4. Narration streams immediately; images are reused from cache or generated and then displayed.
5. Game state (map, inventory, notes, images) persists across turns.

### The LLM's central role
- The LLM decides the story beats and calls tools by emitting JSON directives appended after the narration.
- A static "World Bible" provides setting, items, NPCs, and overall constraints for consistency.
- Better local models improve reliability: cleaner JSON, fewer truncations, smarter tool use.
- The LLM follows storytelling principles: tension building, multiple solutions, fail-forward design, and narrative callbacks.

### What the Turn-LLM sees each turn (NOT the entire game!)

The Turn-LLM receives a **carefully curated subset** of information, not the entire World Bible or full game state. This keeps the LLM fast and focused.

**From GameState (the complete player-relevant state):**
```json
{
  "player_name": "Hero",
  "location": "Dark Passage",
  "health": 95,
  "inventory": ["torch", "rope"],
  "known_map": {/* only rooms you've visited */},
  "notes": ["Found secret door behind waterfall"],
  "recent_conversation": [/* last 5 turns */],
  "story_context": "Player discovered key puzzle. Building tension toward door.",
  "game_flags": {"door_blocked": true, "mech_0": {...}},
  "current_room_items": ["crystal shard"],
  "current_exits": ["Start", "Underground River"],
  "visited_rooms": ["Start", "Dark Passage"],
  "world_theme": "Studio Ghibli background art...",
  "win_condition": "Obtain the golden treasure and return to entrance",
  "rooms_with_images": ["Start", "Dark Passage"],
  "items_with_images": ["torch"]
}
```
Note: This IS the complete game-relevant state. What's NOT included are internal engine details like file paths, image cache metadata, autosave timers, etc.

**From World Bible (ONLY relevant parts for current location):**
```
World context:
  NPC here: Old Hermit - helpful guide
    (can provide: torch and warning)
  Monster here: Cave Troll (medium)
    (weakness: fears fire from torch)
  Puzzle: I have cities but no houses...
    (solving grants: temple access)
  Mechanic: light torch in dark passage -> reveals hidden rope
  Item available: rope - crosses river safely
  Current goal: Gain passage across the underground river
  Hint: Fire is both light and weapon
```

**What the Turn-LLM NEVER sees:**
- NPCs, monsters, riddles from OTHER rooms
- The complete item catalog from the World Bible
- Locations you haven't discovered yet
- The full game's win condition (unless you're near the end)
- Internal engine state (image paths, cache details, etc.)

**Why this selective approach works:**
1. **Speed**: Smaller prompts = faster responses (critical for local LLMs)
2. **Focus**: LLM concentrates on what's immediately relevant
3. **Discovery**: Player genuinely discovers content, not spoiled by omniscient AI
4. **Flexibility**: LLM can still create new content beyond the World Bible
5. **Consistency**: World Bible cues guide without constraining

### How GameState and World Bible work together

**GameState = What IS**
- Current reality: where you are, what you have, what you've done
- Dynamic and changes every turn based on your actions
- The LLM sees ALL of the active game state every turn (inventory, location, health, notes, flags, etc.)
- Does NOT see internal engine data (image file paths, cache metadata, etc.)

**World Bible = What COULD BE**
- The game's potential: NPCs to meet, puzzles to solve, items to find
- Static blueprint created once at game start
- The LLM sees ONLY relevant excerpts for your current location
- Examples: "There's a hermit here who gives torches" (only when you're in his room)

**Goal alignment**
- The win_condition is included in the per-turn context so the Turn-LLM can keep ad‑hoc decisions consistent with the overall objective (just like theme).

**The magic combination:**
- GameState tells the LLM the current facts
- World Bible provides hints about possibilities in this location
- The LLM weaves them together into narrative and decides what happens
- The LLM can follow the World Bible OR invent something completely new

**Example turn:**
```
GameState says: You're in Dark Passage with a torch
World Bible hints: "Mechanic: light torch here -> reveals rope"
Player says: "I light my torch"
LLM decides: Follow the mechanic! Narrates finding rope, adds it via place_items
```

But the LLM could also decide to create something new:
```
LLM decides: Invent new content! Narrates finding a secret door instead
Creates new room "Hidden Vault" via move_to + connect
```

### Dynamic world expansion (beyond the World Bible)
The turn LLM has **complete creative authority** to expand beyond the World Bible - this is a feature, not a bug!

**How it works:**
- **World Bible = Starting framework**, not rigid constraints
  - Provides initial NPCs, monsters, objectives, key items
  - Sets the tone and main story arc
  - But it's a guide, not a prison

- **Turn LLM = Creative engine** with full world-building power
  - Can create ANY new room via `move_to` + `connect`
  - Can place ANY new item via `place_items`
  - Can invent new NPCs, puzzles, secrets on the fly
  - Has complete authority through JSON tools

**Example of dynamic expansion:**
```json
// Player finds secret passage, LLM creates entirely new content:
{
  "state_updates": {
    "move_to": "Crystal Grotto",  // New room not in World Bible!
    "connect": [["Dark Cave", "Crystal Grotto"]],
    "place_items": ["Luminous Shard"],  // New item not in World Bible!
    "add_note": "Behind the waterfall, a hidden grotto sparkles",
    "set_context": "Player found secret area. This leads to alternate path."
  },
  "images": ["Crystal Grotto with luminous shards"]
}
```

**Why this is the killer feature:**
- **Emergent gameplay** - Every playthrough can be unique
- **Responsive storytelling** - LLM adapts to player actions
- **Infinite content** - Not limited to pre-designed areas
- **Player agency** - Their choices unlock content the World Bible never imagined
- **Replayability** - Same World Bible, different adventures each time

The static World Bible provides consistency and narrative spine, while the LLM provides magic and surprise. Pregenerated content offers efficiency (cached images), while dynamic creation offers discovery.

### Tool calls (JSON directives)
The LLM appends JSON like:

```json
{
  "state_updates": {
    "move_to": "Hidden Chamber",
    "connect": [["Hidden Chamber", "Gloomy Cavern"]],
    "place_items": ["Crystal Key"],
    "room_take": ["Crystal Key"],
    "add_note": "Dust cloud hides a small key.",
    "set_context": "Player discovered the key puzzle. Building tension toward door sequence.",
    "change_health": -5
  },
  "images": ["Crystal Key close-up", "Hidden Chamber overview"]
}
```

- **set_context**: The LLM's narrative memory - tracks story threads, foreshadowing, and intentions
- Images are generated only when needed; otherwise cached images are reused automatically
- Display prefers a newly added image that turn; otherwise shows the current room (and will favor a requested item image if cached)

### Getting better results (no code changes)
- Keep your commands short and specific: "show close-up of the Crystal Key" when you want the item image.
- Use in-world phrasing to hint at tool use: "connect the Hidden Chamber to the Gloomy Cavern" or "take the Crystal Key".
- If you see rare dangling ```json in narration, it's harmless truncation; using a slightly larger token limit reduces it.

### How the game improves with better LLMs
The game automatically gets better as local LLMs improve, without any code changes:

**Current LLMs (GPT-3.5 level):**
- Basic room connections and item management
- Simple NPCs with 2-turn memory
- Direct cause-and-effect narratives

**Better LLMs (GPT-4 level):**
- Complex multi-room puzzles using story_context
- NPCs with personality that evolves based on player actions
- Foreshadowing and narrative callbacks across many turns
- Multiple solution paths emerging naturally

**Future LLMs:**
- Dynamic story arcs that adapt to playstyle
- Complex NPC relationships and faction systems
- Emergent gameplay mechanics through creative tool use
- Rich narrative threads maintained across entire sessions

The key: all improvements come from the LLM better using the same simple tools, not from adding complexity to the code.

### Story Context Implementation Details

**What was added to the code:**

1. **GameState class (line 458):** Added `story_context: str = ""` field
   - The LLM's narrative memory for tracking story threads, foreshadowing, and intentions
   - Persists across turns and save/load cycles

2. **SYSTEM_INSTRUCTIONS (lines 1630-1636):** Added storytelling principles
   - Create tension: foreshadow dangers, build to climaxes
   - Reward exploration: hide secrets, reward careful examination  
   - Fail forward: setbacks create new opportunities, not dead ends
   - Remember earlier events: reference past actions when relevant
   - Multiple solutions: consider 2+ ways to overcome obstacles
   - Use story_context to track narrative threads and intentions

3. **JSON tool handler (lines 1913-1915):** Processes set_context
   ```python
   if isinstance(updates.get("set_context"), str):
       state_mgr.state.story_context = updates["set_context"]
       debug_lines.append(f"tool.set_context: {updates['set_context'][:50]}...")
   ```

4. **LLM context (line 2452):** story_context included in every turn
   - The LLM sees this in the JSON context alongside inventory, location, etc.

5. **Debug visibility (line 3099):** Shows in GameState debug view
   - Appears under "conversation" section in Gradio UI
   - Visible in debug logs when LLM uses the tool

6. **Save/load compatibility (line 699):** Backward compatible
   - Old saves work fine, story_context defaults to empty string

**How the LLM uses story_context:**
- Tracks narrative intentions between turns
- Maintains foreshadowing and buildup
- Remembers important story beats for callbacks
- Coordinates multi-turn puzzle sequences
- As LLMs improve, they'll use this more sophisticatedly

### Simple tool ideas (easy future additions)
- highlight_item(subject): temporarily prefer showing an item's image for this turn.
- set_theme(style): update the global art style the diffuser uses.
- mark_interest(subject): add a note and pin the item image in the gallery.
- reveal_map(room, exits): annotate map and exits in one call (pairs well with connect).
- play_sfx(name): trigger a lightweight sound cue (pickup, riddle solved) without changing gameplay.

### Launch (quick)
1) Start Ollama: `ollama serve`
2) Run: `python LMM_adventure_Aug_23_25.py`
3) In the UI, select an Ollama model and optionally load a diffuser, then play.


