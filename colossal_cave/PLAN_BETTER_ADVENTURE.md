# Plan: Make Colossal Cave a Better, More Fun Adventure

> Working notes for upcoming engine + UX improvements to `colossal_cave_July_5_26.py`.
> Saved here so we can pick this up later without re-deriving the analysis.
> All changes are intended to be **surgical** (clear `# CHANGE (...)` comments, no
> file deletions, no comment removals, no unrelated edits).

---

## TL;DR

`colossal_cave_July_5_26.py` is ~5060 lines that already contain almost everything
needed for a great adventure — the wins are mostly about **surfacing existing
state** and **routing input differently**, not new features.

Three priority tiers:

- **P1 — Player-experience wins (smallest diff, biggest first impression).**
  ~5–30 lines each, all inside `talk_to_functions`, all reuse an existing function.
  Auto-greet from MB, single Adventure Log + safety net, auto-surface
  `check_riddle_status` on room entry, ambient adjacent-room cues, status header.
- **P2 — Engine + depth (bigger diff, pays back across every play session).**
  Situational help, auto-save, end-of-run post-mortem, difficulty tiers that
  matter, fast-path commands, prompt trim, combat flavor.
- **P3 — Maintainability and structural cleanup.** Module split, lazy media
  imports, pytest, and (ASK FIRST) the unused SentenceTransformer load.

The README accuracy pass is **already done** (2026-05-04).

---

## Reality check (what the code already has — do not rebuild)

A second pass through `colossal_cave_July_5_26.py` and the helper modules turned up
several pieces I almost duplicated. The plan should **route to / surface** these
instead of writing new ones.

- **Riddle UX is mostly built.** `check_riddle_status()` (~L3487) already lists
  player magic items, says "✅ You have the right item(s)! Type: solve <X>" when
  matched, and even calls Ollama for a typed hint. Today it only fires when the
  player types `riddle status`. **Reduces to: auto-call this on room entry when
  `room.riddles and not riddle_solved`.**
- **`magic`, `inventory`, `status`, `map`, `help` already deterministic.**
  `execute_action` (~L4251–L4288) maps these to direct calls
  (`show_magic_items`, `playerOne.inventory_living()`, etc.). The LLM round-trip
  is the **parse**, not the action. Fast-path can short-circuit just the parse.
- **Three LLM safety nets already exist.** `get_llm_clarification` (~L4311),
  `get_llm_error_suggestions` (~L4331), and `get_llm_error_recovery` (~L4361)
  handle low-confidence / unrecognized / errored commands. The "fallback line"
  proposal adds a deterministic guard *before* these (Ollama down, JSON empty),
  then keeps the existing LLM ladder.
- **Help routing already smart-ish.** `get_contextual_help` (~L3464) routes
  question-shaped commands to `handle_question_with_llm` (~L3748) and otherwise
  returns `HELP_TEXT + complete_instructions`. Add a `progress_summary` line on
  top of this, do not replace.
- **Map already drawn as PNG.** `draw_map_png` (~L3381) produces a real bitmap
  with explored rooms / connections / cheat-mode reveal — wired to
  `map_image_output` in the Gradio UI. Status header is just a single text line
  alongside the existing PNG.
- **MB onboarding text is one call away.** `MasterBeast.llM_generate_self_dialogue`
  (~L3050) returns a fully-aware MB greeting using already-computed game knowledge.

### README accuracy gaps that should also be fixed (separate, surgical edit)

- **Sentence Transformers is loaded but never used.** Line 124 imports
  `SentenceTransformer`, line 397 runs
  `vector_model = SentenceTransformer('all-mpnet-base-v2', cache_folder=model_path)`,
  and `vector_model` has **zero call sites**. README claims sentence transformers
  are used for command parsing — they aren't; today **Ollama JSON-mode** parsing
  + **fuzzywuzzy** (in `Player._find_item_smart`) do the work.
- **Legacy file:** `mini_SentenceTransformer_Feb_1_dict.py` is the older parser
  and is **not imported** by the main game. Mark as legacy in the README, do not
  delete.
- **Missing from README:** Master Beast (MB) addressing pattern, `magic`,
  `riddle status`, the per-game `magicword` / `xyzzy` cheat path.
- **Optional cleanup (ASK first):** the unused `vector_model` load takes ~3 s of
  startup and downloads ~440 MB of model weights to `model_path`. Two-line removal
  if approved; otherwise leave alone.

---

## Make it actually FUN to play (player-experience wins, mostly cheap)

These are the changes a player would notice within the first 5 minutes. Each one
is grounded in code/data the game **already** has.

### A. First-30-seconds onboarding

Today: the player lands, picks New-Game, then has to read the wall of text in
`HELP_TEXT` (~L4808–4903) to discover MB. Many will type "look" and get a flat
description.

Change: when `talk_to_functions` finishes setting up a new game and the player is
in Room 1, **auto-prepend** a 3-line greeting from `MasterBeast.llM_generate_self_dialogue`
plus three concrete starter commands:

- `look` &nbsp;·&nbsp; `magic` &nbsp;·&nbsp; `MB what should I do?`

### B. Ambient adjacent-room cues (atmosphere for free)

Today: rooms feel isolated. The data is already there — every
`Room.connected_rooms` is a `dict[str, Room]` with `monsters`, `items`, `npc`,
`riddles`. Synthesize one short ambient line **from the graph** every time the
player enters a new room:

- adjacent unkilled monster → "*A low growl drifts from the <direction>.*"
- adjacent unsolved riddle → "*A faint chime hums to the <direction>.*"
- adjacent treasure-bearing room → "*Something gleams from the <direction>.*"
- adjacent NPC (non-MB) → "*You hear voices to the <direction>.*"

Pure Python, no LLM, ~30 lines. Massive atmosphere lift.

### C. Riddle UX — never make the player guess what to type

On every room entry where `current_room.riddles and not riddle_solved`, invoke
the existing `check_riddle_status` (or split off its non-LLM tail as
`_riddle_status_non_llm()` if the LLM hint is too slow) and prepend the result
to the transcript. The summary string is already correct; we just need to surface
it automatically.

Bonus: if `set(player_magic_names) >= set(current_room.riddle_magic_items)` and
the player types bare `solve`, fast-path call `playerOne.solve_puzzle(*required_items)`
with no LLM round-trip.

### D. Combat that *feels* like combat

Keep numeric damage outcomes deterministic. After a fast-path-resolved `attack`,
send **one short** prompt to Ollama (≤80 tokens) with the resolved facts
(`weapon`, `damage`, `monster.health_remaining`, was-it-a-kill) for **one**
sentence of color. Cache by `(actor, weapon, monster, hit/kill)` so repeats reuse
text. Uses *less* total LLM time than today (where the LLM is spent on every
parse instead of on flavor).

### E. Status header (one Markdown line)

Compose a single Markdown status line above the transcript:

> ❤️ 230/300 · 🪙 540 · 📍 Room 7 (Glass Hall) · ⚔️ 2 · ✨ 3 magic · ⚠️ growl from west · turn 14

All values exist (`playerOne.health/wealth`, `current_room.number/description`,
weapon/magic counts, ambient cue from §B). Existing Gradio textboxes stay for
back-compat.

### F. Single chronological transcript (the biggest readability win)

Today's UI scatters meaning across `console_output`, `room_desc_output`,
`contents_output`, `long_description_output`, `status_output`. After every turn
the player has to scan five panes.

Change: add a `gr.Markdown` (or `gr.Chatbot`) as the primary surface, fed by a
per-turn buffer:

```text
> take sword
You pick up the Bastard Sword (35 damage).

You hear a low growl from the west.
[Room 7 — Glass Hall]
A vast cavern of glittering glass spires…
Items: Healing Potion, Iron Key
Exits: north, west
```

Existing panes stay (mirror-write for back-compat). **The transcript becomes the game.**

Empty / failure path: if `result == ""` or `process_command` raised before the
existing LLM-recovery ladder, append a deterministic one-liner:
*"(no response — try: look · inventory · go <direction>)"* — covers the
Ollama-down case (where `ollama_generate` ~L3695 currently returns
`"Error generating response"` raw).

### G. Help that responds to the player's situation

Add a small `progress_summary(player, rooms) -> str` (pure Python, no LLM) that
prepends a 3-line **stuck-detector** to whichever existing `get_contextual_help`
branch fires:

- "You haven't visited <N> rooms; try going <direction with most unknown territory>."
- "You have <K> magic items; <M> rooms still hold riddles."
- "MB is in Room 1. Travel back any time and ask her: <copy/paste line>."

Gate: only inject when `len(player.items) < 2` or
`(current_room.riddles and not riddle_solved and no eligible magic in inventory)`.

### H. Save / load that doesn't lose state on a crash

Auto quick-save to `quicksave.pkl` on every successful room change (cheap; pickle
already exists). On startup, if `quicksave.pkl` exists, offer a **"Resume last run"**
button alongside the existing dropdown.

### I. End-of-run summary (the moment players screenshot)

When the player wins (any trophy condition in `check_trophy`) or dies, append a
small post-mortem to the transcript: rooms visited / total, monsters killed,
riddles solved, peak wealth, MVP weapon (most-used). Implementable from existing
counters.

### J. Difficulty that *means* something beyond hit frequency

Today: `DIFFICULTY_ATTACK_FREQ` only changes monster swing rate. Hook each tier
into player UX too:

- **Cheat**: ambient cues spell out exact items in adjacent rooms; MB greeting
  includes the win path.
- **Easy**: ambient cues + riddle hint as today.
- **Medium**: ambient cues only on entering a room (not every turn).
- **Hard**: no ambient cues, no riddle hints; status line shows HP and location only.

---

## Priority 1 — Player-experience wins (smallest diff, biggest first impression)

These are the changes a player notices in the first 5 minutes. Each one is
**a few lines** and reuses an existing function. Pick any one to start.

- **§A onboarding greet** — one new helper called from `talk_to_functions` right
  after `game_state == "New-Game"` finishes; calls existing
  `MasterBeast.llM_generate_self_dialogue` (~L3050) with a fixed "what should I
  do first?" prompt. Append three suggested commands to the transcript:
  `look · magic · MB what should I do?`. **~5 lines.**
- **§F transcript + safety net** — new `gr.Markdown(label="Adventure Log")`
  above the existing image gallery; an accumulator string in `talk_to_functions`
  appends both player input and `console`. Existing textboxes stay (mirror-write
  for back-compat). Append a deterministic one-liner
  *"(no response — try: look · inventory · go &lt;direction&gt;)"* when
  `result == ""` or `ollama_generate` (~L3695) returned `"Error generating response"`.
  **~30 lines.**
- **§C riddle eligibility** — auto-call existing `check_riddle_status` (~L3487)
  inside the `if current_room.riddles and not riddle_solved:` branch (~L4675)
  and prepend its output. The summary string is already correct; we just surface
  it on every entry instead of waiting for the player to type `riddle status`.
  Bonus: bare `solve` fast-path to `playerOne.solve_puzzle(*required_items)`
  when `set(player_magic) >= set(room.riddle_magic_items)`. **~10 lines.**
- **§B ambient cues** — new `def adjacent_cues(room) -> str` (pure Python, no
  LLM); reads `room.connected_rooms[*].monsters/items/npc/riddles`; call once
  when `old_room != playerOne.current_room` in `talk_to_functions`. One short
  line per adjacency: growl / chime / glow / voices. **~30 lines.**
- **§E status header** — new `gr.Markdown` above transcript; one helper builds
  the line from `playerOne.health/wealth/points/kills`,
  `current_room.number/description`, weapon/magic counts, and `adjacent_cues`
  warning. **~15 lines.**

Deliverable: a noticeably more inviting and reactive game; file structure
unchanged; no migration risk.

---

## Priority 2 — Engine + depth (bigger diff, pays back across every play session)

These need more code or more care. Each is independent of the others; do them
in any order after at least one P1 has shipped.

- **§G situational help** — `progress_summary(player, rooms) -> str` prepended
  inside `get_contextual_help` (~L3464); existing `HELP_TEXT + complete_instructions`
  and `handle_question_with_llm` paths unchanged. Gate: only inject when player
  has &lt;2 items or is stuck on an unsolvable riddle. **~20 lines.**
- **§H auto-save** — 4-line addition inside the room-change branch near the
  bottom of `talk_to_functions`: call `save_game_state("quicksave")` if the
  player moved this turn. New "Resume last run" button on startup reads the
  same path. **~10 lines.**
- **§I post-mortem** — extend existing `check_trophy` callsites and the death
  branch (~L4721) to append a stats line from already-tracked counters
  (`player.kills`, `wealth`, `points`, `len(visited)`, `len(trophies)`,
  MVP weapon). **~20 lines.**
- **§J difficulty tiers** — read `difficulty_g` (already global) inside
  `adjacent_cues`, `progress_summary`, and the new status header so cheat /
  easy / medium / hard verbosity actually differs. **~10 lines wired through helpers.**
- **fast-path commands (`try_local_command`)** — called before
  `llm_process_command` inside `process_game_command`; reuses existing
  `Player.move/take/leave/use/study`, `show_magic_items`,
  `playerOne.inventory_living()`, etc. **CRITICAL:** movement fast-path must
  check `dest_room.monsters` and any blocker flags so encounters are not
  silently skipped — this is the main risk class flagged in
  `COMPARISON_AND_BEST_OF_BOTH_WORLDS.md` for the MLX engine. **~100 lines.**
- **prompt trim + token hygiene** — inside `llm_process_command` (~L4005): only
  include riddle prose when the command could plausibly be `solve` or `talk`;
  cache full action grammar after the first turn; add `last_described_room` so
  repeat `look` in the same room sends a short cue. **~50 lines.**
- **combat flavor** — keep numeric outcomes deterministic (in `Player.attack`
  etc.); after a fast-path-resolved swing send **one** ≤80-token Ollama prompt
  with resolved facts (`weapon`, `damage`, `monster.health_remaining`,
  was-it-a-kill) for one sentence of color. Cache by
  `(actor.name, weapon.name, monster.name, hit/kill)`. Net LLM time per session
  goes **down** because parses no longer hit the LLM. **~30 lines.**

Deliverable: the game *feels* like a game (responsive, reactive, dramatic);
core loop is faster on local models.

---

## Priority 3 — Maintainability and structural cleanup (no player-visible change)

Take these on once the player-facing wins are in. They make future work cheaper
but don't move the needle on a play session.

- **Module split** — extract under `colossal_cave/`:

| Module (proposed) | Responsibility |
|-------------------|----------------|
| `colossal_cave/config/devices.py` | MPS/CUDA selection, dtype — no Gradio |
| `colossal_cave/media/diffusion.py` | `set_up_llm_diffuser`, image generation helpers |
| `colossal_cave/media/video.py` | Pyramid Flow / temp files |
| `colossal_cave/engine/session.py` | `GameSession` dataclass holding `rooms`, `player`, flags now global |
| `colossal_cave/engine/commands.py` | `process_command`, fast-path, `execute_action` dispatch |
| `colossal_cave/engine/llm_commands.py` | `llm_process_command`, prompts, parsing |
| `colossal_cave/ui/gradio_app.py` | `gr.Blocks`, wiring only |

  Keep `colossal_cave_July_5_26.py` as a thin entry / re-export so existing paths
  still work. Migration tactic: **move functions verbatim first**, fix imports,
  run one smoke path (new game → move → take → LLM-only line). No behavior change.

- **Lazy media imports + scoped torch patches** — defer `diffusers` / video
  imports until the user enables image/video; replace runtime monkey-patching of
  `torch.from_numpy` / `torch.arange` in `set_up_llm_diffuser` with a **scoped
  context** called only inside FLUX inference. Faster startup, no global torch
  mutation.
- **Pytest core** — extract pure helpers (direction normalization,
  `Player._find_item_smart`, trophy checks, the new `adjacent_cues`) and add
  small `pytest` tests under `colossal_cave/tests/` that run without GPU or
  Ollama.
- **`dead-vector-load` (ASK FIRST)** — `vector_model = SentenceTransformer(...)`
  at line 397 is loaded but never referenced. Removing the import (line 124) +
  load (line 397) saves ~3 s and ~440 MB at startup. Two-line surgical change.
  **Do not silently delete.**
- **Image prompt composition (optional)** — adopt capped, compositional prompt
  building (browser `composeImagePromptForSD` idea, see
  `../browser_adventure/adventure.html`) for room/NPC montages. Even FLUX
  benefits from consistent structure. Borrow the **pattern**, not the schema.

---

## Risk notes

- **Fast-path vs narrative gates**: If you add deterministic `go`, validate
  against combat / blocking semantics already encoded in `Room` / `Monster`
  logic so players cannot bypass encounters accidentally.
- **Gradual rollout**: P1 alone is a real upgrade. P2 lands the bigger
  structural fun (combat, fast-path). P3 is only worth doing once P1+P2 are
  stable.

---

## Open todos (priority-tagged)

| Priority | ID | Description |
|---|---|---|
| **DONE** | `readme-accuracy` | 2026-05-04: edited `README.md` in place — dropped SentenceTransformer parsing claim, marked `mini_SentenceTransformer_Feb_1_dict.py` as legacy, added MB section, `magic` / `riddle status` / `magicword` commands, all standalone scripts and downloaders documented |
| P1 fun | `onboarding` | Auto-greet from MB on first turn in Room 1 with 3 suggested commands; quick-start banner instead of dumping `HELP_TEXT` (~5 lines) |
| P1 fun | `transcript-fallback` | Single chronological Adventure Log (Markdown) above existing gallery; deterministic safety-net line when result is empty / Ollama is down (~30 lines) |
| P1 fun | `riddle-ux` | Auto-call `check_riddle_status` on room entry when room has unsolved riddle; bare `solve` fast-paths to `playerOne.solve_puzzle` when match is unambiguous (~10 lines) |
| P1 fun | `ambient-signals` | `adjacent_cues(room) -> str` pure Python, called when `old_room != current_room` (~30 lines) |
| P1 fun | `status-header` | `gr.Markdown` status line above transcript: HP, wealth, location, weapons, magic count, ambient warning, turn counter (~15 lines) |
| P2 fun | `situational-help` | `progress_summary(player, rooms)` prepended in `get_contextual_help`; gated when player is stuck (~20 lines) |
| P2 fun | `auto-save` | `quicksave.pkl` on every successful room change; "Resume last run" button on startup (~10 lines) |
| P2 fun | `end-of-run` | Post-mortem stats line on win/death from existing counters (~20 lines) |
| P2 fun | `difficulty-tiers` | `difficulty_g` gates verbosity in cues / hints / status header (~10 lines wired through) |
| P2 fun+perf | `fast-path` | `try_local_command` before `llm_process_command`; **must** check `dest_room.monsters` for movement (~100 lines) |
| P2 perf | `prompt-trim` | Trim `llm_process_command` prompt; conditional sections + `last_described_room` cache (~50 lines) |
| P2 fun | `combat-flavor` | One short LLM flavor line per swing, cached by (actor, weapon, monster, hit/kill); deterministic damage stays in code (~30 lines) |
| P3 maintain | `split-modules` | **PARTIAL (2026-07):** `engine/`, `ui/`, `media/`, `config/` extracted; main file still holds classes + game loop. Further split: `engine/commands.py` full dispatch, class move to `engine/models.py`. **Test gate:** run `pytest -m "not gpu and not ollama"` after each extraction. |
| P3 maintain | `lazy-media` | **PARTIAL (2026-07):** `media/lazy_imports.py` added; main game still eager-imports diffusers at startup — wire lazy path when enabling images. |
| P3 maintain | `pytest-core` | **DONE (2026-07):** `tests/` with 40+ CI-safe tests; GitHub Actions workflow `.github/workflows/colossal_cave_tests.yml`. P1/P2 features must add/update tests in matching file (e.g. `adjacent_cues` → `tests/test_adjacent_cues.py`). |
| P3 ASK first | `dead-vector-load` | Two-line removal of unused `SentenceTransformer` load at L124 + L397; saves ~3 s + ~440 MB startup |
