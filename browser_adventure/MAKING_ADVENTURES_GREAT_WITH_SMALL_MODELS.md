# Making Adventure Games Great with Small (≤4B) Local Models

> A playbook for the next agent picking up `browser_adventure/adventure.html` (Gemma 4B in WebGPU + SD 1.5 in a worker). Distilled from the work done on the sibling Python engine [`../llm_adventure/LMM_adventure_April_30.py`](../llm_adventure/LMM_adventure_April_30.py) with 27B–35B class local models. Every pattern below ports down to 4B *better than it ports up to a frontier model* — small models love structure and choke on freedom.

This document is **advisory only**. Do not edit the existing browser game while reading it. When you're ready to act, propose changes in small chunks and ask for confirmation before each.

---

## Audience and goal

You are an LLM agent. You inherited:

- A working single-file in-browser adventure: [`adventure.html`](adventure.html). Renders Gemma 4 E4B narration via Transformers.js + ONNX Runtime Web (WebGPU). Generates scene art via SD 1.5 in a Web Worker.
- A working sibling Python engine [`../llm_adventure/LMM_adventure_April_30.py`](../llm_adventure/LMM_adventure_April_30.py) that runs 27B–35B local models on Apple Silicon. The Python engine has been hardened over many iterations; the browser engine has the *shape* but not the *teeth*.
- A user who wants the browser version to feel **excellent** with a 4B model — not adequate, **excellent**. They will not babysit testing.

Your goal: make the same engine pattern work great when the model is roughly 8× smaller and runs in a memory-constrained browser tab.

---

## Mental model (model-size-agnostic)

The architecture is independent of model size:

```
Player input ─┐
              ├─→ engine fast-path (deterministic state ops)
              │     └─→ no LLM call when possible
              └─→ LLM (narrator + tool-emitter)
                    │
                    ├─→ narration (free-form prose for the player)
                    └─→ JSON tool calls (structured state updates)
                          │
                          └─→ engine validates, applies, never blocks
                                │
                                └─→ warnings logged, state persists
```

Two layers of "story":

1. **World bible** (static blueprint): rooms, items, NPCs, monsters, riddles, win condition, solution chain. Generated once. Validated and repaired before play.
2. **Game state** (dynamic): location, inventory, flags, recent history. Mutated by tool calls.

The LLM is the GM. The engine is the rule-keeper. **Small models need a stronger rule-keeper.**

---

## What changed in the 27–35B Python engine, and why a 4B model needs each thing *more*

These are the moves that made the Python engine reliable. Every one of them ports to the browser, and every one matters more for Gemma 4B than it did for Qwen3.6 30B.

| Pattern | What it does | Why a 4B model needs it more |
|---|---|---|
| **Two-pass world-bible generation** (`_generate_world_skeleton` then `_generate_world_expansion`) | Pass 1 emits ONLY `locations[name+description+exits] + key_items[name+purpose] + structured win_condition`. Pass 2 emits npcs/monsters/riddles/item_locations/mechanics/solution_chain with the skeleton's names pinned literally in the prompt. | A 4B can sometimes nail one tiny well-defined JSON. It almost never nails a 12-field cross-referenced one. Smaller passes = closer to its sweet spot. |
| **Deterministic auto-repair** (`auto_repair_world_bible`) | Code-only fixes: missing item_locations entry → place at chain's location, NPC in invented room → relocate to start, isolated room → connect to predecessor, etc. | 4B leaves more mechanical gaps. Mechanical fixes never need an LLM round-trip and are instant. |
| **Per-gap micro-repair** (`micro_repair_world_bible`, `_ask_one_choice`) | For thematic gaps (monster weakness pointing at a placeholder item, broken exit, chain step missing item), ask the LLM **ONE multiple-choice question with a constrained list**. The model just copies one option from a list. | This is exactly what 4B is good at. It would fail to regenerate the whole bible to fix one thing; it succeeds at "pick one of these." |
| **Solvability validator** (`validate_world_bible_solvability`) | BFS the room graph, check item placements, blocker coverage, chain reachability. Print a one-screen report. Don't let unsolvable bibles ship. | 4B fails this check more often. The validator IS the safety net. |
| **Structured `win_condition`** | `{required_items: [...], required_location: "...", description: "..."}` instead of free text. | Direct equality check vs substring matching. 4B writes ambiguous text; structured fields are unambiguous. |
| **Solution chain requirement** | World bible MUST include an ordered list of steps proving the puzzle is winnable. | Forces the model to *think through* solvability while it's still cheap to validate. The chain is also what micro-repair operates on. |
| **JSON-mode (Ollama `format="json"`) + low temp (0.3)** | Strict JSON at decode time, less drift on names. | 4B can't be trusted to close a brace. Decode-time enforcement removes that failure mode entirely. |
| **Local fast-path commands** (`try_local_command`) | `look`, `inventory`, `take/drop/examine X`, `go/n/s/e/w`, `map`, `wait`, `help` resolved without an LLM call. ~60% of typical inputs. | Saves slow model calls, never desyncs state. Gemma 4B in WebGPU is ~5–30 tok/s; every saved call is multiple seconds back. |
| **Lean per-turn prompt** | Heavy room description injected only on the **first turn** in a room (`last_described_room` cache). Few-shot example only on the first 3 turns or after a `[WARN]`. | 4B has limited context coherence. Every spurious token degrades focus. |
| **State guards as warnings** (`[WARN] room_take MISS`, etc.) | Contradictions are logged but never block the model. | The model stays the GM; a 4B model will make more mistakes, and surfacing them lets the user understand WHY a turn was odd. |
| **Per-turn `[SUMMARY]` line** | One compact line per turn: `loc=X→Y hp=100 inv+=Map flags+=lit json=1 warns=0`. | Keeps debug actionable instead of 30 noisy tool lines. The signal is the delta, not the steps. |
| **Engine notes feedback (off by default)** | Capture this turn's warnings; optionally inject into next turn's user prompt as "fix this." | Default-off because it can crowd a 4B's limited context. Opt-in when actively diagnosing. |
| **Single best-fit few-shot** | When few-shot is added, pick ONE example matching the player's intent (move/take/use/talk/combat). | One focused example outperforms three generic ones on small models — they imitate the closest example slavishly. |
| **End-of-session post-mortem** | On win/death/restart: rooms explored, items collected, chain steps reached, warning histogram, top world-bible gaps. | The "what went wrong" report is the user's main feedback loop on a model they can't introspect. |

---

## The browser game today (April 2026 snapshot)

`adventure.html` already has the SHAPE:

- ✅ World Bible / GameState split (`DEFAULT_WORLD_BIBLE`, `createGameState`)
- ✅ Two-pass generation flow (`generateWorldBible` with Gemma or Ollama)
- ✅ JSON directive parser (`extractJsonFromText`, `applyLlmDirectives`)
- ✅ Some fast-path commands (`look`, `inv`, `map` in `handleTurn`)
- ✅ Save/load to localStorage; export/import JSON
- ✅ `synthesizeAuthorWalkthrough` for help fallback
- ✅ Streaming through `onChunk` callbacks
- ✅ Image gen in worker

It is MISSING the teeth that make 4B reliable:

| Missing | Where to add (function/symbol) | Effort |
|---|---|---|
| Extended fast-path (`take`, `drop`, `go X`, `n/s/e/w`, `examine`, `wait`, `help`) | New `tryLocalCommand(stateMgr, input)` called from `handleTurn` before LLM | small |
| Auto-repair pass after generation | New `autoRepairWorldBible(wb)` called between `generateWorldBible` and play | medium |
| Per-gap micro-repair via Gemma | New `microRepairWorldBible(wb, runLlm)` reusing same LLM the page already loaded | medium |
| Solvability validator | New `validateWorldBibleSolvability(wb)` returning `{ok, gaps, report}` | small |
| Structured `win_condition` schema | Update `DEFAULT_WORLD_BIBLE`, generation prompts, `checkWinCondition` | medium |
| Lean prompt (last-described-room cache) | Update `buildUserPrompt` + add `state.lastDescribedRoom` | small |
| Per-turn `[SUMMARY]` debug line | Update `applyLlmDirectives` to compute pre/post state and append summary | small |
| End-of-session post-mortem | New `formatSessionPostmortem(stateMgr)`; trigger on win/death/restart | small |
| Engine-note feedback (off by default, env-flag-style toggle) | Add a flag (e.g. `INJECT_ENGINE_NOTES = false`); update `buildUserPrompt` | tiny |
| One best-fit few-shot per turn (already partially present in SYSTEM_INSTRUCTIONS) | Move from system prompt to user prompt; choose by intent classifier | small |

---

## Generation pipeline for a 4B model

This is the order of operations that worked on 30B and will work *better* on 4B:

```
PASS 1: skeleton (LLM) ─┐
                        │  validate skeleton-only schema (5+ rooms, exits, structured win_condition, 4+ items)
                        │  if invalid: ONE retry with the gaps fed back, same prompt
                        ▼
PASS 2: expansion (LLM) ─┐ skeleton names pinned literally as "ONLY use these names"
                        │  validate cross-references (every npc.location is a skeleton room, etc.)
                        │  if invalid: ONE retry with gaps fed back
                        ▼
H) auto_repair (CODE) ──┐ mechanical fixes: missing item_locations, isolated rooms, NPCs in invented rooms
                        ▼
M) micro_repair (LLM × N) ─┐ ONE narrow multiple-choice question per residual thematic gap
                        ▼
A) solvability check (CODE) ─┐ BFS reachability + chain consistency report
                        │  if still broken: ONE full-bible LLM retry as last resort
                        ▼
ACCEPT plan, print solvability report
```

**Key principles for a 4B model:**

1. **Each LLM call must have ONE narrow job.** Skeleton, expansion, micro-repair — each is small. Never ask the model to "fix everything that's wrong with this JSON."
2. **Pin prior context literally.** Pass 2's prompt should contain a JSON list `["Cave","Hall","Treasury"]` — not a paragraph saying "use the rooms you generated earlier." Copy text the model can match against.
3. **Force JSON at decode time when possible.** In Ollama: `format="json"`. In Transformers.js + ONNX, full grammar-constrained decoding is harder, but you can:
   - Use [`outlines-js`](https://github.com/outlines-dev/outlines) if it has matured for ORT-Web. Check.
   - Use [WebLLM](https://github.com/mlc-ai/web-llm) which has GBNF grammar support natively.
   - Or implement **strict post-validate-and-retry**: if `JSON.parse` fails, immediately re-prompt with the same prompt + "Your previous output was not valid JSON. Reply with ONLY the JSON, beginning with `{` and ending with `}`." Three retries max.
4. **Lower temperature for generation** (0.2–0.4). Creativity comes from the theme + structure, not from sampling drift.
5. **Validate after every step.** Each pass produces something validatable. Don't accumulate errors.

---

## Runtime pipeline for a 4B model

```
Player input
   │
   ├─ tryLocalCommand → handles ~60% (move/take/drop/look/inv/map/wait/help/examine)
   │     (no LLM call; deterministic; never desyncs state)
   │
   └─ LLM turn (only if input wasn't local)
        │
        ├─ buildUserPrompt:
        │     - compact state JSON (location/inventory/exits/this-room-items)
        │     - room description ONLY on first visit
        │     - this-room cues ALWAYS (NPCs/monsters/riddles HERE)
        │     - 1 best-fit few-shot only if turn < 3 OR last turn warned
        │     - engine notes ONLY if INJECT_ENGINE_NOTES flag is on
        ▼
   Stream prose to UI as tokens arrive
        │
   Parse JSON block at end; apply state updates
        │
   Compute [SUMMARY] line; capture [WARN]s
        │
   Optionally generate scene image from images[0]
```

**Key principles for runtime on 4B:**

1. **Stream output.** Time-to-first-token matters more than total time. Show prose live; reveal image after JSON is parsed.
2. **Token budget hygiene.** Gemma 4 E4B ONNX q4 has effective ~4–8K context in browser. Stay under ~3K input tokens routinely. Lean prompt is the highest-leverage win.
3. **Deterministic narrative fallback.** When the LLM emits no parseable JSON or empty output, synthesize a short narration from world bible + current state (`"You stand in <room>. <description>. You see <items>. Exits: <list>."`). Never show a blank turn.
4. **Engine never blocks the LLM.** Every guard is a `[WARN]`; the model can still introduce new rooms (`connect`), new items (`place_items`), new NPCs in narration. Schema flexibility is the LLM's strength to leverage, not constrain.
5. **Recent history is short.** 2–3 turns max. More confuses 4B; less risks losing thread continuity.

---

## Concrete recommendations for `adventure.html`, in priority order

**Tier 1 — single biggest reliability wins. Do these first.**

1. **Extend the fast-path.** In `handleTurn`, replace the small `look/inv/map` block with a `tryLocalCommand` helper that handles `n/s/e/w/up/down`, `north/south/east/west`, `go X / enter X / move X`, `take X / get X / pick up X`, `drop X / put down X`, `examine X / x X / look at X`, `wait`, `help`. Fall through to LLM on ambiguous matches. Match exit names by exact, then substring, then first-letter for cardinals. ~60% of typical inputs handled with zero LLM calls.

   **Crucial detail (real bug from the Python sibling, May 2026):** the LLM almost always stores item names in a different style than the player types them. The world bible may contain `runic_key`, `elven_lantern`, `wizards_staff` while the narrator describes them as "the runic key", "an elven lantern", "the wizard's staff" and the player types `take the key, lantern, and staff`. Your matcher MUST handle:
   - **Multi-item lists** — split on `,`, ` and `, ` & `, ` plus ` before matching each piece.
   - **Leading articles** — strip `the / a / an / some / my / that / this` before comparing.
   - **Underscore↔space** — normalize both target and candidate by replacing `_` and `-` with spaces.
   - **Apostrophes** — strip `'` and `’` so `wizard's` matches `wizards`.
   - **Token containment** — when a target word set is a subset of a candidate word set, match. `staff` matches `wizards_staff`; `key` matches `runic_key`.
   - **Partial success** — if the player asks for three items and two match, take the two and report the miss in the same response. Don't fail the whole turn.

   See `_normalize_item_token`, `_split_item_list`, and `_match_item` in the Python sibling for a verbatim port target.

2. **Auto-repair after world-bible generation.** Before `switchToGame`, run a deterministic pass:
   - Add missing items referenced by chain/weakness/win to `key_items` with derived purposes.
   - Place each `key_items` entry that lacks an `item_locations` mapping at its `solution_chain.location` (or start room).
   - For each unreachable room, add a bidirectional exit to its predecessor in the locations list.
   - Move NPCs / monsters in invented rooms to the chain location that mentions them, else to start.
   - Set `win_condition.required_location` to start room if it's not in `locations`.
   - Drop `solution_chain.requires_item` if it's not in `key_items` (clear, don't keep a phantom).
   Log every fix. This alone will rescue a large fraction of Gemma 4B's generations without another LLM call.

3. **Solvability validator.** Build the room graph from `locations[].exits`, BFS from start, check win location is reachable, every key_item has a placement, every monster weakness references a key_item, every chain step's location is reachable from the previous. Print a `[solvability]` report block in debug. If 1+ critical gap remains after auto-repair, run micro-repair before accepting.

4. **Per-gap micro-repair.** For each residual thematic gap, ask Gemma one multiple-choice question via a tight helper:
   ```js
   async function askOneChoice(question, choices) {
     const prompt = `${question}\n\nPick exactly ONE option from this list. Reply with only the chosen text on a single line.\nOptions:\n${choices.map(c=>`  - ${c}`).join("\n")}`;
     const resp = await runLlmGeneration(SYSTEM_TERSE, prompt, 64, 0.2);
     const first = resp.trim().split("\n")[0].trim();
     return choices.find(c => c === first)
         || choices.find(c => c.toLowerCase() === first.toLowerCase())
         || choices.find(c => first.toLowerCase().includes(c.toLowerCase()))
         || null;
   }
   ```
   Categories worth fixing this way:
   - Monster weakness pointing at a placeholder/missing item → "Pick a key_item that defeats X."
   - Chain step `requires_item: null` after auto-repair cleared a bogus one → "Pick a key_item, or NONE."
   - Exit pointing at a non-room → "Pick a real room name to replace this exit."

5. **Structured `win_condition`.** Rewrite `DEFAULT_WORLD_BIBLE.win_condition` to `{required_items, required_location, description}`. Update generation prompts to demand the object. Update `checkWinCondition` to do direct struct equality. Drop substring matching.

**Tier 2 — quality of life. Do these after tier 1 lands.**

6. **Lean per-turn prompt.** Add `state.lastDescribedRoom`. In `buildUserPrompt`, only emit the heavy room-description cue when `state.lastDescribedRoom !== state.location`. Set `state.lastDescribedRoom = state.location` at the end of `buildUserPrompt`. Reset to `""` on restart/death.

7. **One best-fit few-shot.** Classify player input into `move/take/use/talk/combat/other` (cheap keyword check; same as `_classify_intent` in Python). Inject ONE matching example into the user prompt only on the first 3 turns OR when the previous turn produced a `[WARN]`. The current `SYSTEM_INSTRUCTIONS` carries multiple examples — those bloat the system prompt every turn. Move one to user-prompt-only and gate it.

8. **Per-turn `[SUMMARY]` line.** In `applyLlmDirectives`, snapshot `{location, inventory, flags, health}` at start, compute deltas at end, append one line: `[SUMMARY] loc=Cave→Hall hp=100 inv+=Map flags+=lit json=1 warns=0`. Replaces the noisy per-tool dump in the debug pane. Add a `VERBOSE_DEBUG` flag (default false) to opt back into the full dump.

9. **Engine-note feedback flag (default off).** Capture `[WARN]` lines from the current turn into `state.engineNotesForNextTurn`. In `buildUserPrompt`, only inject them if `INJECT_ENGINE_NOTES === true`. Default off — the system prompt + recent history are usually enough; nag-notes can crowd a 4B context.

10. **End-of-session post-mortem.** New `formatSessionPostmortem(stateMgr)` returns a one-page report: turns played, rooms explored / total, items collected / total, chain steps reached, warning histogram by category (`room_take MISS: 3`, `move_to UNLINKED: 1`), top 3–5 world-bible gaps (re-run solvability validator). Append to narration on win/death; show in modal on `New Adventure` after `_autosave_if_active()`. This IS the "what to fix" deliverable the user asks for.

**Tier 3 — browser-specific polish.**

11. **JSON enforcement at decode time.** Investigate Transformers.js generation_kwargs for any grammar/JSON-mode option in the version you have. If unavailable today, settle for **strict post-validate**: try `JSON.parse(extractedBlock)`; on failure, re-prompt with "Reply with ONLY the JSON, no narration." Cap at 2 retries. Only do this for world-bible passes (per-turn JSON is downstream of narration, can fall back to deterministic narration).

12. **OOM handling.** WebGPU memory failures throw recognizable errors (`/buffer|memory|allocat/i`). The current `handleTurn` catches them but offers no recovery beyond "type look." Better: on OOM, automatically retry the same turn with a TRIMMED prompt (drop few-shot, drop notes, keep only state + last 1 turn).

13. **Image budget.** Don't generate a new SD image every turn; reuse cached blob URLs for known rooms. Already partially done; tighten the rule: only call `generateSceneImage` if the room is new OR `state.gameFlags` materially changed.

14. **Scene image prompts as cached templates.** When a room is first generated, derive a scene prompt and cache it on the world bible (`location.imagePrompt`). Reuse on revisit. Saves ~2 seconds per known-room turn.

15. **Pre-loaded font / minimal CSS.** First-paint matters because users wait for ~5GB of model weights anyway; the UI should be ready instantly so the loading bar starts immediately.

---

## Anti-patterns to avoid

- ❌ **Asking 4B to regenerate the whole world bible after one error.** It will introduce three new errors. Use micro-repair instead.
- ❌ **Long cross-referenced JSON in one shot.** Use multi-pass with literal name pinning.
- ❌ **Dumping the entire world bible into every turn's prompt.** Inject only what's relevant to THIS room.
- ❌ **Multi-step agentic chains.** "Plan, then act, then reflect." 4B will fall apart. Single-step tool emission only.
- ❌ **Long chain-of-thought reasoning prompts.** The model will fill its budget with reasoning and run out before emitting JSON. If the model has a thinking mode, disable it for narration turns.
- ❌ **Asking 4B to do math or precise counting.** "Make exactly 5 rooms" works; "subtract 3 from health if X and Y but not Z" does not.
- ❌ **Open-ended "design a puzzle" prompts.** Pin the structure; let the model fill in the names and flavor text.
- ❌ **Hard-wiring features to specific model names.** Today's 4B (`gemma-4-E4B`) is tomorrow's calculator. Gate by capability heuristics (`isSmallModel(modelId)` checking size hints), never by exact name.

---

## Universal rule of thumb (any model size)

When the model fails, in this order:

1. **Make the question narrower.** Multiple choice over free form.
2. **Pin prior context literally.** Copy room names into the next prompt as a list.
3. **Validate first; auto-fix what's mechanical; LLM-fix what's thematic; retry as last resort.**
4. **Always provide a deterministic fallback so the game never bricks.**

The same code path that makes a frontier model shine should rescue a 4B browser model. No model-name gates, no special cases. Capability-based heuristics only.

---

## Closing

Build the engine so it is **boring** for a strong model and **heroic** for a weak one. The Python sibling proves the pattern works for 27–35B local models. The patterns scale *down* better than they scale up — every layer of structure becomes more valuable as the model gets smaller. With auto-repair, micro-repair, solvability validation, lean prompts, and a generous fast-path, Gemma 4B in WebGPU can deliver a competent text adventure today, and the same code will give a future 70B browser model room to shine without changing a line.

When in doubt: **read the Python sibling.** Pattern matches are deliberately one-for-one — `tryLocalCommand` ↔ `try_local_command`, `autoRepairWorldBible` ↔ `auto_repair_world_bible`, `microRepairWorldBible` ↔ `micro_repair_world_bible`, `validateWorldBibleSolvability` ↔ `validate_world_bible_solvability`, `formatSessionPostmortem` ↔ `format_session_postmortem`. Port them to JS verbatim where you can.

Good luck. Don't ask the user questions. Ship in tiers. Validate. Stop and let them play.
