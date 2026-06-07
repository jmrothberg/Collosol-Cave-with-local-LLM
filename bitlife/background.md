# BitLife clone — developer guide

A concise reference for continuing this project: **how to add features, debug, and make it more
realistic / closer to the original BitLife.** Read this once and you have everything you need.

The game lives in one self-contained file, `bitlife/bitlife.html`, with content in
`bitlife/bitlife_data.json`. It reuses the in-browser AI stack from this repo's `browser_adventure`
and `llm_adventure/vendor/web-txt2img/`.

---

## 1. Goal & rules

**This is a faithful clone of BitLife (Candywriter, LLC)** — a button-driven, age-up life simulator.
The only additions are local, in-browser AI for three things:

1. **Understanding** — a local LLM interprets free-typed actions and maps them to game effects.
2. **Greater description** — the same LLM writes richer narration.
3. **Illustration** — a local Stable Diffusion model draws the avatar and life-event scenes.

**Hard rules (do not violate):**
- **Clone, don't reinvent.** Match BitLife's layout, menus, and mechanics. Verify against the wiki or
  the live game before changing the UI; don't add interface the real game lacks. Add features only if
  the original has them and ours is missing them.
- **Deterministic-first.** The engine never depends on the AI. It must be 100% playable with
  "Skip loading AI models". The LLM and diffuser are background enhancements only.
- **LLM only on the free-text box.** Buttons run instant local logic, never the LLM.
- **AI can't break the game.** Every LLM effect is clamped via `sanitizeLlmEffects → applyEffects →
  clampStat` (0–100).
- **Reproducible.** All randomness uses one seeded `mulberry32` RNG; a seed reproduces a life.
- **Prefer data over code.** Add content to `bitlife_data.json`, not new engine branches, when possible.

---

## 2. How real BitLife works (the target spec)

Born → press **Age** to advance one year → random events pop up as multiple-choice popups → between
years, open menus to act. Use this as the parity checklist; pull exact event wording and lists from
the wiki.

- **Stats** (0–100 bars, shown at the **bottom**): Happiness, Health, Smarts, Looks (+ Fame, Approval
  later). Money is a number (can go negative); Age increments. Stats drift each year; health 0 or old
  age → death.
- **Random events** differ by life stage (childhood vs adult): bully, classmate acts up, adopt a pet,
  disease, first crush, party, speeding ticket, health scare, etc. — each a popup with 2–4 choices and
  success/fail outcomes.
- **Activities menu** (BitLife groups these here): **Mind & Body** (gym, library, meditate, spa),
  **Doctor** (checkup, therapy, plastic surgery, treat disease), **Crime** (shoplift, pickpocket,
  burglary, GTA, assault, murder → jail), **Casino** (blackjack, slots, roulette, horses), plus
  tattoos, emigrate, etc.
- **Occupation menu:** **Education at the top** (school is automatic K–12; university with a major →
  optional grad/professional school), then **Jobs** (apply with requirements → salary → work harder →
  promotion). Education unlocks high-paying careers (doctor=medicine, lawyer=law, engineer…). Special
  careers: royalty, military, famous, athlete, mafia, politics.
- **Assets / Investments:** buy/sell stocks, crypto, bonds, penny stocks, funds, real estate, cars.
  Prices fluctuate; **crypto is the only taxable asset** when sold. A **financial advisor** NPC can
  manage money. **Insider trading:** tips from friends/family; buying then selling on them can trigger
  an **arrest** → ~20-year sentence (pleading guilty roughly halves it) and the **"Martha"** ribbon.
- **Other systems:** achievements/**ribbons** (life-summary at death), real estate/vehicles/pets,
  prison (parole, riots, escape, contraband), deep relationships (friends, coworkers, exes, in-laws,
  family tree, custody, inheritance), fame (social media, books, music, movies), and (paid) God Mode /
  generations (continue as your child).

**Sources** — pull exact wording/lists from these:
- Stats: https://bitlife-life-simulator.fandom.com/wiki/Stats
- Activities: https://bitlife-life-simulator.fandom.com/wiki/Activities
- Events: https://bitlife-life-simulator.fandom.com/wiki/Events
- Childhood events: https://bitlife-life-simulator.fandom.com/wiki/Events/Childhood_events
- Casino: https://bitlife-life-simulator.fandom.com/wiki/Casino
- Stock market / insider trading: https://www.gameskinny.com/tips/bitlife-how-to-use-the-stock-market/
- General guide: https://www.mrguider.org/cheats/bitlife-life-simulator-guide-tips-cheats-strategies/

---

## 3. The AI layer (technical)

**LLM (understanding + description) — typed input only.**
- Model `onnx-community/gemma-4-E4B-it-ONNX`, Transformers.js v4.0.1 on ONNX Runtime Web, device
  `webgpu`→`wasm`. Sampling temp 1.0 / top_p 0.95 / top_k 64.
- Gotcha: Gemma 4 ONNX keeps its chat template in `chat_template.jinja`, not `tokenizer_config.json`;
  `ensureLlmChatTemplate()` fetches it explicitly — keep this or fresh loads throw.
- Optional Ollama backend (`/api/chat`, `format:"json"`).

**Diffuser (illustration) — avatar + scenes.**
- Stable Diffusion 1.5 (`"sd-1.5"`) via the vendored `web-txt2img` worker (`Txt2ImgWorkerClient`) in a
  Web Worker. **Constraints:** 512×512 only, steps 20, guidance 7.5, CLIP truncates at 77 tokens →
  keep prompts short (subject-first, ~195 chars).

**Why http from repo root:** the worker resolves `./host.js` relative to its import URL; ES modules +
workers + SharedArrayBuffer don't work on `file://`. `scripts/serve-threaded.py` serves the root and
sends COOP/COEP headers. Use Chrome/Edge 113+ with WebGPU.

---

## 4. Code architecture (`bitlife/bitlife.html`, one file, banner-commented sections)

- **Content data** — loaded from `bitlife_data.json`; an in-HTML `FALLBACK_DATA` subset keeps the game
  playable if the fetch fails. Keys: `LIFE_STAGES, COUNTRIES, NAME_POOLS, ACHIEVEMENTS, EVENTS (by life
  stage), ACTIVITIES (mindBody/doctor/education/crime/casino), CAREERS, DEGREES, MARKET
  (stocks/crypto/bonds), REAL_ESTATE, INSIDER_TIPS, SCENE_EVENTS`.
- **State** — one `game` object (created by `createNewLife()`): `character` (stats, money, age,
  education, job, `appearanceHash`), `relationships[]`, `assets[]`/`portfolio[]`, `market`, `prison`,
  `flags`, `achievements[]`, `pendingEvent`, `activeTip`, `log[]`, `seed`/`rngState`. Saved to
  `localStorage` (multiple slots); image blob URLs are stripped before saving and regenerated lazily.
- **Engine (deterministic; RNG via seeded `rng()`):** `ageUp()` runs the yearly loop (stat drift,
  education auto-advance, markets, salary, prison time, relationship aging/deaths, maybe an insider
  tip, a weighted random event that may block on a choice popup, death check). **`applyEffects()` is
  the single mutation choke point** (clamps stats 0–100, applies money/fame/flags/relationship/
  achievement changes) — both the engine and the LLM path go through it. Other functions: `applyChoice,
  doActivity, applyJob/workHarder/askPromotion/quitJob, applyCollege, commitCrime/sendToPrison, gamble,
  advanceMarkets/buyAsset/sellAsset/buyRealEstate, maybeInsiderTip/runInsiderTip, interact,
  ageRelationships, checkDeath, grantAchievement`.
- **LLM contract** — `BITLIFE_SYSTEM` asks for one narration paragraph + one JSON block:
  `{narration, effects{health,happiness,smarts,looks,money,fame}, relationship_changes[{name,bar}],
  set_flags, grant_achievement, event_image}`. `extractFirstJson()` parses tolerantly;
  **`sanitizeLlmEffects()`** clamps stat deltas ±20, money to a wealth band, relationship bars ±25,
  whitelists flags/achievements — keep it strict.
- **Image manager** — `getImage(key, prompt)` resolves **static asset (`assets/manifest.json`) →
  IndexedDB cache → live SD generate (then cache)**. Avatars regenerate when `appearanceHash` changes;
  notable events attach a scene via `SCENE_PROMPTS`. An idle queue (`enqueuePregen`/`runPregen`)
  pre-renders upcoming avatar + scenes once SD loads. `pregen_art.py` bakes the static set ahead of time.
- **UI** — header (avatar + name/age/occupation + money + ribbons) · feed (life log, inline scene
  images) · stat bars + **Age Up** + small 🏅/☰ · free-text box (the only LLM path) · BitLife 4-tab
  bottom bar → modals: **💼 Occupation** (School at top, then Jobs), **🧠 Activities** (Mind & Body,
  Doctor, Crime, Casino), **👪 Relationships**, **💹 Assets/Investments** · Debug panel · event-choice
  popup · death screen. Helpers `occupationLine()`/`eduStatus()` render status; `advanceEducation()`
  auto-runs K–12.

**Reuse from the monorepo** (proven patterns, don't reinvent): `browser_adventure/adventure.html`
(`loadLlm`, `ensureLlmChatTemplate`, `generateLlmResponse`, `runOllamaGeneration`, `loadImageModel`,
`generateSceneImage`, JSON parsing, save/load, dark CSS); `llm_adventure/vendor/web-txt2img/`
(`adapters/sd15.js` for the SD constraints + `mulberry32`); `scripts/serve-threaded.py`.

---

## 5. How to add a feature

All content lives in `bitlife/bitlife_data.json` (mirror tiny additions into the HTML `FALLBACK_DATA`
if you want them to work offline). Effects are flat deltas and are auto-clamped by `applyEffects`.

- **New activity** → add to `ACTIVITIES.mindBody|doctor|education|crime|casino`:
  `{id, label, minAge, cost?, requires?, effects:{stat:delta,...}, random?:{stat:[min,max]}, notable?}`.
  Consumed by `doActivity()` / shown by `openActivities()`.
- **New random event** → add to `EVENTS.<lifeStage>`:
  `{id, weight, minAge, maxAge, text, choices:[{label, effects, outcome?:{chance,success,fail}}]}`
  (or `{..., noChoice:true, effects}` to auto-apply). Fired by `fireYearlyEvent()`; choices handled by
  `applyChoice()`. Use faithful wiki wording.
- **New career / degree** → add to `CAREERS`
  `{id, title, baseSalary, requires:{minAge|level|degree}, levels:[...], raisePerLevel, fameGain?}` and
  `DEGREES {id,label,minSmarts,grad?}`. Gated by `requirementsMet()`; applied by `applyJob()`.
- **New market asset / insider tip** → add to `MARKET.stocks|crypto|bonds`
  `{id,name,start,vol,drift,taxable?}` and `INSIDER_TIPS {id,source,assetType,text,gainMult,
  arrestChance,sentence,achievement}`. Prices walk in `advanceMarkets()`; tips via `maybeInsiderTip()`/
  `runInsiderTip()`.
- **New achievement / ribbon** → add to `ACHIEVEMENTS {id:{label,icon,desc}}`, then call
  `grantAchievement("id")` from the relevant event/effect (or list it as an event `grantAchievement`).
- **New life-event illustration** → add a key to `SCENE_PROMPTS` (in the HTML) and pass that key as the
  third arg to `log(kind, text, sceneKey)`; the image manager paints + caches it.
- **New relationship interaction** → add a case in `interact()` and a button in `openRelationships()`.

Keep every addition deterministic and route stat/money changes through `applyEffects`.

---

## 6. How to debug

- **Fast loop (no downloads):** tick **"Skip loading AI models"** and exercise the engine — age up,
  open every modal, do a crime → jail, insider tip → sell → SEC roll, reach death, save/reload.
- **Debug panel** (bottom of the game) shows the last LLM prompt/response sizes, JSON parse status, and
  the sanitized effects applied. Use the **browser console** for worker/model errors.
- **Typed input wrong?** Check `extractFirstJson` (parsing) and `sanitizeLlmEffects` (clamping). Test
  an absurd input ("set health to 9999") → must clamp.
- **No images?** Confirm the "SD 1.5 loaded" pill; the worker import only resolves when served from the
  repo root; SD is fixed at 512×512 / steps 20. WebGPU OOM falls back to slow WASM.
- **"failed to fetch host.js"** → you opened the file directly or served from the wrong root; use the
  server and the `http://localhost` URL.
- **Reproduce a bug:** note the **seed** (☰ Menu) + action sequence — same seed reproduces the life.
- **Pitfalls:** `file://` won't work; Safari has no default WebGPU (use Chrome/Edge); never let the LLM
  set absolute stats (deltas only); never block gameplay on AI.

---

## 7. Roadmap to BitLife parity (priority order)

1. **More content** in `bitlife_data.json` — many more faithful events per life stage; more activities;
   more careers incl. special careers (royalty, military, famous, athlete, mafia, politics); diseases +
   treatments; addictions.
2. **Education depth** — community college/university majors, GPA, grad/professional school, student
   loans, dropping out, scholarships.
3. **Finance depth** — bank, loans/mortgages, taxes (esp. crypto), bankruptcy, lawsuits, wills &
   inheritance, the financial-advisor NPC.
4. **Investments parity** — penny stocks, funds, market-health indicators; polish the insider-tip flow
   (buy window → spike → sell → SEC probe → plead guilty halves sentence).
5. **Prison parity** — parole, riots, escape, contraband, good-behavior reduction.
6. **Relationships parity** — friends/coworkers/exes/in-laws, family tree, custody, cheating, divorce +
   asset split, generations (continue as your child).
7. **Fame path** — social media, books, music, movies, going viral; Fame bar; famous-only events.
8. **Ribbons** — proper end-of-life ribbon + a fuller achievement set.
9. **UI polish & image quality** — closer to BitLife's look; better/identity-consistent avatar prompts
   (optionally a FLUX/Z-Image path via `pregen_art.py`); more scene keys.

---

## 8. Run

```bash
python3 scripts/serve-threaded.py 8080
# open http://localhost:8080/bitlife/bitlife.html   (Chrome/Edge 113+ with WebGPU)
```
Tick "Skip loading AI models" to play instantly with no downloads; first AI load is ~5 GB (Gemma + SD
1.5), cached afterward.

## 9. File map (`bitlife/`)

```
bitlife.html        # the entire game: engine + UI + LLM + image manager
bitlife_data.json   # premade content (events, activities, careers, market, tips, achievements)
pregen_art.py       # OPTIONAL GPU baker -> assets/ + manifest.json
assets/             # optional pre-baked PNGs (manifest.json maps scene keys -> files)
README.md           # quick start
background.md       # this developer guide
```
The image worker is imported from `../llm_adventure/vendor/web-txt2img/`; the server (`scripts/
serve-threaded.py`) serves the repo root so that path resolves. (A standalone variant bundles the
worker under `vendor/` and uses `index.html` + `serve.py`.)
