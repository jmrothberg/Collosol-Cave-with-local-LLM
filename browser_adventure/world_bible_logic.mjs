/** Pure world-bible logic — shared by adventure.html and world_bible_tests.mjs */

export function fixCommonJsonErrors(s) {
  s = s.replace(/,\s*}/g, "}").replace(/,\s*]/g, "]");
  let ob = 0, cb = 0, osq = 0, csq = 0;
  for (const c of s) { if (c === "{") ob++; else if (c === "}") cb++; else if (c === "[") osq++; else if (c === "]") csq++; }
  while (cb < ob) { s += "}"; cb++; }
  while (csq < osq) { s += "]"; csq++; }
  while (s.split("}").length - 1 > s.split("{").length - 1) s = s.replace(/}$/, "");
  while (s.split("]").length - 1 > s.split("[").length - 1) s = s.replace(/]$/, "");
  return s;
}

export function tryParseJson(s) {
  try { const o = JSON.parse(s); if (o && typeof o === "object") return o; } catch {}
  try { const o = JSON.parse(fixCommonJsonErrors(s)); if (o && typeof o === "object") return o; } catch {}
  return null;
}

/** Extract JSON objects from LLM text. Returns array of parsed objects. */
export function extractJsonFromText(text) {
  const results = [];
  const fenceRe = /```(?:json)?\s*([\s\S]*?)```/gi;
  let m;
  while ((m = fenceRe.exec(text)) !== null) {
    const parsed = tryParseJson(m[1].trim());
    if (parsed) results.push(parsed);
  }
  if (results.length) return results;

  let depth = 0, start = -1;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === "{") { if (depth === 0) start = i; depth++; }
    else if (text[i] === "}") {
      depth--;
      if (depth === 0 && start >= 0) {
        const parsed = tryParseJson(text.slice(start, i + 1));
        if (parsed) results.push(parsed);
        start = -1;
      }
    }
  }
  return results;
}

/** Helper: extract first valid JSON object from LLM output. */
export function extractFirstJson(raw) {
  const cleaned = raw.replace(/```json/gi, "").replace(/```/g, "").trim();
  const candidates = extractJsonFromText(cleaned);
  if (candidates.length) return candidates[0];
  const start = cleaned.indexOf("{");
  const end = cleaned.lastIndexOf("}") + 1;
  if (start >= 0 && end > start) return tryParseJson(cleaned.slice(start, end));
  return null;
}

/** Map skeleton from pass 1: try several strategies (truncated JSON, prose with embedded object). */
export function extractMapSkeleton(raw, genDebugArr) {
  if (raw == null || !String(raw).trim()) return null;
  let cut = String(raw).trim();
  const brace = cut.indexOf("{");
  if (brace > 0) {
    cut = cut.slice(brace);
    if (genDebugArr) genDebugArr.push(`[world-gen detail] skeleton: stripped ${brace} chars before first {`);
  }
  let sk = extractFirstJson(cut);
  if (sk && Array.isArray(sk.rooms) && sk.rooms.length) return sk;
  const all = extractJsonFromText(cut);
  for (const obj of all) {
    if (obj && Array.isArray(obj.rooms) && obj.rooms.length) return obj;
  }
  if (genDebugArr) genDebugArr.push(`[world-gen detail] skeleton: no object with .rooms[] (${all.length} JSON object(s) found)`);
  return sk;
}

/** Render win_condition as printable text whether structured object or legacy string. */
export function winConditionText(wb) {
  if (!wb || !wb.win_condition) return "";
  const wc = wb.win_condition;
  if (typeof wc === "string") return wc;
  if (typeof wc === "object") {
    if (wc.description && String(wc.description).trim()) return String(wc.description);
    const item = (wc.required_items || [])[0];
    const loc = wc.required_location;
    if (item && loc) return `Obtain ${item} and return to ${loc}`;
    if (loc) return `Reach ${loc}`;
    return "Complete the adventure";
  }
  return String(wc);
}

/** Normalize string win_condition to structured object (mutates wb). Returns true if changed. */
export function normalizeWinCondition(wb) {
  if (!wb || typeof wb !== "object") return false;
  const rooms = (wb.locations || []).map(l => l && l.name).filter(Boolean);
  if (!rooms.length) return false;
  const firstRoom = rooms[0];
  const roomSet = new Set(rooms);

  if (typeof wb.win_condition === "string" || !wb.win_condition || typeof wb.win_condition !== "object") {
    const oldText = typeof wb.win_condition === "string" ? wb.win_condition : "Complete the adventure";
    const finalGives = (wb.puzzle_chain || []).slice().reverse().find(s => s && s.gives);
    const finalItem = (finalGives && finalGives.gives) ||
      ((wb.key_items || [])[((wb.key_items || []).length - 1)] || {}).name || null;
    wb.win_condition = {
      required_items: finalItem ? [finalItem] : [],
      required_location: firstRoom,
      description: oldText,
    };
    return true;
  }

  let changed = false;
  if (!Array.isArray(wb.win_condition.required_items)) {
    wb.win_condition.required_items = [];
    changed = true;
  }
  if (!wb.win_condition.required_location || !roomSet.has(wb.win_condition.required_location)) {
    wb.win_condition.required_location = firstRoom;
    changed = true;
  }
  if (typeof wb.win_condition.description !== "string" || !wb.win_condition.description.trim()) {
    const item = wb.win_condition.required_items[0] || "the prize";
    wb.win_condition.description = `Obtain ${item} and return to ${firstRoom}`;
    changed = true;
  }
  return changed;
}

/** Build a plain-text solution from structured bible fields (always available). */
export function synthesizeAuthorWalkthrough(wb) {
  if (!wb) return "";
  const lines = [];
  lines.push("=== How to win (auto-built from world bible) ===");
  const wcText = winConditionText(wb);
  if (wcText) lines.push(`Win: ${wcText}`);
  if (wb.main_arc) lines.push(`Story: ${wb.main_arc}`);
  if (Array.isArray(wb.objectives) && wb.objectives.length) {
    lines.push("Goals:");
    wb.objectives.forEach((o, i) => lines.push(`  ${i + 1}. ${o}`));
  }
  if (Array.isArray(wb.puzzle_chain) && wb.puzzle_chain.length) {
    lines.push("Steps (in order):");
    for (const s of wb.puzzle_chain) {
      const g = s.gives ? ` → obtain: ${s.gives}` : "";
      const u = s.unlocks != null && s.unlocks !== undefined && s.unlocks !== "" ? ` → unlocks: ${s.unlocks}` : "";
      lines.push(`  ${s.step}. ${s.action}${g}${u}`);
    }
  }
  if (Array.isArray(wb.key_items) && wb.key_items.length) {
    lines.push("Key items:");
    for (const k of wb.key_items) {
      if (k && typeof k.name === "string" && k.name) lines.push(`  - ${k.name}: ${k.purpose || ""}`);
    }
  }
  if (Array.isArray(wb.progression_hints) && wb.progression_hints.length) {
    lines.push(`Hints: ${wb.progression_hints.join(" | ")}`);
  }
  return lines.join("\n");
}

export function validateWorldBible(wb) {
  const issues = [];
  const warnings = [];
  const required = ["objectives", "locations", "key_items", "win_condition"];
  for (const f of required) {
    if (!wb[f] || (Array.isArray(wb[f]) && wb[f].length === 0)) issues.push(`Missing or empty: ${f}`);
  }
  if (wb.locations && wb.locations.length < 3) issues.push(`Too few locations (${wb.locations.length}), need at least 3`);
  if (wb.key_items && wb.key_items.length < 3) issues.push(`Too few items (${wb.key_items.length}), need at least 3`);
  if (wb.key_items) {
    for (const item of wb.key_items) {
      if (typeof item === "object" && !item.purpose) issues.push(`Item '${item.name || "?"}' has no purpose`);
    }
  }

  const roomNames = new Set();
  if (wb.locations) {
    for (const loc of wb.locations) {
      if (typeof loc === "object" && !loc.description) issues.push(`Location '${loc.name || "?"}' has no description`);
      if (loc.name) roomNames.add(loc.name);
    }

    const reachable = new Set();
    for (const loc of wb.locations) {
      if (!Array.isArray(loc.exits) || loc.exits.length === 0) {
        warnings.push(`Location '${loc.name}' has no exits`);
      } else {
        reachable.add(loc.name);
        for (const exit of loc.exits) {
          reachable.add(exit);
          if (!roomNames.has(exit)) warnings.push(`Exit '${exit}' from '${loc.name}' not in locations list`);
        }
      }
    }
    for (const name of roomNames) {
      if (!reachable.has(name)) warnings.push(`Location '${name}' is orphaned (unreachable)`);
    }
  }

  if (Array.isArray(wb.puzzle_chain)) {
    const itemNames = new Set((wb.key_items || []).map(k => (k.name || "").toLowerCase()));
    for (const step of wb.puzzle_chain) {
      if (step.gives && !itemNames.has(step.gives.toLowerCase())) {
        warnings.push(`puzzle_chain step ${step.step}: gives '${step.gives}' not in key_items`);
      }
      if (step.unlocks && roomNames.size && !roomNames.has(step.unlocks)
          && step.unlocks !== "win" && step.unlocks !== null) {
        warnings.push(`puzzle_chain step ${step.step}: unlocks '${step.unlocks}' not a known room`);
      }
    }
  } else {
    warnings.push("No puzzle_chain defined");
  }

  for (const npc of (wb.npcs || [])) {
    if (npc.location && roomNames.size && !roomNames.has(npc.location)) {
      warnings.push(`NPC '${npc.name}' location '${npc.location}' not in locations`);
    }
  }
  for (const mon of (wb.monsters || [])) {
    if (mon.location && roomNames.size && !roomNames.has(mon.location)) {
      warnings.push(`Monster '${mon.name}' location '${mon.location}' not in locations`);
    }
  }

  if (wb.item_locations && roomNames.size) {
    for (const [item, locDesc] of Object.entries(wb.item_locations)) {
      const matchesRoom = [...roomNames].some(rn => locDesc.toLowerCase().includes(rn.toLowerCase()));
      if (!matchesRoom) warnings.push(`item_locations['${item}'] doesn't clearly reference a room`);
    }
  }

  return { valid: issues.length === 0, issues: [...issues, ...warnings] };
}

/** Build synthetic puzzle_chain from items + rooms (mutates wb). Returns fix messages. */
export function buildSyntheticPuzzleChain(wb) {
  const fixes = [];
  const rooms = (wb.locations || []).map(l => l && l.name).filter(Boolean);
  const items = (wb.key_items || []).map(k => k && k.name).filter(Boolean);
  if (items.length < 2 || rooms.length < 2) return fixes;

  const firstRoom = rooms[0];
  const lastRoom = rooms[rooms.length - 1];
  const itemLocs = wb.item_locations = (wb.item_locations && typeof wb.item_locations === "object") ? wb.item_locations : {};
  const intermediateRooms = rooms.slice(0, Math.max(1, rooms.length - 1));
  const itemsForRooms = items.slice();
  const finalItem = itemsForRooms.pop();
  const synth = [];
  let stepNum = 1;

  for (let i = 0; i < itemsForRooms.length; i++) {
    const room = intermediateRooms[Math.min(i, intermediateRooms.length - 1)];
    synth.push({
      step: stepNum++,
      action: `find and take the ${itemsForRooms[i]} in ${room}`,
      gives: itemsForRooms[i],
      unlocks: null,
      location: room,
    });
  }
  if (rooms.length > 2) {
    synth.push({
      step: stepNum++,
      action: `travel deeper through the cavern toward ${lastRoom}`,
      gives: null,
      unlocks: lastRoom,
      location: rooms[rooms.length - 2],
    });
  }
  synth.push({
    step: stepNum++,
    action: `claim the ${finalItem} in ${lastRoom}`,
    gives: finalItem,
    unlocks: "win",
    location: lastRoom,
  });

  itemLocs[finalItem] = lastRoom;
  for (let i = 0; i < itemsForRooms.length; i++) {
    const room = intermediateRooms[Math.min(i, intermediateRooms.length - 1)];
    itemLocs[itemsForRooms[i]] = room;
  }
  wb.puzzle_chain = synth;
  fixes.push(`puzzle_chain: synthetic ${synth.length}-step chain built from ${items.length} items × ${rooms.length} rooms`);
  return fixes;
}

/** True when solvability gaps are chain/win related (hard floor should rebuild chain). */
export function hasChainSolvabilityGaps(gaps) {
  if (!Array.isArray(gaps) || !gaps.length) return false;
  return gaps.some(g => {
    const s = String(g).toLowerCase();
    return s.includes("puzzle_chain") || s.includes("win item") || s.includes("final chain step")
      || s.includes("never given") || s.includes("does not unlock");
  });
}

/** Force synthetic chain rebuild when solvability still fails (mutates wb). */
export function forceSyntheticChainHardFloor(wb) {
  if (!wb) return [];
  wb.puzzle_chain = [];
  const fixes = buildSyntheticPuzzleChain(wb);
  if (normalizeWinCondition(wb)) fixes.push("win_condition normalized after synthetic chain");
  return fixes;
}

/**
 * BFS-based solvability validator. Returns {ok, gaps, report}.
 * Normalizes win_condition in-place before checking.
 */
export function validateWorldBibleSolvability(wb) {
  const gaps = [];
  if (!wb || !Array.isArray(wb.locations) || !wb.locations.length) {
    return { ok: false, gaps: ["no locations"], report: "BLOCKED: no rooms" };
  }
  normalizeWinCondition(wb);

  const rooms = wb.locations.map(l => l && l.name).filter(Boolean);
  const roomSet = new Set(rooms);
  const adj = {};
  for (const loc of wb.locations) {
    if (!loc || !loc.name) continue;
    adj[loc.name] = (loc.exits || []).filter(e => roomSet.has(e));
  }
  const start = rooms[0];
  const reachable = new Set([start]);
  const queue = [start];
  while (queue.length) {
    const r = queue.shift();
    for (const next of (adj[r] || [])) {
      if (!reachable.has(next)) { reachable.add(next); queue.push(next); }
    }
  }
  for (const r of rooms) {
    if (!reachable.has(r)) gaps.push(`unreachable room: ${r}`);
  }
  const winLoc = wb.win_condition && wb.win_condition.required_location;
  if (winLoc && roomSet.has(winLoc) && !reachable.has(winLoc)) {
    gaps.push(`win location not reachable: ${winLoc}`);
  }
  const itemLocs = wb.item_locations || {};
  for (const k of (wb.key_items || [])) {
    if (!k || !k.name) continue;
    if (!itemLocs[k.name]) gaps.push(`key_item '${k.name}' has no placement`);
  }
  for (const step of (wb.puzzle_chain || [])) {
    if (!step || !step.location) continue;
    if (!reachable.has(step.location)) gaps.push(`chain step ${step.step} at unreachable ${step.location}`);
  }
  const chainGives = new Set((wb.puzzle_chain || []).map(s => s && s.gives ? String(s.gives).toLowerCase() : "").filter(Boolean));
  for (const it of ((wb.win_condition && wb.win_condition.required_items) || [])) {
    if (it && !chainGives.has(String(it).toLowerCase())) {
      gaps.push(`win item '${it}' is never given by any chain step`);
    }
  }
  if ((wb.puzzle_chain || []).length < 2) {
    gaps.push(`puzzle_chain has only ${(wb.puzzle_chain || []).length} step(s) — not a playable adventure`);
  }
  const chain = wb.puzzle_chain || [];
  if (chain.length >= 2) {
    const last = chain[chain.length - 1];
    const lastRoom = rooms[rooms.length - 1];
    if (last && last.location && last.location !== lastRoom) {
      gaps.push(`final chain step is in '${last.location}' instead of last room '${lastRoom}'`);
    }
    if (last && last.unlocks !== "win") {
      gaps.push(`final chain step does not unlock "win" (got '${last.unlocks}')`);
    }
  }

  const report = gaps.length
    ? `solvability: ${gaps.length} gap(s)\n  - ` + gaps.join("\n  - ")
    : `solvability: OK (${rooms.length} rooms, ${(wb.key_items || []).length} items, ${(wb.puzzle_chain || []).length} chain steps, all reachable)`;
  return { ok: gaps.length === 0, gaps, report };
}

/** Deep-clone a bible for tests (JSON round-trip). */
export function cloneWorldBible(wb) {
  return JSON.parse(JSON.stringify(wb));
}

/** Prepare bible for validation tests: normalize + auto-repair on a clone. */
export function prepareBibleForValidation(wb) {
  const copy = cloneWorldBible(wb);
  autoRepairWorldBible(copy);
  return copy;
}

/**
 * Deterministic auto-repair — code-only fixes after generation. No LLM round-trip.
 * Returns a list of repairs applied.
 */
export function autoRepairWorldBible(wb) {
  const fixes = [];
  if (!wb || typeof wb !== "object") return fixes;
  const rooms = (wb.locations || []).map(l => l && l.name).filter(Boolean);
  if (!rooms.length) return fixes;
  const firstRoom = rooms[0];
  const lastRoom = rooms[rooms.length - 1];
  const roomSet = new Set(rooms);
  const lowerRoomMap = new Map(rooms.map(r => [r.toLowerCase(), r]));

  for (const npc of (wb.npcs || [])) {
    if (!npc || typeof npc !== "object") continue;
    if (!npc.location || !roomSet.has(npc.location)) {
      const guess = npc.location ? lowerRoomMap.get(String(npc.location).toLowerCase()) : null;
      npc.location = guess || firstRoom;
      fixes.push(`npc '${npc.name || "?"}' relocated → ${npc.location}`);
    }
  }
  for (const mon of (wb.monsters || [])) {
    if (!mon || typeof mon !== "object") continue;
    if (!mon.location || !roomSet.has(mon.location)) {
      const guess = mon.location ? lowerRoomMap.get(String(mon.location).toLowerCase()) : null;
      mon.location = guess || lastRoom;
      fixes.push(`monster '${mon.name || "?"}' relocated → ${mon.location}`);
    }
  }

  const itemLocs = wb.item_locations = (wb.item_locations && typeof wb.item_locations === "object") ? wb.item_locations : {};
  const itemLocsLower = new Map(Object.keys(itemLocs).map(k => [k.toLowerCase(), k]));
  for (const k of (wb.key_items || [])) {
    if (!k || typeof k.name !== "string" || !k.name) continue;
    const nameLower = k.name.toLowerCase();
    if (itemLocsLower.has(nameLower)) continue;
    let placedAt = null;
    for (const step of (wb.puzzle_chain || [])) {
      if (step && step.gives && String(step.gives).toLowerCase() === nameLower && step.location && roomSet.has(step.location)) {
        placedAt = step.location;
        break;
      }
    }
    if (!placedAt) placedAt = (k.name === (wb.win_condition && wb.win_condition.required_items && wb.win_condition.required_items[0])) ? lastRoom : firstRoom;
    itemLocs[k.name] = placedAt;
    fixes.push(`item '${k.name}' placed → ${placedAt}`);
  }

  if (Array.isArray(wb.puzzle_chain)) {
    const before = wb.puzzle_chain.length;
    wb.puzzle_chain = wb.puzzle_chain.filter(s => s && (!s.location || roomSet.has(s.location)));
    const after = wb.puzzle_chain.length;
    if (before !== after) fixes.push(`puzzle_chain: dropped ${before - after} step(s) with invalid location`);
  }

  if (!Array.isArray(wb.puzzle_chain) || wb.puzzle_chain.length < 3) {
    fixes.push(...buildSyntheticPuzzleChain(wb));
  }

  const itemNamesLower = new Set((wb.key_items || []).map(k => k && k.name ? k.name.toLowerCase() : "").filter(Boolean));
  if (Array.isArray(wb.puzzle_chain)) {
    for (const step of wb.puzzle_chain) {
      if (!step) continue;
      if (step.gives && !itemNamesLower.has(String(step.gives).toLowerCase())) {
        fixes.push(`chain step ${step.step}: cleared phantom gives='${step.gives}'`);
        step.gives = null;
      }
    }
  }

  if (normalizeWinCondition(wb)) {
    fixes.push(`win_condition upgraded to structured shape (room=${firstRoom})`);
  }

  return fixes;
}

/** Score a chain candidate so best-of-N can pick the winner. */
/** Normalize room names for fuzzy match (The Foo_Bar → foo bar). */
export function normalizeRoomKey(name) {
  return String(name || "")
    .toLowerCase()
    .replace(/^the\s+/, "")
    .replace(/[_\-]+/g, " ")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** Normalize item names for fuzzy match. */
export function normalizeItemKey(name) {
  return String(name || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/** True when game room and bible room refer to the same place. */
export function roomsMatch(a, b) {
  const ka = normalizeRoomKey(a);
  const kb = normalizeRoomKey(b);
  return ka.length > 0 && ka === kb;
}

/** Snap LLM room name to canonical locations[].name; returns raw if no good match. */
export function snapRoomToBible(rawName, wb) {
  if (!rawName || !wb) return rawName;
  const rooms = (wb.locations || []).map(l => l && l.name).filter(Boolean);
  if (!rooms.length) return rawName;
  if (rooms.includes(rawName)) return rawName;

  const rawKey = normalizeRoomKey(rawName);
  for (const canon of rooms) {
    if (normalizeRoomKey(canon) === rawKey) return canon;
  }

  let best = null;
  let bestScore = 0;
  for (const canon of rooms) {
    const canonKey = normalizeRoomKey(canon);
    if (canonKey.includes(rawKey) || rawKey.includes(canonKey)) {
      const score = Math.min(canonKey.length, rawKey.length) / Math.max(canonKey.length, rawKey.length);
      if (score > bestScore) { bestScore = score; best = canon; }
    }
    const rawTokens = new Set(rawKey.split(" ").filter(Boolean));
    const canonTokens = canonKey.split(" ").filter(Boolean);
    let overlap = 0;
    for (const t of canonTokens) if (rawTokens.has(t)) overlap++;
    if (canonTokens.length) {
      const tokenScore = overlap / Math.max(canonTokens.length, rawTokens.size || 1);
      if (overlap >= 2 && tokenScore > bestScore) { bestScore = tokenScore; best = canon; }
    }
  }
  if (best && bestScore >= 0.55) return best;
  return rawName;
}

/** Snap LLM item name to canonical key_items[].name. */
export function snapItemToBible(rawName, wb) {
  if (!rawName || !wb) return rawName;
  const items = (wb.key_items || []).map(k => k && k.name).filter(Boolean);
  if (!items.length) return rawName;
  if (items.includes(rawName)) return rawName;

  const rawKey = normalizeItemKey(rawName);
  for (const it of items) {
    if (normalizeItemKey(it) === rawKey) return it;
  }
  for (const it of items) {
    const itKey = normalizeItemKey(it);
    if (rawKey.length >= 3 && itKey.includes(rawKey)) return it;
  }
  const rawTokens = rawKey.split(" ").filter(t => t.length >= 3);
  if (rawTokens.length) {
    const matches = items.filter(it => {
      const itKey = normalizeItemKey(it);
      return rawTokens.every(t => itKey.includes(t));
    });
    if (matches.length === 1) return matches[0];
  }
  return rawName;
}

/** Merge duplicate room key (LLM alias) into canonical bible room in game state. */
export function mergeRoomAlias(state, alias, canonical) {
  if (!state || !alias || !canonical || alias === canonical) return;
  if (state.knownMap[alias]) {
    const src = state.knownMap[alias];
    if (!state.knownMap[canonical]) state.knownMap[canonical] = { exits: [], notes: "" };
    const dst = state.knownMap[canonical];
    if (src.notes && !dst.notes) dst.notes = src.notes;
    for (const ex of (src.exits || [])) {
      if (!dst.exits.includes(ex)) dst.exits.push(ex);
    }
    delete state.knownMap[alias];
  }
  for (const entry of Object.values(state.knownMap || {})) {
    if (Array.isArray(entry.exits)) {
      for (let i = 0; i < entry.exits.length; i++) {
        if (entry.exits[i] === alias) entry.exits[i] = canonical;
      }
    }
  }
  if (state.roomItems && state.roomItems[alias]) {
    if (!state.roomItems[canonical]) state.roomItems[canonical] = [];
    for (const it of state.roomItems[alias]) {
      if (!state.roomItems[canonical].includes(it)) state.roomItems[canonical].push(it);
    }
    delete state.roomItems[alias];
  }
  if (state.location === alias) state.location = canonical;
  if (state.roomImageBlobs && state.roomImageBlobs[alias]) {
    state.roomImageBlobs[canonical] = state.roomImageBlobs[alias];
    delete state.roomImageBlobs[alias];
  }
  if (Array.isArray(state.roomsWithImages) && state.roomsWithImages.includes(alias)) {
    state.roomsWithImages = state.roomsWithImages.filter(r => r !== alias);
    if (!state.roomsWithImages.includes(canonical)) state.roomsWithImages.push(canonical);
  }
}

export function scoreChainCandidate(chainObj, rooms, items, firstRoom, lastRoom) {
  if (!chainObj || !Array.isArray(chainObj.puzzle_chain)) return -Infinity;
  const chain = chainObj.puzzle_chain;
  if (!chain.length) return -Infinity;
  const roomSet = new Set(rooms);
  const itemSetLower = new Set(items.map(i => String(i).toLowerCase()));
  let score = 0;
  score += Math.min(chain.length, 8);
  score += chain.filter(s => s && s.location && roomSet.has(s.location)).length;
  score += chain.filter(s => s && (!s.gives || itemSetLower.has(String(s.gives).toLowerCase()))).length;
  if (chain[0] && chain[0].location === firstRoom) score += 5;
  const last = chain[chain.length - 1];
  if (last && last.location === lastRoom) score += 5;
  if (last && last.unlocks === "win") score += 5;
  if (last && last.gives && itemSetLower.has(String(last.gives).toLowerCase())) score += 3;
  return score;
}
