/**
 * Shared world-bible unit tests — runs in browser (debug panel) and Node (CI).
 */
import {
  extractFirstJson,
  extractMapSkeleton,
  validateWorldBible,
  validateWorldBibleSolvability,
  autoRepairWorldBible,
  prepareBibleForValidation,
  cloneWorldBible,
  scoreChainCandidate,
  normalizeWinCondition,
  hasChainSolvabilityGaps,
  snapRoomToBible,
  snapItemToBible,
  roomsMatch,
  mergeRoomAlias,
} from "./world_bible_logic.mjs";

/** Known fixture paths (relative to browser_adventure/). */
export const FIXTURE_PATHS = [
  "default_cave.json",
];

function mkMinimalBible(overrides = {}) {
  return {
    objectives: ["Explore", "Find key", "Win"],
    locations: [
      { name: "Start", description: "Entry", exits: ["Mid"] },
      { name: "Mid", description: "Middle", exits: ["Start", "End"] },
      { name: "End", description: "Treasure room", exits: ["Mid"] },
    ],
    key_items: [
      { name: "lantern", purpose: "lights dark paths" },
      { name: "key", purpose: "opens the vault" },
      { name: "gem", purpose: "the treasure" },
    ],
    item_locations: { lantern: "Start", key: "Mid", gem: "End" },
    puzzle_chain: [],
    win_condition: "Get the gem",
    main_arc: "A short test adventure",
    ...overrides,
  };
}

function runCase(name, fn) {
  try {
    const detail = fn();
    return { name, pass: true, detail: detail || "" };
  } catch (e) {
    return { name, pass: false, detail: e.message || String(e) };
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || "assertion failed");
}

/**
 * Run all regression + live tests.
 * @param {{ getActiveBible?: () => object|null, fetchFixture?: (path: string) => Promise<object>, sourceLabel?: string }} opts
 * @returns {Promise<{ regression: object[], live: object[], elapsedMs: number, regressionPass: number, regressionTotal: number, livePass: number, liveTotal: number }>}
 */
export async function runWorldBibleTests(opts = {}) {
  const t0 = performance.now();
  const regression = [];
  const getActiveBible = opts.getActiveBible || (() => null);
  const fetchFixture = opts.fetchFixture || (async () => { throw new Error("fetchFixture not provided"); });

  // ── Fixture tests ──
  for (const path of FIXTURE_PATHS) {
    regression.push(await (async () => {
      const name = `fixture: ${path}`;
      try {
        const raw = await fetchFixture(path);
        const prepared = prepareBibleForValidation(raw);
        const { valid } = validateWorldBible(prepared);
        assert(valid, "schema invalid after repair");
        const sol = validateWorldBibleSolvability(prepared);
        assert(sol.ok, sol.report);
        return { name, pass: true, detail: sol.report.split("\n")[0] };
      } catch (e) {
        return { name, pass: false, detail: e.message || String(e) };
      }
    })());
  }

  // ── Inline regression tests ──
  regression.push(runCase("orphan room detected", () => {
    const wb = mkMinimalBible({
      locations: [
        { name: "Start", description: "Entry", exits: ["Mid"] },
        { name: "Mid", description: "Middle", exits: ["Start"] },
        { name: "Orphan", description: "Isolated", exits: [] },
      ],
    });
    autoRepairWorldBible(wb);
    const sol = validateWorldBibleSolvability(wb);
    assert(!sol.ok, "expected solvability failure");
    assert(sol.gaps.some(g => g.includes("unreachable room: Orphan")), "expected orphan gap");
    return "unreachable Orphan reported";
  }));

  regression.push(runCase("empty chain → synthetic chain", () => {
    const wb = mkMinimalBible({ puzzle_chain: [] });
    const fixes = autoRepairWorldBible(wb);
    assert(wb.puzzle_chain.length >= 3, `chain too short: ${wb.puzzle_chain.length}`);
    assert(fixes.some(f => f.includes("synthetic")), "expected synthetic chain fix");
    const sol = validateWorldBibleSolvability(wb);
    assert(sol.ok, sol.report);
    return `${wb.puzzle_chain.length} steps, solvability OK`;
  }));

  regression.push(runCase("extractFirstJson fenced block", () => {
    const raw = 'Here is the plan:\n```json\n{"npcs":[{"name":"Sage"}]}\n```\nDone.';
    const obj = extractFirstJson(raw);
    assert(obj && Array.isArray(obj.npcs) && obj.npcs[0].name === "Sage", "parse failed");
    return "parsed fenced JSON";
  }));

  regression.push(runCase("extractMapSkeleton finds rooms", () => {
    const raw = 'Sure! {"rooms":[{"name":"Hall","exits":["Vault"]},{"name":"Vault","exits":["Hall"]}]}';
    const sk = extractMapSkeleton(raw, null);
    assert(sk && sk.rooms && sk.rooms.length === 2, "skeleton missing rooms");
    return "2 rooms extracted";
  }));

  regression.push(runCase("scoreChainCandidate picks complete chain", () => {
    const rooms = ["A", "B", "C"];
    const items = ["torch", "key", "gem"];
    const bad = { puzzle_chain: [{ step: 1, action: "x", gives: "torch", location: "A" }] };
    const good = {
      puzzle_chain: [
        { step: 1, action: "start", gives: "torch", location: "A" },
        { step: 2, action: "win", gives: "gem", unlocks: "win", location: "C" },
      ],
    };
    const scoreBad = scoreChainCandidate(bad, rooms, items, "A", "C");
    const scoreGood = scoreChainCandidate(good, rooms, items, "A", "C");
    assert(scoreGood > scoreBad, `good=${scoreGood} bad=${scoreBad}`);
    return `good=${scoreGood} > bad=${scoreBad}`;
  }));

  regression.push(runCase("win item gap detected", () => {
    const wb = mkMinimalBible({
      puzzle_chain: [
        { step: 1, action: "get lantern", gives: "lantern", location: "Start" },
        { step: 2, action: "get key", gives: "key", location: "Mid" },
        { step: 3, action: "reach end", gives: null, unlocks: "win", location: "End" },
      ],
      win_condition: { required_items: ["gem"], required_location: "Start", description: "Get gem" },
    });
    // Test detection only — do not auto-repair (repair would rebuild chain).
    const sol = validateWorldBibleSolvability(wb);
    assert(!sol.ok, "expected failure");
    assert(hasChainSolvabilityGaps(sol.gaps), "expected chain solvability gaps");
    return sol.gaps.find(g => g.includes("win item")) || sol.gaps[0];
  }));

  regression.push(runCase("normalizeWinCondition from string", () => {
    const wb = mkMinimalBible({
      win_condition: "Obtain the gem and return",
      puzzle_chain: [
        { step: 1, action: "take gem", gives: "gem", unlocks: "win", location: "End" },
        { step: 2, action: "return", gives: null, unlocks: null, location: "Start" },
      ],
    });
    autoRepairWorldBible(wb);
    assert(typeof wb.win_condition === "object", "win_condition not object");
    assert(Array.isArray(wb.win_condition.required_items), "missing required_items");
    return `required_items: ${wb.win_condition.required_items.join(", ")}`;
  }));

  regression.push(runCase("snapRoomToBible underscore alias", () => {
    const wb = {
      locations: [
        { name: "The Threshold of Gloomfang", exits: ["The Whispering Antechamber"] },
        { name: "The Whispering Antechamber", exits: ["The Threshold of Gloomfang"] },
      ],
    };
    assert(snapRoomToBible("Threshold_of_Gloomfang", wb) === "The Threshold of Gloomfang");
    assert(snapRoomToBible("Whispering_Antechamber", wb) === "The Whispering Antechamber");
    assert(roomsMatch("Threshold_of_Gloomfang", "The Threshold of Gloomfang"));
    return "underscore aliases snapped";
  }));

  regression.push(runCase("snapItemToBible partial name", () => {
    const wb = {
      key_items: [
        { name: "Mithril-threaded Rope", purpose: "traverse chasms" },
        { name: "Elven Sightstone", purpose: "see hidden paths" },
      ],
    };
    assert(snapItemToBible("rope", wb) === "Mithril-threaded Rope");
    assert(snapItemToBible("mithril threaded rope", wb) === "Mithril-threaded Rope");
    return "rope → Mithril-threaded Rope";
  }));

  regression.push(runCase("mergeRoomAlias consolidates state", () => {
    const state = {
      location: "Threshold_of_Gloomfang",
      knownMap: {
        Threshold_of_Gloomfang: { exits: ["Whispering_Antechamber"], notes: "entry" },
        "The Threshold of Gloomfang": { exits: [], notes: "" },
      },
      roomItems: {
        Threshold_of_Gloomfang: ["Mithril-threaded Rope"],
        "The Threshold of Gloomfang": [],
      },
      roomImageBlobs: {},
      roomsWithImages: [],
    };
    mergeRoomAlias(state, "Threshold_of_Gloomfang", "The Threshold of Gloomfang");
    assert(state.location === "The Threshold of Gloomfang");
    assert(!state.knownMap.Threshold_of_Gloomfang);
    assert(state.roomItems["The Threshold of Gloomfang"].includes("Mithril-threaded Rope"));
    return "alias merged into canonical room";
  }));

  // ── Live tests on active bible ──
  const live = [];
  const active = getActiveBible();
  const sourceLabel = opts.sourceLabel || "activeWorldBible";

  if (active && typeof active === "object") {
    const wb = cloneWorldBible(active);
    autoRepairWorldBible(wb);

    live.push(runCase(`live: schema valid (${sourceLabel})`, () => {
      const { valid, issues } = validateWorldBible(wb);
      assert(valid, issues.join("; "));
      return "valid";
    }));

    live.push(runCase(`live: solvability (${sourceLabel})`, () => {
      const sol = validateWorldBibleSolvability(wb);
      assert(sol.ok, sol.report);
      return sol.report.split("\n")[0];
    }));

    live.push(runCase(`live: chain length >= 2 (${sourceLabel})`, () => {
      const n = (wb.puzzle_chain || []).length;
      assert(n >= 2, `only ${n} step(s)`);
      return `${n} steps`;
    }));

    live.push(runCase(`live: no unreachable rooms (${sourceLabel})`, () => {
      const sol = validateWorldBibleSolvability(wb);
      const unreachable = sol.gaps.filter(g => g.includes("unreachable room"));
      assert(!unreachable.length, unreachable.join("; "));
      return "all rooms reachable";
    }));
  } else {
    live.push({ name: `live: (${sourceLabel})`, pass: false, detail: "no active world bible loaded" });
  }

  const elapsedMs = Math.round(performance.now() - t0);
  const regressionPass = regression.filter(r => r.pass).length;
  const livePass = live.filter(r => r.pass).length;

  return {
    regression,
    live,
    elapsedMs,
    regressionPass,
    regressionTotal: regression.length,
    livePass,
    liveTotal: live.length,
  };
}

/** Format test results for the debug panel. */
export function formatTestResults(results) {
  const lines = ["── Unit tests ──"];
  lines.push(`Regression: ${results.regressionPass}/${results.regressionTotal} passed (${results.elapsedMs} ms)`);
  for (const r of results.regression) {
    lines.push(`  ${r.pass ? "✓" : "✗"} ${r.name}${r.detail ? ` — ${r.detail}` : ""}`);
  }
  lines.push(`Live: ${results.livePass}/${results.liveTotal} passed`);
  for (const r of results.live) {
    lines.push(`  ${r.pass ? "✓" : "✗"} ${r.name}${r.detail ? ` — ${r.detail}` : ""}`);
  }
  return lines;
}
