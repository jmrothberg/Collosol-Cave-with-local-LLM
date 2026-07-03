/**
 * Node test runner wrapper — same tests as the in-browser debug panel button.
 * Run: node --test browser_adventure/tests/world_bible_logic.test.mjs
 */
import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { runWorldBibleTests } from "../world_bible_tests.mjs";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

async function fetchFixture(path) {
  const text = await readFile(join(ROOT, path), "utf8");
  return JSON.parse(text);
}

describe("world_bible_logic", () => {
  it("regression + live (no active bible)", async () => {
    const results = await runWorldBibleTests({
      getActiveBible: () => null,
      fetchFixture,
      sourceLabel: "node-ci",
    });
    assert.equal(results.regressionPass, results.regressionTotal,
      `regression failures:\n${results.regression.filter(r => !r.pass).map(r => `${r.name}: ${r.detail}`).join("\n")}`);
    assert.ok(results.liveTotal >= 1);
  });
});
