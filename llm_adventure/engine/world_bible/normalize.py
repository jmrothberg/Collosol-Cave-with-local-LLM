"""Load and normalize default world bible from browser_adventure."""

import json
import os
from typing import Any, Dict, List, Optional

from config.paths import SCRIPT_DIR
from engine.bible_store import get_world_bible, set_world_bible
from engine.world_bible.repair import auto_repair_world_bible
from engine.world_bible.win_condition import parse_win_condition


def normalize_browser_chain(wb: Dict[str, Any]) -> None:
    """Convert browser puzzle_chain steps to engine solution_chain schema."""
    chain = wb.get("solution_chain") or []
    if not chain or not isinstance(chain, list):
        return
    if isinstance(chain[0], dict) and "location" in chain[0]:
        return

    loc_names = [
        str(l.get("name") or "").strip()
        for l in (wb.get("locations") or [])
        if isinstance(l, dict)
    ]
    loc_names = [n for n in loc_names if n]

    item_names = [
        str(i.get("name") or "").strip()
        for i in (wb.get("key_items") or [])
        if isinstance(i, dict)
    ]
    item_names = [n for n in item_names if n]

    entity_loc: Dict[str, str] = {}
    for e in (wb.get("npcs") or []) + (wb.get("monsters") or []):
        if isinstance(e, dict):
            nm = str(e.get("name") or "").strip()
            lc = str(e.get("location") or "").strip()
            if nm and lc:
                entity_loc[nm.lower()] = lc

    item_loc: Dict[str, str] = {}
    for item, desc in (wb.get("item_locations") or {}).items():
        s = str(desc)
        for ln in loc_names:
            if ln and ln in s:
                item_loc[item.lower()] = ln
                break

    new_chain: List[Dict[str, Any]] = []
    for i, step in enumerate(chain):
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or step.get("result") or "")
        gives = step.get("gives")
        unlocks = step.get("unlocks")
        action_l = action.lower()

        loc: Optional[str] = None
        for ln in loc_names:
            if ln and ln.lower() in action_l:
                loc = ln
                break
        if not loc:
            for nm, ll in entity_loc.items():
                if nm and nm in action_l:
                    loc = ll
                    break
        if not loc and isinstance(gives, str) and gives.lower() in item_loc:
            loc = item_loc[gives.lower()]
        if not loc and loc_names:
            loc = loc_names[0]

        gives_l = gives.lower() if isinstance(gives, str) else ""
        req: Optional[str] = None
        for it in item_names:
            if it and it.lower() != gives_l and it.lower() in action_l:
                req = it
                break

        new_chain.append({
            "step": step.get("step", i + 1),
            "location": loc,
            "requires_item": req,
            "blocker": str(unlocks or ""),
            "result": (str(gives) if gives else action),
        })

    wb["solution_chain"] = new_chain


def load_default_world_bible_if_needed() -> bool:
    """Load browser_adventure/default_cave.json when no bible is set."""
    if isinstance(get_world_bible(), dict) and get_world_bible():
        return False
    candidates = [
        os.path.join(SCRIPT_DIR, "default_cave.json"),
        os.path.join(SCRIPT_DIR, "..", "browser_adventure", "default_cave.json"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "browser_adventure", "default_cave.json"),
    ]
    for raw in candidates:
        path = os.path.normpath(raw)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                wb = json.load(f)
        except Exception as e:
            print(f"[default_cave] failed to load {path}: {e}")
            continue
        if "puzzle_chain" in wb and "solution_chain" not in wb:
            wb["solution_chain"] = wb.pop("puzzle_chain")
        normalize_browser_chain(wb)
        try:
            parse_win_condition(wb)
        except Exception:
            pass
        try:
            repairs = auto_repair_world_bible(wb)
            if repairs:
                print(f"[default_cave] auto-repair applied {len(repairs)} fix(es):")
                for r in repairs[:6]:
                    print(f"  - {r}")
        except Exception:
            pass
        set_world_bible(wb)
        rooms = len(wb.get("locations", []) or [])
        items = len(wb.get("key_items", []) or [])
        chain_steps = len(wb.get("solution_chain", []) or [])
        print(f"[default_cave] Loaded {path} ({rooms} rooms, {items} key items, {chain_steps} chain steps)")
        return True
    print("[default_cave] No default_cave.json found.")
    return False
