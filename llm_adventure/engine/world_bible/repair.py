"""Deterministic auto-repair and synthetic solution_chain fallback."""

import re
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from engine.world_bible.graph import build_room_graph
from engine.world_bible.win_condition import parse_win_condition

if TYPE_CHECKING:
    from engine.llm import LLMEngine


def synthesize_solution_chain(world_bible: Dict[str, Any]) -> List[str]:
    """CHANGE (solvability May 2026): build a minimal playable chain when LLM omits one."""
    repairs: List[str] = []
    chain = world_bible.get("solution_chain") or []
    if isinstance(chain, list) and len(chain) >= 3:
        return repairs

    loc_names = [
        str((l or {}).get("name", "")).strip()
        for l in (world_bible.get("locations") or [])
        if isinstance(l, dict)
    ]
    loc_names = [n for n in loc_names if n]
    if not loc_names:
        return repairs

    item_names = [
        str((i or {}).get("name", "")).strip()
        for i in (world_bible.get("key_items") or [])
        if isinstance(i, dict)
    ]
    item_names = [n for n in item_names if n]

    parsed = parse_win_condition(world_bible)
    win_loc = parsed.get("required_location") or loc_names[0]
    win_items = parsed.get("required_items") or []
    win_item = win_items[0] if win_items else (item_names[-1] if item_names else None)

    il = world_bible.get("item_locations") or {}
    start = loc_names[0]
    new_chain: List[Dict[str, Any]] = []
    step_n = 1
    prev_req: Optional[str] = None

    for it in item_names:
        if win_item and it == win_item and len(item_names) > 1:
            continue
        room = start
        desc = str(il.get(it, ""))
        for rn in loc_names:
            if rn.lower() in desc.lower():
                room = rn
                break
        new_chain.append({
            "step": step_n,
            "location": room,
            "requires_item": prev_req,
            "blocker": f"obtain {it}",
            "result": f"Pick up {it}",
        })
        prev_req = it
        step_n += 1

    if win_item:
        new_chain.append({
            "step": step_n,
            "location": win_loc,
            "requires_item": win_item,
            "blocker": "win condition",
            "result": parsed.get("description") or "Victory",
        })

    if len(new_chain) >= 2:
        world_bible["solution_chain"] = new_chain
        repairs.append(f"synthetic chain: built {len(new_chain)} steps (LLM chain was too short)")
    return repairs


def auto_repair_world_bible(world_bible: Dict[str, Any]) -> List[str]:
    """Apply local fixes to world_bible IN PLACE. Idempotent on clean bibles."""
    repairs: List[str] = []
    if not isinstance(world_bible, dict):
        return repairs

    locs = world_bible.get("locations")
    if not isinstance(locs, list) or not locs:
        return repairs

    loc_names: List[str] = []
    for loc in locs:
        if isinstance(loc, dict):
            nm = str(loc.get("name") or "").strip()
            if nm:
                loc_names.append(nm)
    if not loc_names:
        return repairs

    loc_set = set(loc_names)
    start_room = loc_names[0]

    key_items = world_bible.get("key_items")
    if not isinstance(key_items, list):
        key_items = []
        world_bible["key_items"] = key_items
    item_names: List[str] = []
    for it in key_items:
        if isinstance(it, dict):
            n = str(it.get("name") or "").strip()
            if n:
                item_names.append(n)
    item_set = set(item_names)

    referenced: Dict[str, str] = {}
    wc = world_bible.get("win_condition")
    if isinstance(wc, dict):
        for it in wc.get("required_items") or []:
            it = str(it).strip()
            if it and it not in item_set:
                referenced.setdefault(it, "Required to win the game")

    chain = world_bible.get("solution_chain") or []
    if isinstance(chain, list):
        for step in chain:
            if not isinstance(step, dict):
                continue
            req = step.get("requires_item")
            if isinstance(req, str) and req.strip() and req.strip() not in item_set:
                referenced.setdefault(req.strip(), f"Used at {step.get('location','?')}")

    for m in world_bible.get("monsters", []) or []:
        if isinstance(m, dict):
            wk = str(m.get("weakness") or "").strip()
            if wk and len(wk) < 40 and wk not in item_set and re.match(r"^[A-Za-z][A-Za-z\s'\-]+$", wk):
                referenced.setdefault(wk, f"Defeats {m.get('name','enemy')}")

    for rid in world_bible.get("riddles", []) or []:
        if isinstance(rid, dict):
            rew = str(rid.get("reward") or "").strip()
            if rew and len(rew) < 40 and rew not in item_set and re.match(r"^[A-Za-z][A-Za-z\s'\-]+$", rew):
                referenced.setdefault(rew, f"Reward of riddle at {rid.get('location','?')}")

    for new_item, purpose in referenced.items():
        key_items.append({"name": new_item, "purpose": purpose})
        item_set.add(new_item)
        item_names.append(new_item)
        repairs.append(f"added missing key_item '{new_item}' ({purpose})")

    item_locations = world_bible.get("item_locations")
    if not isinstance(item_locations, dict):
        item_locations = {}
        world_bible["item_locations"] = item_locations

    chain_loc_for_item: Dict[str, Optional[str]] = {}
    if isinstance(chain, list):
        for step in chain:
            if not isinstance(step, dict):
                continue
            loc_at = str(step.get("location") or "").strip()
            loc_at = loc_at if loc_at in loc_set else None
            req = step.get("requires_item")
            if isinstance(req, str) and req.strip():
                chain_loc_for_item.setdefault(req.strip(), loc_at)
            res_text = str(step.get("result") or "")
            for it in item_names:
                if it and it.lower() in res_text.lower():
                    chain_loc_for_item.setdefault(it, loc_at)

    for it in item_names:
        if it not in item_locations:
            target_loc = chain_loc_for_item.get(it) or start_room
            item_locations[it] = f"{target_loc}: placed by auto-repair"
            repairs.append(f"placed item '{it}' at '{target_loc}' (was missing from item_locations)")
        else:
            loc_desc = str(item_locations[it])
            if not any(rn.lower() in loc_desc.lower() for rn in loc_set):
                target_loc = chain_loc_for_item.get(it) or start_room
                item_locations[it] = f"{target_loc}: {loc_desc}"
                repairs.append(f"item_locations['{it}']: prefixed with real room '{target_loc}'")

    sym: Dict[str, Set[str]] = {n: set() for n in loc_set}
    for loc in locs:
        if not isinstance(loc, dict):
            continue
        nm = str(loc.get("name") or "").strip()
        if not nm:
            continue
        for ex in loc.get("exits") or []:
            if isinstance(ex, str) and ex.strip() in loc_set:
                sym[nm].add(ex.strip())
                sym[ex.strip()].add(nm)

    seen: Set[str] = {start_room}
    stack = [start_room]
    while stack:
        cur = stack.pop()
        for nxt in sym.get(cur, ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)

    if seen != loc_set:
        unreachable = [n for n in loc_names if n not in seen]
        for room in unreachable:
            idx = loc_names.index(room)
            anchor = loc_names[idx - 1] if idx > 0 else start_room
            for loc in locs:
                if not isinstance(loc, dict):
                    continue
                nm = str(loc.get("name") or "").strip()
                if nm == anchor:
                    ex_list = loc.setdefault("exits", [])
                    if isinstance(ex_list, list) and room not in ex_list:
                        ex_list.append(room)
                if nm == room:
                    ex_list = loc.setdefault("exits", [])
                    if isinstance(ex_list, list) and anchor not in ex_list:
                        ex_list.append(anchor)
            sym[anchor].add(room)
            sym[room].add(anchor)
            repairs.append(f"auto-connected '{anchor}' <-> '{room}' (room was unreachable from start)")

    for who_key in ("npcs", "monsters", "riddles"):
        for entry in world_bible.get(who_key, []) or []:
            if not isinstance(entry, dict):
                continue
            loc_at = str(entry.get("location") or "").strip()
            if loc_at and loc_at not in loc_set:
                target = start_room
                ent_name = str(entry.get("name") or entry.get("hint") or "").lower()
                if ent_name and isinstance(chain, list):
                    for step in chain:
                        if not isinstance(step, dict):
                            continue
                        if (
                            ent_name in str(step.get("blocker", "")).lower()
                            or ent_name in str(step.get("result", "")).lower()
                        ):
                            sl = str(step.get("location") or "").strip()
                            if sl in loc_set:
                                target = sl
                                break
                entry["location"] = target
                label = who_key[:-1]
                repairs.append(
                    f"{label} '{entry.get('name','?')}' moved to '{target}' (was unknown room '{loc_at}')"
                )

    if isinstance(wc, dict):
        rl = wc.get("required_location")
        if rl and rl not in loc_set:
            wc["required_location"] = start_room
            repairs.append(f"win_condition.required_location '{rl}' set to start room '{start_room}'")
        ri = wc.get("required_items") or []
        if isinstance(ri, list):
            cleaned_ri = [x for x in ri if isinstance(x, str) and x.strip() in item_set]
            if cleaned_ri != ri:
                wc["required_items"] = cleaned_ri
                repairs.append("win_condition.required_items pruned to items present in key_items")

    if isinstance(chain, list):
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue
            sl = str(step.get("location") or "").strip()
            if sl and sl not in loc_set:
                step["location"] = start_room
                repairs.append(f"solution_chain[{i}].location '{sl}' set to '{start_room}' (unknown room)")
            req = step.get("requires_item")
            if isinstance(req, str) and req.strip() and req.strip() not in item_set:
                step["requires_item"] = None
                repairs.append(f"solution_chain[{i}].requires_item '{req}' cleared (not in key_items)")

    repairs.extend(synthesize_solution_chain(world_bible))
    world_bible.pop("_win_parsed", None)
    return repairs


def _ask_one_choice(
    wb_engine: "LLMEngine", system_msg: str, question: str, choices: List[str]
) -> Optional[str]:
    from engine.parse import strip_llm_thinking_blocks

    if not choices:
        return None
    prompt = (
        f"{question}\n\n"
        "Pick exactly ONE option from this list. Reply with the chosen text on a single line, "
        "with no quotes, no JSON, no explanation, no markdown — just the option text.\n"
        "Options:\n"
        + "\n".join(f"  - {c}" for c in choices)
    )
    resp = wb_engine.generate(system_msg, prompt) or ""
    cleaned = strip_llm_thinking_blocks(resp).strip().strip("`'\" \n")
    if not cleaned:
        return None
    first_line = cleaned.splitlines()[0].strip().strip("`'\" -*")
    if not first_line:
        return None
    for c in choices:
        if c == first_line:
            return c
    fl = first_line.lower()
    for c in choices:
        if c.lower() == fl:
            return c
    for c in choices:
        if c.lower() in fl or fl in c.lower():
            return c
    return None


def micro_repair_world_bible(
    world_bible: Dict[str, Any], wb_engine: "LLMEngine", system_msg: str
) -> List[str]:
    """Targeted LLM micro-repairs for thematic gaps."""
    repairs: List[str] = []
    if not isinstance(world_bible, dict):
        return repairs

    loc_names = [
        str((l or {}).get("name", "")).strip()
        for l in (world_bible.get("locations") or [])
        if isinstance(l, dict)
    ]
    loc_names = [l for l in loc_names if l]
    loc_set = set(loc_names)

    item_names = [
        str((i or {}).get("name", "")).strip()
        for i in (world_bible.get("key_items") or [])
        if isinstance(i, dict)
    ]
    item_names = [n for n in item_names if n]

    h_placeholders = set()
    for ki in world_bible.get("key_items", []) or []:
        if not isinstance(ki, dict):
            continue
        purpose = str(ki.get("purpose") or "")
        if purpose.startswith("Defeats ") or purpose.startswith("Reward of "):
            n = str(ki.get("name") or "").strip()
            if n:
                h_placeholders.add(n)

    real_items = [n for n in item_names if n not in h_placeholders]

    for m in world_bible.get("monsters", []) or []:
        if not isinstance(m, dict):
            continue
        wk = str(m.get("weakness") or "").strip()
        if wk and wk in h_placeholders and real_items:
            question = (
                f"Monster '{m.get('name','?')}' (in {m.get('location','?')}) currently has "
                f"weakness '{wk}', which was auto-generated and is not a real game item. "
                "Choose the most thematic existing item the player would use to defeat this monster."
            )
            chosen = _ask_one_choice(wb_engine, system_msg, question, real_items)
            if chosen and chosen != wk:
                m["weakness"] = chosen
                world_bible["key_items"] = [
                    ki for ki in world_bible.get("key_items", []) or []
                    if not (isinstance(ki, dict) and ki.get("name") == wk)
                ]
                repairs.append(
                    f"micro: monster '{m.get('name','?')}' weakness '{wk}' → existing item '{chosen}'"
                )
                if wk in item_names:
                    item_names.remove(wk)
                h_placeholders.discard(wk)
                real_items = [n for n in item_names if n not in h_placeholders]

    chain = world_bible.get("solution_chain") or []
    if isinstance(chain, list):
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                continue
            if step.get("requires_item") is None:
                blk = str(step.get("blocker") or "").strip()
                sl = str(step.get("location") or "?")
                if blk and len(real_items) >= 2:
                    question = (
                        f"In solution_chain step {i+1} at room '{sl}', the player faces blocker "
                        f"'{blk}'. Choose ONE existing item the player would need to overcome this "
                        "blocker, or pick 'NONE' if no item is needed."
                    )
                    chosen = _ask_one_choice(wb_engine, system_msg, question, real_items + ["NONE"])
                    if chosen and chosen != "NONE":
                        step["requires_item"] = chosen
                        repairs.append(f"micro: solution_chain[{i}].requires_item set to '{chosen}'")

    for loc in world_bible.get("locations", []) or []:
        if not isinstance(loc, dict):
            continue
        nm = str(loc.get("name") or "").strip()
        ex_list = loc.get("exits")
        if not isinstance(ex_list, list):
            continue
        new_list: List[str] = []
        for ex in ex_list:
            if not isinstance(ex, str):
                continue
            ex_s = ex.strip()
            if not ex_s:
                continue
            if ex_s in loc_set:
                if ex_s not in new_list:
                    new_list.append(ex_s)
                continue
            others = [l for l in loc_names if l != nm and l not in new_list]
            if not others:
                continue
            question = (
                f"Room '{nm}' lists an exit to '{ex_s}', but '{ex_s}' is not a real room. "
                "Choose ONE real room name to replace it."
            )
            chosen = _ask_one_choice(wb_engine, system_msg, question, others)
            if chosen and chosen not in new_list:
                new_list.append(chosen)
                repairs.append(f"micro: exit '{nm}' → '{ex_s}' replaced with '{chosen}'")
        if new_list != ex_list:
            loc["exits"] = new_list

    world_bible.pop("_win_parsed", None)
    return repairs
