"""World Bible schema and solvability validation."""

from typing import Any, Dict, List, Tuple

from engine.world_bible.graph import build_room_graph, collect_key_item_names
from engine.world_bible.win_condition import parse_win_condition


def validate_world_bible(world_bible: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate that a world bible creates a playable, non-trivial game."""
    issues: List[str] = []

    required_fields = ["objectives", "locations", "key_items", "win_condition"]
    for field in required_fields:
        if field not in world_bible or not world_bible[field]:
            issues.append(f"Missing or empty required field: {field}")

    wc = world_bible.get("win_condition")
    if wc is not None:
        if isinstance(wc, dict):
            if not wc.get("required_items") and not wc.get("required_location"):
                issues.append("win_condition has no required_items and no required_location — not winnable")
        elif isinstance(wc, str):
            if not wc.strip():
                issues.append("win_condition is an empty string")
        else:
            issues.append(f"win_condition must be a dict or string, got {type(wc).__name__}")

    if "locations" in world_bible:
        if len(world_bible["locations"]) < 5:
            issues.append(
                f"Too few locations ({len(world_bible['locations'])}), need at least 5 for interesting gameplay"
            )

    if "key_items" in world_bible:
        if len(world_bible["key_items"]) < 4:
            issues.append(
                f"Too few items ({len(world_bible['key_items'])}), need at least 4 for puzzle solving"
            )

    if "progression_hints" in world_bible:
        if len(world_bible.get("progression_hints", [])) < 2:
            issues.append("Not enough progression hints for player guidance")

    if "key_items" in world_bible:
        for item in world_bible["key_items"]:
            if isinstance(item, dict) and not item.get("purpose"):
                issues.append(f"Item '{item.get('name', 'unknown')}' has no purpose defined")

    if "locations" in world_bible:
        for loc in world_bible["locations"]:
            if isinstance(loc, dict) and not loc.get("description"):
                issues.append(f"Location '{loc.get('name', 'unknown')}' has no description")

    return len(issues) == 0, issues


def _item_placed_at_room(item_locations: Dict[str, Any], item: str, room: str, loc_set: set) -> bool:
    if item not in item_locations:
        return False
    loc_desc = str(item_locations[item])
    return room.lower() in loc_desc.lower() or any(
        rn.lower() in loc_desc.lower() for rn in loc_set if rn.lower() == room.lower()
    )


def _validate_ordered_item_acquisition(
    world_bible: Dict[str, Any], item_names: List[str], loc_set: set, sym: Dict[str, set]
) -> List[str]:
    """CHANGE (solvability May 2026): simulate chain order — required items must be
    obtainable at or before the step that needs them."""
    gaps: List[str] = []
    chain = world_bible.get("solution_chain") or []
    item_locations = world_bible.get("item_locations") or {}
    if not chain or not isinstance(chain, list):
        return gaps

    item_set = set(item_names)
    held: set = set()
    prev_loc: str = ""

    for i, step in enumerate(chain):
        if not isinstance(step, dict):
            continue
        sl = str(step.get("location") or "").strip()
        req = step.get("requires_item")
        if isinstance(req, str) and req.strip():
            req = req.strip()
            if req not in item_set:
                continue
            if req not in held:
                if not _item_placed_at_room(item_locations, req, sl, loc_set):
                    gaps.append(
                        f"solution_chain[{i}] needs '{req}' at '{sl}' but item_locations "
                        f"does not place it at or before that room"
                    )
                else:
                    held.add(req)
        # Items given by step result text
        res = str(step.get("result") or "")
        for it in item_names:
            if it.lower() in res.lower():
                held.add(it)
        prev_loc = sl or prev_loc

    return gaps


def validate_world_bible_solvability(world_bible: Dict[str, Any]) -> Tuple[bool, List[str], str]:
    """Build room graph, check placements, chain, and ordered item acquisition."""
    gaps: List[str] = []
    report: List[str] = []
    if not isinstance(world_bible, dict):
        return False, ["world_bible is not a dict"], "world_bible is not a dict"

    loc_names, sym, seen, graph_gaps = build_room_graph(world_bible)
    gaps.extend(graph_gaps)
    loc_set = set(loc_names)
    if not loc_set:
        return False, gaps or ["no usable locations"], "no usable locations"

    start = loc_names[0]
    item_names = collect_key_item_names(world_bible)
    item_set = set(item_names)

    item_locations = world_bible.get("item_locations") or {}
    placed_count = 0
    for it in item_names:
        if it not in item_locations:
            gaps.append(f"key_item '{it}' has no entry in item_locations (player cannot obtain it)")
            continue
        loc_desc = str(item_locations[it])
        if any(rn.lower() in loc_desc.lower() for rn in loc_set):
            placed_count += 1
        else:
            gaps.append(f"item_locations['{it}'] = '{loc_desc}' references no known room")

    parsed_win = parse_win_condition(world_bible)
    win_loc = parsed_win.get("required_location")
    win_items = parsed_win.get("required_items") or []
    if win_loc and win_loc not in loc_set:
        gaps.append(f"win_condition.required_location '{win_loc}' is not a known room")
    elif win_loc and win_loc not in seen:
        gaps.append(f"win_condition.required_location '{win_loc}' not reachable from start '{start}'")
    for it in win_items:
        if it not in item_set:
            gaps.append(f"win_condition.required_items references '{it}' missing from key_items")
    if not win_items and not win_loc:
        gaps.append("win_condition has no required_items and no required_location — cannot determine victory")

    for who_key, label in (("npcs", "npc"), ("monsters", "monster"), ("riddles", "riddle")):
        for entry in world_bible.get(who_key, []) or []:
            if not isinstance(entry, dict):
                continue
            loc_at = str(entry.get("location") or "").strip()
            who = str(entry.get("name") or entry.get("hint") or "?")
            if loc_at and loc_at not in loc_set:
                gaps.append(f"{label} '{who}' lives in unknown room '{loc_at}'")

    for m in world_bible.get("monsters", []) or []:
        if not isinstance(m, dict):
            continue
        wk = str(m.get("weakness") or "").lower()
        if wk and not any(it.lower() in wk for it in item_set):
            gaps.append(
                f"monster '{m.get('name', '?')}' weakness '{m.get('weakness')}' "
                "does not reference any key_item — player has no defined way to defeat it"
            )

    chain = world_bible.get("solution_chain") or []
    if not chain:
        gaps.append("no solution_chain provided — cannot verify the puzzle is winnable")
    else:
        prev_loc: str = ""
        for i, step in enumerate(chain):
            if not isinstance(step, dict):
                gaps.append(f"solution_chain[{i}] is not an object")
                continue
            sl = str(step.get("location") or "").strip()
            req = step.get("requires_item")
            if sl and sl not in loc_set:
                gaps.append(f"solution_chain[{i}].location '{sl}' is unknown")
            if isinstance(req, str) and req and req not in item_set:
                gaps.append(f"solution_chain[{i}].requires_item '{req}' missing from key_items")
            if prev_loc and sl in loc_set and prev_loc in loc_set and sl != prev_loc:
                sub = {prev_loc}
                st = [prev_loc]
                while st:
                    cu = st.pop()
                    for nx in sym.get(cu, ()):
                        if nx not in sub:
                            sub.add(nx)
                            st.append(nx)
                if sl not in sub:
                    gaps.append(f"solution_chain[{i}] jumps from '{prev_loc}' to '{sl}' with no path")
            prev_loc = sl
        if win_loc and isinstance(chain[-1], dict):
            last_loc = str(chain[-1].get("location") or "").strip()
            if last_loc and last_loc != win_loc:
                gaps.append(f"solution_chain ends at '{last_loc}' but win_condition requires '{win_loc}'")

    gaps.extend(_validate_ordered_item_acquisition(world_bible, item_names, loc_set, sym))

    report.append(f"[solvability] rooms={len(loc_set)} reachable_from_'{start}'={len(seen)}/{len(loc_set)}")
    report.append(f"[solvability] items={len(item_set)} placed_in_known_rooms={placed_count}/{len(item_set)}")
    if win_loc:
        report.append(
            f"[solvability] win_location='{win_loc}' reachable={'yes' if win_loc in seen else 'NO'}"
        )
    report.append(f"[solvability] solution_chain steps={len(chain)}")
    if gaps:
        report.append(f"[solvability] gaps ({len(gaps)}):")
        for g in gaps[:25]:
            report.append(f"  - {g}")
        if len(gaps) > 25:
            report.append(f"  - … {len(gaps) - 25} more")
    else:
        report.append("[solvability] gaps: none — looks solvable")
    return (len(gaps) == 0), gaps, "\n".join(report)
