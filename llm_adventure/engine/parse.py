"""JSON extraction, repair, and LLM output cleaning."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from engine.world_bible.validate import validate_world_bible


def fix_common_json_errors(json_str: str) -> str:
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    json_str = re.sub(r'("\w+")\s*:\s*([^,}]+)\s+(?="[^"]+"\s*:)', r'\1: \2,', json_str)

    def fix_unquoted_keys(match):
        key = match.group(1)
        if re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
            return f'"{key}": '
        return match.group(0)

    json_str = re.sub(r'(\w+):\s', fix_unquoted_keys, json_str)
    open_braces = json_str.count('{')
    close_braces = json_str.count('}')
    open_brackets = json_str.count('[')
    close_brackets = json_str.count(']')
    while close_braces < open_braces:
        json_str += '}'
        close_braces += 1
    while close_brackets < open_brackets:
        json_str += ']'
        close_brackets += 1
    while json_str.count('}') > json_str.count('{'):
        json_str = json_str.rstrip('}')
    while json_str.count(']') > json_str.count('['):
        json_str = json_str.rstrip(']')
    return json_str


def parse_json_with_fallbacks(
    json_str: str, recognized_keys: List[str] = None
) -> Tuple[Optional[Dict[str, Any]], str]:
    if recognized_keys is None:
        recognized_keys = ["state_updates", "images"]
    try:
        result = json.loads(json_str)
        if isinstance(result, dict):
            return result, ""
    except Exception:
        pass
    try:
        fixed_json = fix_common_json_errors(json_str)
        result = json.loads(fixed_json)
        if isinstance(result, dict):
            return result, ""
    except Exception:
        pass
    partial_result = {}
    error_msgs = []
    quoted_pattern = re.compile(
        r'"(' + '|'.join(recognized_keys) + r')"?\s*:\s*({.*?}|\[.*?\]|"[^"]*"|\d+|true|false|null)',
        re.IGNORECASE | re.DOTALL,
    )
    unquoted_pattern = re.compile(
        r'(?<!")\b(' + '|'.join(recognized_keys) + r')\b(?!")\s*:\s*({.*?}|\[.*?\]|"[^"]*"|\d+|true|false|null)',
        re.IGNORECASE | re.DOTALL,
    )
    for pattern in [quoted_pattern, unquoted_pattern]:
        for match in pattern.finditer(json_str):
            key, value_str = match.groups()
            try:
                partial_result[key] = json.loads(value_str)
            except Exception as e:
                error_msgs.append(f"Failed to parse {key}: {str(e)[:50]}")
    if partial_result:
        return partial_result, f"Partial parse succeeded: {', '.join(error_msgs)}"
    return None, f"All parsing strategies failed. Raw text: {json_str[:200]}..."


def extract_json_from_text(text: str) -> List[Dict[str, Any]]:
    candidates = []
    recognized_keys = ["state_updates", "images"]
    start_idx = text.find('{')
    if start_idx != -1:
        brace_count = 0
        json_start = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_candidate = text[json_start : i + 1]
                    parsed, _error = parse_json_with_fallbacks(json_candidate, recognized_keys)
                    if parsed:
                        candidates.append(parsed)
                    remaining = text[i + 1 :]
                    next_start = remaining.find('{')
                    if next_start != -1:
                        json_start = i + 1 + next_start
                        brace_count = 0
                    else:
                        break
    if not candidates:
        partial_dict = {}
        for key in recognized_keys:
            pattern = rf'"{key}"?\s*:\s*({{.*?}}|\[.*?\]|"[^"]*"|\d+|true|false|null)'
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                matched_text = match.group(0)
                colon_idx = matched_text.find(':')
                if colon_idx > 0:
                    key_part = matched_text[:colon_idx].strip().strip('"')
                    value_part = matched_text[colon_idx + 1 :].strip()
                    try:
                        partial_dict[key_part] = json.loads(value_part)
                    except Exception:
                        pass
        if partial_dict:
            candidates.append(partial_dict)
    return candidates


def strip_llm_thinking_blocks(text: str) -> str:
    t = text
    for pat in (
        r"<think>[\s\S]*?</think>",
        r"<\|think\|>[\s\S]*?<\|/think\|>",
        r"<\|channel\|>thought[\s\S]*?<\|channel\|>",
        r"<\|channel\|>[\s\S]*?<\|channel\|>",
        r"<channel\|>[\s\S]*?\|>",
    ):
        t = re.sub(pat, "", t, flags=re.IGNORECASE)
    t = re.sub(r"<think>[\s\S]{0,50000}$", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^[\s\n]*<channel\|>\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^[\s\n]*<\|channel\|>\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


def score_world_bible_candidate(d: Dict[str, Any]) -> int:
    score = 0
    if not isinstance(d, dict):
        return 0
    for key in ("objectives", "locations", "key_items", "win_condition", "solution_chain"):
        if d.get(key):
            score += 10
    for key in ("npcs", "monsters", "riddles", "item_locations", "progression_hints", "mechanics", "main_arc"):
        if d.get(key):
            score += 5
    locs = d.get("locations") or []
    if isinstance(locs, list):
        score += min(len(locs), 10)
        for loc in locs:
            if isinstance(loc, dict) and loc.get("exits"):
                score += 2
    items = d.get("key_items") or []
    if isinstance(items, list):
        score += min(len(items), 8)
    chain = d.get("solution_chain") or []
    if isinstance(chain, list):
        score += min(len(chain) * 2, 20)
    return score


def pick_world_bible_from_candidates(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    dicts = [c for c in candidates if isinstance(c, dict)]
    if not dicts:
        return None
    ranked = sorted(dicts, key=score_world_bible_candidate, reverse=True)
    for c in ranked:
        ok, _ = validate_world_bible(c)
        if ok:
            return c
    return ranked[0] if ranked else None
