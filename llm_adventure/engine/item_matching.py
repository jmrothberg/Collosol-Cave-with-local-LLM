"""Item and exit name matching for fast-path commands."""

import re
from typing import List, Optional

_ARTICLE_RE = re.compile(r"^(the|a|an|some|my|that|this)\s+", re.IGNORECASE)


def normalize_item_token(s: str) -> str:
    """Lowercase, drop leading article, unify underscores/spaces for matching."""
    if not s:
        return ""
    t = s.strip().strip("\"' .,!?")
    t = _ARTICLE_RE.sub("", t).strip()
    t = t.replace("'", "").replace("'", "")
    t = t.replace("_", " ").replace("-", " ")
    t = re.sub(r"\s+", " ", t).lower()
    return t


def split_item_list(s: str) -> List[str]:
    """Split multi-item targets on commas, 'and', '&', 'plus'."""
    if not s:
        return []
    # CHANGE (solvability May 2026): try whole string as one item name first.
    whole = s.strip()
    if whole and ("," not in whole and " and " not in whole.lower() and "&" not in whole):
        return [whole]
    parts = re.split(r"\s*(?:,|\band\b|&|\bplus\b)\s*", s, flags=re.IGNORECASE)
    out = [p.strip() for p in parts if p and p.strip()]
    return out or ([whole] if whole else [])


def match_item(target: str, candidates: List[str]) -> Optional[str]:
    """Best-effort match of target against candidate item names."""
    if not target or not candidates:
        return None
    # Whole-string match before splitting (e.g. "sword and shield" as one item).
    for c in candidates:
        if c.lower() == target.lower():
            return c
    for c in candidates:
        if normalize_item_token(c) == normalize_item_token(target):
            return c
    t_norm = normalize_item_token(target)
    if not t_norm:
        return None
    for c in candidates:
        if c.lower() == target.lower():
            return c
    for c in candidates:
        if normalize_item_token(c) == t_norm:
            return c
    for c in candidates:
        cn = normalize_item_token(c)
        if t_norm in cn or cn in t_norm:
            return c
    t_tokens = set(t_norm.split())
    if t_tokens:
        for c in candidates:
            c_tokens = set(normalize_item_token(c).split())
            if t_tokens.issubset(c_tokens):
                return c
    return None


def match_exit(text: str, exits: List[str]) -> Optional[str]:
    """Return the best-matching exit name from exits, or None."""
    if not exits:
        return None
    t = (text or "").strip().lower()
    if not t:
        return None
    for ex in exits:
        if ex.lower() == t:
            return ex
    for ex in exits:
        if t in ex.lower() or ex.lower() in t:
            return ex
    if len(t) <= 5:
        for ex in exits:
            if ex.lower().startswith(t[0]):
                return ex
    return None
