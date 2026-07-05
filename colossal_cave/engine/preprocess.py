"""Command preprocessing — typo correction before LLM parsing."""

import re


def preprocess_command(command: str) -> str:
    """Preprocess command for better LLM understanding."""
    common_typos = {
        'attck': 'attack', 'atack': 'attack', 'attac': 'attack',
        'tke': 'take', 'tak': 'take', 'tkae': 'take',
        'ridde': 'riddle', 'ridle': 'riddle', 'riddel': 'riddle',
        'sovle': 'solve', 'slove': 'solve', 'solv': 'solve',
        'waer': 'wear', 'ware': 'wear', 'wera': 'wear',
        'noth': 'north', 'soth': 'south', 'esst': 'east', 'wst': 'west',
        'invntory': 'inventory', 'hlp': 'help', 'hlep': 'help',
        'trdae': 'trade', 'traed': 'trade', 'talkk': 'talk',
        'examin': 'examine', 'examne': 'examine', 'examinee': 'examine',
    }

    # CHANGE (accuracy): whole-word replacement only — 'tak' must not corrupt 'take'
    result = command
    for typo, correct in common_typos.items():
        result = re.sub(rf'\b{re.escape(typo)}\b', correct, result, flags=re.IGNORECASE)

    return result.strip()
