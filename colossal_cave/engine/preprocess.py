"""Command preprocessing — typo correction before LLM parsing."""


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
        'trdae': 'trade', 'traed': 'trade', 'talkk': 'talk'
    }

    cmd_lower = command.lower()
    for typo, correct in common_typos.items():
        if typo in cmd_lower:
            command = command.replace(typo, correct)
            command = command.replace(typo.upper(), correct.upper())
            command = command.replace(typo.capitalize(), correct.capitalize())

    return command.strip()
