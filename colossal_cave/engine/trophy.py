"""Trophy / victory condition checking."""

from typing import Any, Callable, Dict, List, Optional, Tuple


def check_trophy(
    trophy: str,
    condition: Callable[[Any], bool],
    victory_message: str,
    rooms: Dict[int, Any],
    trophies: List[str],
    player: Any,
    victory_action: str = "",
) -> Tuple[str, Optional[Any], List[Any]]:
    """
    Check if a trophy condition is met across all rooms.
    Returns (console_text, player_image_or_none, trophie_images).
    Mutates trophies list and player state when awarded.
    """
    player_image = None
    trophie_images: List[Any] = []
    console = ""

    if trophy in trophies:
        return console, player_image, trophie_images

    all_conditions_met = True
    for room in rooms.values():
        if condition(room):
            all_conditions_met = False

    if all_conditions_met:
        console = console + (f"Congratulations, {victory_message}!\n")
        player.drawing_self(victory_action)
        player_image = player.image
        player.current_room.add_treasure(False, f"{trophy.capitalize()}-Trophie")
        temp_console, trophie_images = player.take(f"{trophy.capitalize()}-Trophie")
        trophies.append(trophy)
        console = console + temp_console + f"Your trophies: {', '.join(trophies)}\n"

    return console, player_image, trophie_images
