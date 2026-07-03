"""Map PNG generation from room graph."""

from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont


def draw_map_png(rooms: Dict[int, Any], cheat_mode: bool = False, output_path: str = "map.png") -> Image.Image:
    """Draw explored room map as PNG. Accepts rooms dict instead of global."""
    num_rooms = len(rooms)
    num_rooms_side = int(num_rooms ** 0.5)
    room_size = 60
    connection_size = 8
    w, h = 10, 10

    image_size = ((num_rooms_side) * room_size, (num_rooms_side) * room_size)
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for number, room in rooms.items():
        x = ((number - 1) % num_rooms_side) * room_size
        y = ((number - 1) // num_rooms_side) * room_size
        if room.explored or cheat_mode:
            if room.image:
                imagetopaste = room.image.resize((room_size, room_size), Image.LANCZOS)
                image.paste(imagetopaste, (x, y))

    for number, room in rooms.items():
        x = ((number - 1) % num_rooms_side) * room_size
        y = ((number - 1) // num_rooms_side) * room_size
        if room.explored or cheat_mode:
            for direction, _connected_room in room.connected_rooms.items():
                if direction == "east":
                    draw.line(
                        [(x + room_size / 1.2, y + room_size / 2),
                         (x + room_size * 1.2, y + room_size / 2)],
                        fill="green", width=connection_size,
                    )
                if direction == "south":
                    draw.line(
                        [(x + room_size / 2, y + room_size / 1.2),
                         (x + room_size / 2, y + room_size * 1.2)],
                        fill="blue", width=connection_size,
                    )
                if direction == "west":
                    draw.line(
                        [(x - (room_size / 1.2 - room_size), y + room_size / 2),
                         (x - (room_size * 1.2 - room_size), y + room_size / 2)],
                        fill="pink", width=connection_size,
                    )
                if direction == "north":
                    draw.line(
                        [(x + room_size / 2, y + room_size / 1.2 - room_size),
                         (x + room_size / 2, y + room_size * 1.2 - room_size)],
                        fill="orange", width=connection_size,
                    )
        draw.text((x + (room_size - w) / 2, y + (room_size - h) / 2), str(number), fill="black", font=font)

    image.save(output_path)
    return image
