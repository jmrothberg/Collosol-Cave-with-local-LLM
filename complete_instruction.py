def get_complete_instructions():
    
    instructions = """
    Complete Instructions for the Adventure Game:

    1. Basic Commands:
       - go <direction>: Move in a direction (north, south, east, or west)
       - take <item>: Pick up an item from the room
       - leave <item>: Drop an item in the room
       - use <item>: Use an item from your inventory ( for magic items)
       - study <item>: Examine an item in the room or your inventory
       - attack <monster> with <weapon>: Fight a monster using a specific weapon
       - trade <your_item> for <npc_item>: Trade with an NPC
       - talk <message>: Speak to an NPC in the room
       - solve <item> and <item>: Attempt to solve a riddle using magic items

    2. Special Commands:
       - details: Get an updated room description
       - draw: Update the room image
       - help: Show basic help message

    3. Cheats and Secret Commands:
       - xyzzy: Activate cheat mode
       - <magicword> <room_number>: Create magical connections (random)
       - health <number>: Set your health to a specific value (cheat mode)

    4. Game Objectives:
       - Collect all treasures
       - Defeat all monsters
       - Solve all riddles
       - Explore all rooms
       - Collect trophies for each completed objective

    5. Tips for Interacting with NPCs and Monsters:
       - When talking to NPCs, be specific about what you want to know. 
         For example:
         * "What do you know about the riddles in this dungeon?"
         * "Can you tell me about any hidden treasures?"
         * "What items do you have for trade?"
       - Use the 'about' command to ask NPCs about the game world:
         * about the magic items needed for riddles
         * about the locations of monsters
       - Be cautious when attacking monsters. They may counterattack if 
         you're wealthy or if you've initiated combat.

    6. Riddle Solving:
       - Pay attention to the riddles and hints in each room
       - Collect magic items from various rooms, NPCs, and monsters
       - 'solve' command with the magic items to complete riddles

    7. Inventory Management:
       - Regularly check your inventory using the 'inventory' command
       - Equip armor to increase your protection against monster attacks
       - Use magic items to heal yourself when needed

    8. Exploration:
       - Try to explore all rooms to gain the explorer trophy
       - Use the map to keep track of visited and unvisited rooms
       - Remember that NPCs can move between rooms

    9. Difficulty Levels:
       - Easy: More information on room contents and NPC inventories
       - Medium: Less information is given, requiring more exploration
       - Hard: Minimal information provided, maximum challenge
       - Cheat: Allows use of cheat commands for easier gameplay

    10. Saving and Loading:
        - Use the 'Save-Game' option to save your progress
        - Use the 'Load-Game' option to continue a saved game

    11. Tips for Command Phrasing:
      Direct phrases that start with the action you want to take
      Movement, "go" or "move": "go north", "move east"
      Taking items, "take" or "get": "take sword", "get potion"
      Leaving items, "leave" or "drop": "leave gold", "drop armor"
      Attacks, u"attack [monster] with [weapon]": "attack dragon with sword"
      Talking, "talk" or "ask": "talk to wizard", "ask about riddle"
      Trading, "trade [your item] for [their item]": "trade gold for potion"
      Solving riddles,"solve with [item] and [item]": "solve with wand and gem"
      Studying items,"study" or "examine": "study map", "examine potion"    

    Remember, the key to success is careful exploration, strategic use of items, and clever interaction with NPCs and monsters. Good luck on your adventure!
    """
    return instructions