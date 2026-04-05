#data for adventure
#JMR LLM based adventure game
#2024-01-13 Added npcs and npc_descs
#added room_descs_riddles
#2024-01-14 Added double riddles, hints, magic items, action words
#2024-02-09 removed names that did not sound right for the catagory.
import json

room_desc = [ "A damp echoey cavern, the air heavy with the scent of wet stone.",
    "A narrow tunnel, the walls slick with moisture, the only sound the drip of water in the distance.",
    "A vast chamber, the ceiling lost in darkness, the floor scattered with sharp stalagmites.",
    "A claustrophobic passage, the walls closing in, the air thick and stale.",
    "A gloomy grotto, the walls glistening with condensation, the sound of a subterranean river nearby.",
    "A shadowy alcove, the floor slick with a thin layer of water, the faint sound of dripping echoing off the walls.",
    "A dimly lit cavern, the air filled with the scent of minerals, the ground uneven and treacherous.",
    "A winding tunnel, the walls rough and jagged, the distant sound of water echoing eerily.",
    "A vast chamber, the ceiling covered in sharp stalactites, the air heavy with the scent of damp earth.",
    "A tight squeeze through a narrow passage, the walls cold and wet, the sound of your own breathing echoing back at you.",
    "A darkened grotto, the air thick with the scent of moss, the ground slick with a thin layer of water.",
    "A large cavern, the walls covered in a thin layer of moisture, the distant sound of dripping water the only sound.",
    "A sprawling cavern, the air filled with the scent of ancient dust, the floor littered with remnants of past adventurers.",
    "A tight crawlway, the walls covered in a thin layer of slime, the distant sound of a grue growling ominously.",
    "A claustrophobic tunnel, the walls etched with cryptic symbols, the air heavy with the scent of damp parchment.",
    "A shadowy recess, the floor covered in a thick layer of moss, the faint sound of a distant echo reverberating off the walls.",
    "A winding passage, the walls rough and scarred from the claws of some unknown creature, the distant sound of dripping water creating an eerie melody.",
    "A gloomy alcove, the air thick with the scent of mildew, the ground slick with a thin layer of algae.",
    "A sprawling chamber, the air thick with the scent of ancient stone, the floor littered with the remnants of past explorers.",
    "A narrow crevice, the walls slick with a strange luminescent fungus, the distant sound of a subterranean river echoing softly.",
    "A vast cavern, the ceiling adorned with glittering crystals, the air filled with the scent of damp earth.",
    "A tight passage, the walls covered in a thin layer of frost, the distant sound of a creature's growl sending chills down your spine.",
    "A dimly lit grotto, the air heavy with the scent of moss, the ground uneven and treacherous.",
    "A winding tunnel, the walls scarred with the marks of a long-forgotten battle, the distant sound of clashing swords echoing eerily.",
    "A large chamber, the ceiling lost in darkness, the floor covered in a thick layer of dust.",
    "A claustrophobic crawlway, the walls etched with ancient runes, the air filled with the scent of old parchment.",
    "A shadowy alcove, the floor slick with a thin layer of ice, the faint sound of a distant waterfall echoing off the walls.",
    "A vast cavern, the ceiling covered in a blanket of bats, the air heavy with the scent of guano.",
    "A narrow tunnel, the walls adorned with glowing gemstones, the distant sound of a creature's hiss echoing softly.",
    "A sprawling chamber, the air filled with the scent of stale water, the floor littered with the bones of past adventurers.",
    "A tight squeeze through a narrow passage, the walls cold and slick, the sound of your own heartbeat echoing back at you.",
    "A dimly lit grotto, the air thick with the scent of mildew, the ground uneven and treacherous.",
    "A winding tunnel, the walls rough and jagged, the distant sound of a creature's roar echoing eerily.",
    "A large cavern, the walls covered in a thin layer of frost, the distant sound of a creature's growl sending chills down your spine.",
    "A sprawling chamber, the air filled with the scent of ancient stone, the floor littered with the remnants of past explorers.",
    "A narrow crevice, the walls slick with a strange luminescent fungus, the distant sound of a subterranean river echoing softly.",
    "A vast cavern, the ceiling adorned with glittering crystals, the air filled with the scent of damp earth.",
    "A tight passage, the walls covered in a thin layer of frost, the distant sound of a creature's growl sending chills down your spine.",
    "A dimly lit grotto, the air heavy with the scent of moss, the ground uneven and treacherous.",
    "A winding tunnel, the walls scarred with the marks of a long-forgotten battle, the distant sound of clashing swords echoing eerily.",
    "A large chamber, the ceiling lost in darkness, the floor covered in a thick layer of dust.",
    "A claustrophobic crawlway, the walls etched with ancient runes, the air filled with the scent of old parchment.",
    "A shadowy alcove, the floor slick with a thin layer of ice, the faint sound of a distant waterfall echoing off the walls.",
    "A vast cavern, the ceiling covered in a blanket of bats, the air heavy with the scent of guano.",
    "A narrow tunnel, the walls adorned with glowing gemstones, the distant sound of a creature's hiss echoing softly.",
    "A sprawling chamber, the air filled with the scent of stale water, the floor littered with the bones of past adventurers.",
    "A tight squeeze through a narrow passage, the walls cold and slick, the sound of your own heartbeat echoing back at you.",
    "A dimly lit grotto, the air thick with the scent of mildew, the ground uneven and treacherous.",
    "A winding tunnel, the walls rough and jagged, the distant sound of a creature's roar echoing eerily.",
    "A large cavern, the walls covered in a thin layer of frost, the distant sound of a creature's growl sending chills down your spine.",
    "A sprawling chamber, the air filled with the scent of ancient stone, the floor littered with the remnants of past explorers.",
    "A narrow crevice, the walls slick with a strange luminescent fungus, the distant sound of a subterranean river echoing softly.",
    "A vast cavern, the ceiling adorned with glittering crystals, the air filled with the scent of damp earth.",
    "A tight passage, the walls covered in a thin layer of frost, the distant sound of a creature's growl sending chills down your spine.",
    "A dimly lit grotto, the air heavy with the scent of moss, the ground uneven and treacherous.",
    "A winding tunnel, the walls scarred with the marks of a long-forgotten battle, the distant sound of clashing swords echoing eerily.",
    "A large chamber, the ceiling lost in darkness, the floor covered in a thick layer of dust.",
    "A claustrophobic crawlway, the walls etched with ancient runes, the air filled with the scent of old parchment.",
    "A shadowy alcove, the floor slick with a thin layer of ice, the faint sound of a distant waterfall echoing off the walls.",
    ]
    
riddle_room_desc = [ 
    "A vast cavern, the ceiling lost in the darkness, the floor scattered with ancient bones and forgotten treasures.",
    "A narrow tunnel, the walls slick with a strange luminescent ooze, the sound of your own heartbeat echoing back at you.",
    "A large chamber, the walls covered in a thin layer of frost, the distant sound of a grue's growl sending chills down your spine.",
    "A labyrinthine tunnel, the walls etched with a complex map, the air filled with the scent of old leather and parchment.",
    "A hidden alcove, filled with the soft glow of luminescent fungi.",
    "A subterranean lake, the water black and still, reflecting the faint light from above.",
    "A winding path, the floor littered with bones, a chilling reminder of those who came before.",
    "A grand hall, the walls etched with ancient runes, their meaning lost to time.",
    "A circular room, the walls lined with ancient books, the air heavy with the scent of old parchment and ink.",
    "A small alcove, filled with the soft glow of a single lantern, the flickering light casting long shadows.",
    "A vast library, the shelves filled with dusty tomes, the silence broken only by the occasional drip of water.",
    "A hidden chamber, the walls covered in a mosaic of colorful tiles, the air filled with the scent of exotic spices.",
    "A narrow corridor, the walls lined with portraits of stern-looking individuals, their eyes seeming to follow you.",
    "A grand observatory, the ceiling open to the night sky, the air filled with the soft hum of a telescope.",
    "A small courtyard, the walls covered in ivy, the air filled with the scent of blooming flowers.",
    "A large greenhouse, the air heavy with the scent of damp earth and growing things.",
    "A darkened room, the only light coming from a single candle, the air filled with the scent of wax and smoke.",
    "A grand throne room, the walls lined with tapestries, the air filled with the faint echo of past celebrations."
]

double_riddle_room_desc = [
    "A grand chamber, the ceiling adorned with glittering stalactites, the air echoing with the faint whispers of long-lost explorers.",
    "A dimly lit grotto, the walls adorned with bioluminescent fungi, the sound of a subterranean waterfall nearby",
    "A vast library, filled with ancient tomes and scrolls, the scent of old parchment and ink heavy in the air",
    "A mystical grove, the trees shimmering with ethereal light, the sound of unseen creatures rustling in the undergrowth",
    "A towering observatory, the stars visible through the massive telescope, the air filled with the hum of arcane machinery",
    "A hidden shrine, the walls covered in cryptic symbols, the flickering light of candles casting eerie shadows",
    "A forgotten crypt, the air heavy with the scent of decay, the silence broken only by the distant drip of water",
    "A secret laboratory, filled with strange apparatus and bubbling potions, the air crackling with magical energy",
    "A sacred spring, the water crystal clear and filled with glowing fish, the air filled with the scent of exotic flowers",
    "A celestial vault, the walls covered in star maps and celestial symbols, the air filled with the hum of cosmic energy"]

################################################################################
# RIDDLE SYSTEM DATA STRUCTURE
# 
# All arrays are indexed together:
# - riddles[0] is solved by riddle_magic_items[0]
# - hints[0] gives a clue for riddles[0]
# - riddle_action_words[0] is the action for riddles[0]
#
# For double riddles:
# - double_riddles[0] requires BOTH items in double_magic_items[0]
# - double_hints[0] has TWO hints (one for each required item)
# - double_action_words[0] is the action for double_riddles[0]
################################################################################

# SINGLE RIDDLES - Each riddle has a corresponding hint, magic item, and action word at the same index
riddles = [
    "What has many keys, but can't open a single lock?",                                                              # 0 - Piano
    "I am a letter that stands alone in the center, found in the heart of spring months but never at their borders. What am I?",  # 1 - Letter R (improved clarity)
    "What is so fragile that saying its name breaks it?",                                                             # 2 - Silence
    "What can be broken, but is never held?",                                                                         # 3 - Promise
    "What has a heart that doesn't beat?",                                                                            # 4 - Stone/Artichoke
    "What can you catch, but not throw?",                                                                             # 5 - Reflection
    "What is always in front of you but can't be seen?",                                                              # 6 - Future
    "What has to be broken before you can use it?",                                                                   # 7 - Egg
    "I fly without wings, I cry without eyes. Wherever I go, darkness follows me. What am I?",                        # 8 - Cloud
    "I am taken from a mine, and shut up in a wooden case, from which I am never released, and yet I am used by almost every person. What am I?",  # 9 - Pencil
    "I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?",  # 10 - Fire
    "I dance and flicker with an endless appetite, turning all I touch to ash and light. What am I?",                 # 11 - Fire (improved to avoid duplicate)
    "I mark the end of a journey, bearing names of those who rest. Visitors bring flowers and memories to my chest. What am I?",  # 12 - Gravestone (clearer)
    "I have black and white soldiers standing in perfect rows. When they march, they sing without using their voice. What am I?",  # 13 - Piano (different angle)
    "I am full of holes, but I can still hold water. What am I?",                                                     # 14 - Sponge
    "When more is added to me, I become less. I am small in number but large in meaning. What am I?",                 # 15 - Few (more intuitive)
    "I am a small land surrounded by water, but add one letter and I become a passage between. What am I?",            # 16 - Isle (clearer connection)
    "I cover your floor from wall to wall, hiding the ground beneath. I can fly in stories, but in reality I stay beneath your feet. What am I?"  # 17 - Carpet (more direct)
]

# DOUBLE RIDDLES - Each requires TWO magic items to solve
double_riddles = [
    "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I? And what can capture my essence?",  # 0 - Echo/Wind
    "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I? And what can guide you through my complexities?",  # 1 - Map
    "I fly without wings, I cry without eyes. Wherever I go, darkness follows me. What am I? And what can light my path?",  # 2 - Cloud/Shadow
    "I am taken from a mine, and shut up in a wooden case, from which I am never released, and yet I am used by almost every scholar. What am I? And what can contain my essence?",  # 3 - Pencil
    "I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I? And what can shield me from my nemesis?",  # 4 - Fire/Water
    "I dance and consume, never satisfied until all is ash. What am I? And what feeds my endless hunger?",  # 5 - Fire/Wood (simplified)
    "I rule without a crown, my kingdom is a table of green felt. Red or black, I decide fates with my brothers. Who am I? And where is my realm?",  # 6 - King of Hearts (clearer)
    "I have keys but no locks, hammers but no nails. My voice can move hearts without words. What am I? And what tool brings me to life?",  # 7 - Piano (different angle)
    "I shine brightest in darkness, yet many never see me. What am I? And what surface can capture my glow?",  # 8 - Starlight (simplified)
    "I wind between stone guardians, offering safe passage to those who seek the castle. What am I? And what protects travelers upon me?"  # 9 - Castle Path (clearer)
]

# HINTS for single riddles - Index matches riddles[]
hints = [
    "Play the piano in the corner.",                # 0 - Piano riddle
    "Think of letters that appear in both March and April. Which one is in the middle?",  # 1 - Letter R riddle (improved)
    "Be quiet and think.",                          # 2 - Silence riddle
    "Wear a promise.",                              # 3 - Promise riddle
    "Weigh the stone statue.",                      # 4 - Stone Heart riddle
    "Try to catch your reflection in the water.",   # 5 - Reflection riddle
    "Think about the future.",                      # 6 - Future riddle
    "Look for an egg.",                             # 7 - Egg riddle
    "Look for a cloud.",                            # 8 - Cloud riddle
    "Find a pencil.",                               # 9 - Pencil riddle
    "Search for a fire.",                           # 10 - Fire riddle (first)
    "Seek the eternal flame that dances.",          # 11 - Fire riddle (improved)
    "Visit those who have passed on.",              # 12 - Gravestone riddle (improved)
    "Black and white keys make beautiful music.",   # 13 - Piano riddle (improved)
    "Look for a sponge.",                           # 14 - Sponge riddle
    "Sometimes less is more. Think small numbers.", # 15 - Few riddle (improved)
    "An island becomes a walkway with one letter.", # 16 - Isle riddle (improved)
    "Magic carpets fly in tales, but real ones cover floors."  # 17 - Carpet riddle (improved)
]

# HINTS for double riddles - Each has TWO hints, index matches double_riddles[]
double_hints = [
    ["The wind carries voices, but what can hold them?", "A stone that echoes, perhaps?"],                         # 0 - Echo/Wind riddle
    ["A guide that always points true, even without a house or a tree in sight.", "A map needs direction to be useful."],  # 1 - Map riddle (improved)
    ["Clouds bring rain and shadow. What charm controls them?", "Shadows flee from light. What lantern banishes darkness?"],  # 2 - Cloud/Shadow riddle (improved)
    ["Graphite from the earth writes our thoughts.", "Wood encases the graphite, protecting the lead within."],     # 3 - Pencil riddle (improved)
    ["Fire is a living gem that breathes.", "Water is fire's ancient enemy, held in an orb."],                      # 4 - Fire/Water riddle (improved)
    ["Fire dances with eternal hunger.", "Only aged wood can feed such ancient flames."],                           # 5 - Fire/Wood riddle (improved)
    ["The King rules the card table.", "Every jester needs a stage for their games."],                              # 6 - King of Hearts riddle (improved)
    ["Music comes from keys without locks.", "Every key needs an instrument to sing."],                             # 7 - Piano riddle (improved)
    ["Stars shine brightest in the darkest night.", "Mirrors capture and reflect all light."],                      # 8 - Starlight riddle (improved)
    ["Castle paths wind between protective walls.", "Tower shields guard those who travel."]                         # 9 - Castle Path riddle (improved)
]

# MAGIC ITEMS for single riddles - Index matches riddles[]
riddle_magic_items = [
    "Piano of Time",          # 0 - Answer: Piano
    "Letter R Rune",          # 1 - Answer: Letter R (maRch/apRil)
    "Silence Orb",            # 2 - Answer: Silence
    "Promise Ring",           # 3 - Answer: Promise
    "Stone Heart",            # 4 - Answer: Stone/Artichoke (heart that doesn't beat)
    "Mirror of Shadows",      # 5 - Answer: Reflection (catch but not throw)
    "Future's Eye",           # 6 - Answer: Future
    "Dragon's Egg",           # 7 - Answer: Egg
    "Cloud Tear",             # 8 - Answer: Cloud
    "Scholar's Pencil",       # 9 - Answer: Pencil
    "Fire Seed",              # 10 - Answer: Fire
    "Eternal Flame",          # 11 - Answer: Fire (duplicate riddle, different item)
    "Gravestone Rubbing",     # 12 - Answer: Gravestone
    "Piano Key",              # 13 - Answer: Piano (duplicate riddle, different item)
    "Sponge of Absorption",   # 14 - Answer: Sponge
    "Few's Feather",          # 15 - Answer: Few (few + er = fewer)
    "Isle Stone",             # 16 - Answer: Isle (isle + a = aisle)
    "Carpet Fragment"         # 17 - Answer: Carpet (car-pet-carp)
]

# MAGIC ITEMS for double riddles - Each riddle needs BOTH items at the index
double_magic_items = [
    ["Echo Stone", "Whispering Wind"],      # 0 - Echo/Wind
    ["Map of Truth", "Compass of Direction"],  # 1 - Map
    ["Cloud Charm", "Shadow Lantern"],      # 2 - Cloud/Shadow
    ["Graphite Rod", "Wooden Case"],        # 3 - Pencil
    ["Fire Gem", "Water Orb"],              # 4 - Fire/Water
    ["Flame", "Elder Wood"],                # 5 - Fire/Wood
    ["King's Card", "Jester's Table"],      # 6 - King of Hearts/Card Table
    ["Soul Key", "Piano"],                  # 7 - Piano
    ["Starlight", "Moon Mirror"],           # 8 - Starlight/Mirror
    ["Castle Path", "Tower Shield"]         # 9 - Castle Path
]

# ACTION WORDS for single riddles - Index matches riddles[]
riddle_action_words = [
    "play",      # 0 - Piano
    "check",     # 1 - Letter R
    "quiet",     # 2 - Silence
    "wear",      # 3 - Promise
    "use",       # 4 - Stone Heart
    "look",      # 5 - Mirror
    "look",      # 6 - Future
    "break",     # 7 - Egg
    "look",      # 8 - Cloud
    "use",       # 9 - Pencil
    "plant",     # 10 - Fire
    "take",      # 11 - Fire (second)
    "take",      # 12 - Gravestone
    "play",      # 13 - Piano (second)
    "squeeze",   # 14 - Sponge
    "think",     # 15 - Few
    "find",      # 16 - Isle
    "look"       # 17 - Carpet
]

# ACTION WORDS for double riddles - Index matches double_riddles[]
double_action_words = [
    "listen",      # 0 - Echo/Wind
    "place",       # 1 - Map
    "light",       # 2 - Cloud/Shadow
    "hit",         # 3 - Pencil
    "extinguish",  # 4 - Fire/Water (fixed typo)
    "burn",        # 5 - Fire/Wood
    "play",        # 6 - King of Hearts
    "play",        # 7 - Piano
    "reflect",     # 8 - Starlight
    "walk"         # 9 - Castle Path
]

riddle_keys = {}

room_descs_riddles = []

################################################################################
# ROOM GENERATION WITH RIDDLES
#
# Three types of rooms are created:
# 1. Regular rooms (no riddles) - use room_desc[]
# 2. Single riddle rooms - use riddle_room_desc[] with corresponding riddles[]
# 3. Double riddle rooms - use double_riddle_room_desc[] with double_riddles[]
#
# Example: Room with riddle_room_desc[0] will have:
# - Riddle: riddles[0] ("What has many keys...")
# - Magic Item Needed: riddle_magic_items[0] ("Piano of Time")
# - Hint: hints[0] ("Play the piano in the corner")
# - Action: riddle_action_words[0] ("play")
################################################################################

# Rooms without riddles
for desc in room_desc:
    room_descs_riddles.append({
        "description": desc,
        "riddles": [],
        "hints": [],
        "magic_items": [],
        "action_words": ()
    })

# Rooms with one riddle (using indexed arrays)
for i, desc in enumerate(riddle_room_desc):
    room_descs_riddles.append({
        "description": desc,
        "riddles": [riddles[i]],                    # Riddle at index i
        "hints": [hints[i]],                        # Corresponding hint
        "magic_items": [riddle_magic_items[i]],     # Item that solves it
        "action_words": riddle_action_words[i]      # Action to perform
    })

# Rooms with double riddles (need TWO items)
for i, desc in enumerate(double_riddle_room_desc):
    room_descs_riddles.append({
        "description": desc,
        "riddles": [double_riddles[i]],             # Double riddle at index i
        "hints": double_hints[i],                   # TWO hints (list)
        "magic_items": double_magic_items[i],       # TWO items needed (list)
        "action_words": double_action_words[i]      # Action to perform
    })

for i in range(0, len(room_descs_riddles)):
    desc = room_descs_riddles[i]["description"]
    riddles = room_descs_riddles[i]["riddles"]
    hints = room_descs_riddles[i]["hints"]
    magic_items = room_descs_riddles[i]["magic_items"]
    action_words = room_descs_riddles[i]["action_words"]
    print(i, ": ", desc, ", ", riddles, ", ", hints, ", ", magic_items, " ", action_words)


monsters = ["Dragon", "Goblin","Giant Spider", "Undead Warrior", "Cave Troll", "Basilisk", "Shadow Beast", "Spectral Wraith", "Troll", "Ogre", "Vampire",
           "Werewolf", "Zombie", "Ghost", "Demon", "Giant Spider", "Banshee", "Mummy", "Cyclops", "Harpy", "Minotaur", "Kraken", "Gorgon", "Chimera",
           "Sphinx", "Griffin", "Centaur", "Siren", "Nymph", "Basilisk", "Phoenix", "Hydra", "Lich", "Wraith", "Specter", "Poltergeist", "Djinn", "Yeti", 
            "Sasquatch", "Manticore", "Leviathan", "Cerberus", "Succubus", "Incubus", "Naga","Fire Drake", "Ice Goblin", "Sand Serpent","Skeleton Knight", "Mountain Troll",
            "Nightmare Beast", "Ethereal Wisp", "Stone Golem", "Swamp Ogre", "Bloodsucker", "Moon Beast", 
            "Flesh Eater", "Spectral Apparition", "Hellspawn", "Web Weaver", "Screaming Specter", "Desert Mummy", "One-eyed Titan", "Wind Harpy",
            "Labyrinth Minotaur", "Sea Kraken", "Stone Gorgon", "Fire Chimera", "Riddle Sphinx", "Sky Griffin", "Forest Centaur", "Sea Siren", "Forest Dryad", 
            "Stone Basilisk", "Firebird", "Water Hydra", "Undead Lich", "Shadow Wraith", "Ethereal Specter", "Mischievous Poltergeist", "Desert Genie", 
            "Snow Yeti", "Forest Bigfoot", "Lion Scorpion", "Hellhound", "Dream Demon", "Nightmare Demon"]

weapons = ["Sword", "Axe", "Bow", "Dagger","Bow and Arrows", "Dagger", "Mace", "Warhammer", "Crossbow", "Flaming Torch", "Mace", "Staff", "Crossbow", "Spear", 
            "Halberd", "Warhammer", "Flail", "Scimitar", "Glaive", "Longbow", "Shortsword", "Greatsword", "Battleaxe", "Morningstar", "Rapier", "Katana", 
            "Falchion", "Trident", "Javelin", "Pike", "Lance", "Longsword", "Claymore", "Sabre", "Cutlass", "Estoc", "Scythe", "Khopesh", "Dirk", "Machete", 
            "Cudgel", "Club", "Quarterstaff", "Bastard Sword","Broadsword", "Battle Axe", "Longbow", "Stiletto", 
            "Arrows and Quiver", "Poison Dagger", "Spiked Mace", "Sledgehammer", "Repeating Crossbow", "Burning Torch", "Flanged Mace", "Wizard's Staff", 
            "Bolt Thrower", "Javelin", "Poleaxe", "Sledgehammer", "Chain Flail", "Curved Sword", "Polearm", "Recurve Bow", "Short Blade", "Two-handed Sword", 
            "War Axe", "Spiked Morningstar", "Fencing Sword", "Samurai Sword", "Broad Falchion", "Three-pronged Spear", "Throwing Spear", "Long Pike", 
            "Cavalry Lance", "Knight's Sword", "Two-handed Claymore", "Cavalry Sabre", "Pirate's Cutlass", "Thrusting Sword", "Reaper's Scythe", 
            "Egyptian Sword", "Stabbing Dirk", "Jungle Machete", "Heavy Cudgel", "Wooden Club", "Monk's Quarterstaff", "Hand-and-a-half Sword"]

armors = ["Chainmail", "Plate Armor", "Leather Armor", "Scale Mail", "Chain Shirt", "Breastplate", "Splint Armor", "Ring Mail", "Studded Leather", "Padded Armor","Shield", "Small Shield", "Knight's Shield", "Castle Shield"]

treasures = ["Gold Coins", "Diamonds","Emerald Necklace", "Ancient Artifact", "Royal Crown", "Silver Chalice", "Jeweled Scepter", 
            "Ancient Artifact", "Royal Crown", "Precious Gems", "Silver Chalice", "Golden Statue", "Rare Books", 
            "Emerald Necklace", "Ruby Ring", "Sapphire Bracelet", "Platinum Brooch", "Ivory Figurine", "Silk Tapestry", "Pearl Earrings", "Jade Idol", 
            "Bronze Mirror", "Crystal Vase", "Leather-bound Tome", "Engraved Locket", "Golden Goblet", "Silver Scepter", "Ornate Chest", 
            "Exquisite Painting", "Ancient Scroll", "Rare Manuscript",
            "Sacred Chalice", "Royal Diadem", "Ancient Coin", "Exotic Spices", "Silk Robes", "Golden Harp", "Ivory Horn", "Emerald Amulet","Golden Doubloons", 
            "Uncut Diamonds", "Sapphire Necklace", "King's Crown", "Silver Goblet", "Gem-encrusted Scepter", 
            "Forgotten Artifact", "Queen's Crown", "Rare Gemstones", "Golden Chalice", "Bronze Statue", "Hidden Map", "Ancient Tomes", 
            "Ruby Necklace", "Diamond Ring", "Topaz Bracelet", "Platinum Pin", "Ivory Sculpture", "Silk Mosaic", "Pearl Pendant", "Jade Totem", 
            "Copper Mirror", "Crystal Pitcher", "Engraved Pendant", "Golden Cup", "Ornate Trunk", "Ancient Papyrus", "Rare Parchment", "Invaluable Relic", 
            "King's Tiara", "Silk Gowns", "Ivory Flute", "Emerald Talisman"]

magic_items = ["Healing Potion", "Ring of Power", "Amulet of Protection", "Staff of Wisdom", "Charm of Luck", "Book of Spells", "Elixir of Life", "Phoenix Feather", 
            "Pendant of Shielding", "Rod of Insight", "Elixir of Vitality", "Phoenix Plume", "Healing Amulet", "Youth Elixir", "Scroll of Wisdom", "Immortality Elixir"]

npcs = ["Jordana", "Noah","Elana", "Gabby", "Jacob", "Sabina","Bonnie","Thorgar", "Eldrin", "Morgana", "Lilith", "Bael", "Nyx", "Orion", "Vega", 
        "Rigel", "Helga", "Fenrir", "Odin", "Freya", "Loki", "Eir", "Baldur", "Tyr", "Frigg", "Idun","Morg", "Thorn", "Bael", "Grim", "Vex", 
        "Krag", "Dusk", "Blaze", "Frost", "Gale", "Shade", "Echo", "Rift", "Quill", "Wisp", "Slate", "Flint", "Bramble", "Crag", "Moss", "Pike", 
        "Rook", "Hawk", "Raven", "Vale", "Reed", "Ash", "Birch", "Zephyr", "Cinder", "Tide", "Storm", "Pyre", "Shiver", "Bolt", "Quake", "Gloom", 
        "Fawn", "Petal", "Thicket", "Grove", "Brook", "Breeze", "Glade", "Cliff", "Stone", "Flame", "Frostbite", "Gust"]

npc_descs = ["an old wizard with a long white beard and a mysterious aura", 
            "a charming princess with a secret love for adventure", 
            "a grumpy dwarf blacksmith who makes the best weapons in the kingdom", 
            "a cunning thief with a heart of gold and quick fingers", 
            "a wise old woman who can see the future in her dreams", 
            "a brave knight with a shiny armor and a strong sense of justice", 
            "a mischievous fairy who loves playing tricks on travelers", 
            "a friendly innkeeper with a knack for storytelling", 
            "a mysterious stranger cloaked in shadows with an unknown agenda",
            "a seasoned warrior with a scarred face and a haunted past",
            "a cheerful bard who can play any instrument and knows all the local legends",
            "a stern guard captain who takes his duties very seriously",
            "a cunning sorceress with a pet raven and a taste for riddles",
            "a kind-hearted priest who heals the wounded and helps the poor",
            "a silent assassin who moves like a shadow and never misses his target",
            "a grizzled ranger with a keen eye and a deep love for nature",
            "a jovial giant who loves to drink and tell tall tales",
            "a wise old king with a long white beard and a gentle heart",
            "a beautiful queen with a sharp mind and a regal bearing",
            "a young prince with a rebellious streak and a thirst for adventure",
            "a shy maiden with a sweet smile and a voice like a nightingale",
            "a wizened hermit who lives in the woods and knows many secrets",
            "a fierce dragon with scales like steel and breath of fire",
            "a sly goblin with a quick wit and a quicker blade",
            "a noble elf with a bow as tall as a man and eyes full of wisdom",
            "a seasoned warrior with a scarred face and a haunted past",
            "a cheerful bard who can play any instrument and knows all the local legends",
            "a stern guard captain who takes his duties very seriously",
            "a cunning sorceress with a pet raven and a taste for riddles",
            "a kind-hearted priest who heals the wounded and helps the poor",
            "a silent assassin who moves like a shadow and never misses his target",
            "a grizzled ranger with a keen eye and a deep love for nature",
            "a jovial giant who loves to drink and tell tall tales",
            "a wise old king with a long white beard and a gentle heart",
            "a beautiful queen with a sharp mind and a regal bearing",
            "a young prince with a rebellious streak and a thirst for adventure",
            "a shy maiden with a sweet smile and a voice like a nightingale",
            "a wizened hermit who lives in the woods and knows many secrets",
            "a fierce dragon with scales like steel and breath of fire",
            "a sly goblin with a quick wit and a quicker blade",
            "a noble elf with a bow as tall as a man and eyes full of wisdom"
            "a stoic paladin with a gleaming sword and an unshakeable faith",
            "a cunning rogue with a hidden dagger and a charming smile",
            "a wise druid who can speak to animals and control the elements",
            "a fearless barbarian with a massive axe and a booming laugh",
            "a mysterious necromancer with a staff of bone and a cloak of shadows",
            "a gentle healer with a soothing touch and a calming presence",
            "a charismatic bard with a lute and a voice that can charm the stars",
            "a stern monk with a disciplined mind and lightning-fast reflexes",
            "a crafty alchemist with a bag full of potions and a curious nature",
            "a noble knight with a polished armor and a code of honor",
            "a mischievous imp with a wicked grin and a love for chaos",
            "a wise oracle with a crystal ball and a cryptic prophecy",
            "a fierce werewolf with a gruff demeanor and a heart of gold",
            "a graceful elf archer with a keen eye and a swift arrow",
            "a stoic golem with a body of stone and a soul of kindness",
            "a cunning vampire with a charming smile and a thirst for knowledge",
            "a brave squire with a wooden sword and dreams of knighthood",
            "a wise-cracking jester with a colorful outfit and a quick wit",
            "a mysterious wanderer with a hooded cloak and a past full of secrets",
            "a diligent scholar with a pile of books and a thirst for knowledge",
            "a fierce amazon warrior with a spear and a strong will",
            "a gentle dryad with a love for nature and a song in her heart",
            "a gruff troll with a club and a surprisingly soft heart",
            "a wise mermaid with a beautiful voice and a love for the sea",
            "a brave centaur with a bow and a noble spirit",
            "a cunning fox spirit with a mischievous grin and a love for pranks",
            "a noble unicorn with a shimmering mane and a gentle nature",
            "a wise sphinx with a riddle on her lips and a secret in her heart",
            "a brave griffin with a majestic wingspan and a fierce loyalty",
            "a mysterious ghost with a sad smile and a tale of woe",
            "a diligent gnome tinkerer with a bag of tools and a mind full of ideas",
            "a fierce minotaur with a mighty axe and a strong sense of honor",
            "a gentle naiad with a love for rivers and a song in her heart",
            "a wise treant with a deep voice and a love for the forest",
            "a brave phoenix with a fiery plumage and a spirit that never dies",
            "a mysterious djinn with a swirling form and a knack for granting wishes",
            "a diligent dwarf miner with a pickaxe and a heart of gold",
            "a fierce orc warrior with a massive sword and a strong sense of honor",
            "a gentle pixie with a love for flowers and a sprinkle of magic dust",
            ]

adventure_data = {
    "room_descs_riddles": room_descs_riddles,
    "monsters": monsters,
    "weapons": weapons,
    "armors": armors,
    "treasures": treasures,
    "magic_items": magic_items,
    "npcs": npcs,
    "npc_descs": npc_descs
}

with open('adventure_dataRA.json', 'w') as f:
    json.dump(adventure_data, f)

