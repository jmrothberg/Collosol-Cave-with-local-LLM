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

doulbe_riddle_room_desc = [
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

riddles = [
    "What has many keys, but can't open a single lock?",
    "What is seen in the middle of March and April that can't be seen at the beginning or end of either month?",
    "What is so fragile that saying its name breaks it?",
    "What can be broken, but is never held?",
    "What has a heart that doesn’t beat?",
    "What can you catch, but not throw?",
    "What is always in front of you but can’t be seen?",
    "What has to be broken before you can use it?",
    "I fly without wings, I cry without eyes. Wherever I go, darkness follows me. What am I?",
    "I am taken from a mine, and shut up in a wooden case, from which I am never released, and yet I am used by almost every person. What am I?",
    "I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?",
    "I am always hungry and will die if not fed, but whatever I touch will soon turn red. What am I?",
    "I have a name that is not mine, and no one thinks of me in their prime. People cry at my sight, and lie by me all day and night. What am I?",
    "I am a box that holds keys without locks, yet they can unlock your soul. What am I?",
    "I am full of holes, but I can still hold water. What am I?",
    "I am a word of letters three; add two and fewer there will be. What am I?",
    "I am a word that begins with the letter i. If you add the letter a to me, I become a new word with a different meaning, but that sounds exactly the same. What word am I?",
    "I am a word of six; my first three letters refer to an automobile; my last three letters refer to a household animal; my first four letters is a fish; my whole is found in your room. What am I?"
]

double_riddles = [
    "I speak without a mouth and hear without ears. I have no body, but I come alive with wind. What am I? And what can capture my essence?",
    "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I? And what can guide you through my complexities?",
    "I fly without wings, I cry without eyes. Wherever I go, darkness follows me. What am I? And what can light my path?",
    "I am taken from a mine, and shut up in a wooden case, from which I am never released, and yet I am used by almost every scholar. What am I? And what can contain my essence?",
    "I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I? And what can shield me from my nemesis?",
    "I am always hungry and will die if not fed, but whatever I touch will soon turn red. What am I? And what can satiate my hunger?",
    "I have a heart that doesn’t beat. I have a home but I never sleep. I can take a man’s house and build another’s, And I love to play games with my many brothers. I am a king among fools. Who am I? And where can I be played?",
    "I am a box that holds keys without locks, yet they can unlock your deepest emotions. What am I? And what can play my keys?",
    "I am a mirror for the famous, but when seen by the common, I am often ignored. What am I? And what can reflect my beauty?",
    "I am a path situated between high walls with towers. Despite my length, I am not tiring to walk. What am I? And what can protect you on my journey?"]

hints = [
    "Play the piano in the corner.",
    "Check the calendar on the wall.",
    "Be quiet and think.",
    "Wear a promise.",
    "weigh the stone statue.",
    "Try to catch your reflection in the water.",
    "Think about the future.",
    "Look for an egg.",
    "Look for a cloud.",
    "Find a pencil.",
    "Search for a fire.",
    "Look for a gravestone.",
    "Find a piano.",
    "Look for a sponge.",
    "Think about the word 'few'.",
    "Find the word 'isle'.",
    "Look for a 'carpet'.",
    "Look for a candle."
]

double_hints = [
    ["The wind carries voices, but what can hold them?", "A stone that echoes, perhaps?"],
    ["A guide that always points true, even without a house or a tree in sight.", "A map of truth, perhaps?"],
    ["A charm that can summon a cloud, and a lantern that can light the shadows.", "A charm and a lantern, perhaps?"],
    ["A rod of graphite, a case of wood, a scholar's tools.", "A rod and a case, perhaps?"],
    ["A gem that needs air, an orb that repels water.", "A gem and an orb, perhaps?"],
    ["A flame that needs feeding, wood that can satiate.", "A flame and wood, perhaps?"],
    ["A card for a king, a table for a jester.", "A card and a table, perhaps?"],
    ["A key for the soul, a piano for the melody.", "A key and a piano, perhaps?"],
    ["Starlight for the famous, a mirror for the moon.", "Starlight and a mirror, perhaps?"],
    ["A path for a castle, a shield for a tower.", "A path and a shield, perhaps?"]]

riddle_magic_items = [
    "Piano of Time", "Calendar of Ages", "Silence Orb", "Promise Ring", "Stone Heart", "Mirror of Shadows", "Future's Eye", "Dragon's Egg",
    "Cloud Tear", "Miner's Pencil", "Fire Seed", "Gravestone Rubbing", "Piano Key", "Sponge of Absorption", "Few's Feather", "Isle Stone", "Carpet Fragment", "Candle of Shadows"]

double_magic_items = [
    ["Echo Stone", "Whispering Wind"],
    ["Map of Truth", "Compass of Direction"],
    ["Cloud Charm", "Shadow Lantern"],
    ["Graphite Rod", "Wooden Case"],
    ["Fire Gem", "Water Orb"],
    ["Flame", "Elder Wood"],
    ["King's Card", "Jester's Table"],
    ["Soul Key", "Piano"],
    ["Starlight", "Moon Mirror"],
    ["Castle Path", "Tower Shield"]]

riddle_action_words = [
    "play", "check", "quiet", "wear", "use", "look", "look", "break",
    "look", "use", "plant", "take", "play", "squeeze", "think", "find", "look", "look"]

double_action_words = ["listen", "place", "light", "hit", "extiguish", "play", "burn", "play", "reflect", "walk"]

riddle_keys = {}

room_descs_riddles = []

# Rooms without riddles
for desc in room_desc:
    room_descs_riddles.append({
        "description": desc,
        "riddles": [],
        "hints": [],
        "magic_items": [],
        "action_words": ()
    })
    #room_descs_riddles.append(room_desc_riddle_hints_items_action)

# Rooms with one riddle
for i, desc in enumerate(riddle_room_desc):
    room_descs_riddles.append({
        "description": desc,
        "riddles": [riddles[i]], #make sure this is a list
        "hints": [hints[i]],    #make sure this is a list
        "magic_items": [riddle_magic_items[i]],     #make sure this is a list
        "action_words": riddle_action_words[i]
    })
    #room_descs_riddles.append(room_desc_riddle_hints_items_action)

# Rooms with riddle with two items  
for i, desc in enumerate(doulbe_riddle_room_desc):
    room_descs_riddles.append({
        "description": desc,
        "riddles": [double_riddles[i]],
        "hints": double_hints[i],
        "magic_items": double_magic_items[i],
        "action_words": double_action_words[i]
    })
    #room_descs_riddles.append(room_desc_riddle_hints_items_action)

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

weapons = ["Sword", "Axe", "Bow", "Dagger", "Bow and Arrows", "Mace", "Warhammer", "Crossbow", "Flaming Torch", "Staff", "Spear", 
           "Halberd", "Flail", "Scimitar", "Glaive", "Longbow", "Shortsword", "Greatsword", "Battleaxe", "Morningstar", "Rapier", "Katana", 
           "Falchion", "Trident", "Javelin", "Pike", "Lance", "Longsword", "Claymore", "Sabre", "Cutlass", "Estoc", "Scythe", "Khopesh", "Dirk", "Machete", 
           "Cudgel", "Club", "Quarterstaff", "Bastard Sword", "Broadsword", "Battle Axe", "Stiletto", 
           "Arrows and Quiver", "Poison Dagger", "Spiked Mace", "Sledgehammer", "Repeating Crossbow", "Burning Torch", "Flanged Mace", "Wizard's Staff", 
           "Bolt Thrower", "Poleaxe", "Chain Flail", "Curved Sword", "Polearm", "Recurve Bow", "Short Blade", "Two-handed Sword", 
           "War Axe", "Spiked Morningstar", "Fencing Sword", "Samurai Sword", "Broad Falchion", "Three-pronged Spear", "Throwing Spear", "Long Pike", 
           "Cavalry Lance", "Knight's Sword", "Two-handed Claymore", "Cavalry Sabre", "Pirate's Cutlass", "Thrusting Sword", "Reaper's Scythe", 
           "Egyptian Sword", "Stabbing Dirk", "Jungle Machete", "Heavy Cudgel", "Wooden Club", "Monk's Quarterstaff", "Hand-and-a-half Sword"]

armors = ["Chainmail", "Plate Armor", "Leather Armor", "Scale Mail", "Chain Shirt", "Breastplate", "Splint Armor", "Ring Mail", "Studded Leather", "Padded Armor", "Shield", "Small Shield", "Knight's Shield", "Castle Shield"]

treasures = ["Gold Coins", "Diamonds", "Emerald Necklace", "Ancient Artifact", "Royal Crown", "Silver Chalice", "Jeweled Scepter", 
             "Precious Gems", "Golden Statue", "Rare Books", 
             "Ruby Ring", "Sapphire Bracelet", "Platinum Brooch", "Ivory Figurine", "Silk Tapestry", "Pearl Earrings", "Jade Idol", 
             "Bronze Mirror", "Crystal Vase", "Leather-bound Tome", "Engraved Locket", "Golden Goblet", "Silver Scepter", "Ornate Chest", 
             "Exquisite Painting", "Ancient Scroll", "Rare Manuscript",
             "Sacred Chalice", "Royal Diadem", "Ancient Coin", "Exotic Spices", "Silk Robes", "Golden Harp", "Ivory Horn", "Emerald Amulet", "Golden Doubloons", 
             "Uncut Diamonds", "Sapphire Necklace", "King's Crown", "Silver Goblet", "Gem-encrusted Scepter", 
             "Queen's Crown", "Rare Gemstones", "Golden Chalice", "Bronze Statue", "Hidden Map", "Ancient Tomes", 
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

