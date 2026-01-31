# Demonstration system to take player commands and turn into game actions
import os
import torch
from sentence_transformers import SentenceTransformer, util

import numpy as np
import json

# Load the pre-trained model
#vector_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
vector_model = SentenceTransformer('all-mpnet-base-v2', cache_folder='/Users/jonathanrothberg/')
#vector_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='/Users/jonathanrothberg/')

# Expanded lists of entities
room_desc = ["mysterious cave", "ancient castle", "dark forest", "hidden dungeon", "secret chamber"]
weapons = ["sword", "spear", "bow", "magic wand", "axe", "dagger", "blade", "katana", "staff"]
magic_items = ["magic wand", "healing potion", "magic elixir", "strength brew", "invisibility potion", "health draught", "mana potion"]
treasures = ["golden crown", "ancient artifact", "enchanted necklace", "lost relic", "treasure chest", "priceless statue"]
items = weapons + magic_items + treasures
npcs = ["jacob","nikki","ivy", "gabby", "King Arthur", "Merchant Zac", "Wizard Gandalf", "Queen Guinevere", "Priest Merlin", "village elder", "local lord"]
monsters = ["dragon", "troll", "goblin", "skeleton warrior", "orc", "vampire", "beast", "monster", "creature"]
riddles = ["riddle", "puzzle", "enigma", "mystery", "conundrum", "brainteaser", "problem", "challenge"]
room_items = ["sword","torch", "lantern", "candle", "sconce", "fireplace", "brazier", "candelabra"]
my_items = ["silver coins", "sword", "spear", "bow", "magic wand", "axe", "dagger", "blade", "katana", "staff", "magic wand", "healing potion", "magic elixir", "strength brew", 
            "invisibility potion", "health draught", "mana potion", "golden crown", "ancient artifact", "enchanted necklace", "lost relic", "treasure chest", "priceless statue"]
npc_items = ["gold coins", "silver coins", "copper coins", "platinum coins", "diamonds", "emeralds", "sapphires", "rubies", "jewels", "gems"]

directions = ["north", "south", "east", "west", "up", "down", "left", "right"]

dict_entities = {"room_desc": room_desc, "weapons": weapons, "magic_items": magic_items, "treasures": treasures, "items": items, "npcs": npcs, "monsters": monsters, "riddles": riddles, "room_items": room_items, "my_items": my_items, "npc_items": npc_items, "directions": directions}

dict_entity_embeddings = {}
for entity, entity_list in dict_entities.items():
    dict_entity_embeddings[entity] = vector_model.encode(entity_list)

# Action lists
solve_actions = ["solve", "answer", "work out", "resolve","decipher", "decode"]
attack_actions = ["throw","fight", "attack", "battle", "kill", "shoot", "stab", "hit", "slay", "strike"] 
trade_actions = ["trade", "barter", "exchange", "negotiate", "swap", "bargain", "deal", "commerce"]
talk_actions = ["ask", "tell", "talk", "speak", "chat", "converse", "communicate", "inquire", "question"]
use_actions = ["use","consume", "apply", "utilize","drink", "eat", "swollow" ]
take_actions = ["take", "pick up", "grab", "collect"]
leave_actions = ["leave","drop", "discard", "abandon"]
study_actions = ["describe","examine", "inspect", "look at", "check", "study", "observe"]
go_actions = ["go", "go [direction]", "go right", "go left", "move", "walk", "travel", "proceed", "advance", "step", "journey", "venture", "progress", "head",
              "go north", "move south", "walk east", "travel west", "proceed up", "advance down", "step left", "journey right"]

dict_actions = {"solve": solve_actions, "attack": attack_actions, "trade": trade_actions, "talk": talk_actions, "use": use_actions, "take": take_actions, "leave": leave_actions, "study": study_actions, "go": go_actions}

dict_action_embeddings = {}      
for action, action_list in dict_actions.items():
    dict_action_embeddings[action] = vector_model.encode(action_list)


# Function to find the most similar action
def find_similar_action(user_input):
    user_input_embedding = vector_model.encode(user_input)
    best_action = None
    best_score = 0
    extracted_action = {}
    for action_type, embeddings in dict_action_embeddings.items():
        distances = util.pytorch_cos_sim(user_input_embedding, embeddings)
        # This line finds the maximum similarity score and its index in the tensor of similarity scores.
        max_score, max_index = torch.max(distances, dim=1)
        action_list = dict_actions[action_type]  # Assuming you have a list of actions for each action type
        if max_score > 0.3:
            print(f"Best: {action_type} {max_score.item():.2f} Extracted Action {action_list[max_index.item()]}")
        if max_score > best_score and max_score > 0.2:  # Threshold for considering a match
            best_action = action_type
            best_score = max_score
            extracted_action[action_type] = action_list[max_index.item()]
    print (f'Best Action: {best_action} {best_score.item():.2f}, Extracted Action: {extracted_action[best_action]} \n')
    return best_action


# Function to extract entities from user input
def extract_entities(user_input):
    user_input_embedding = vector_model.encode(user_input)
    extracted_entities = {}
    for entity_type, embeddings in dict_entity_embeddings.items():
        distances = util.pytorch_cos_sim(user_input_embedding, embeddings)
        best_match_score, best_index = torch.max(distances, dim=1)
        if best_match_score > 0.3:  # Threshold for considering a match
            # This line retrieves the list of entities of the current type using the global() function and the name of the entity type.
            entity_list = dict_entities[entity_type]
            extracted_entities[entity_type] = entity_list[best_index.item()]
            print(f'Best Entities: {entity_type} {best_match_score.item():.2f}, Extracted Entity: {extracted_entities[entity_type]}')
    return extracted_entities


def go(direction=None):
    return f"Going {direction}"

def fight(monster=None, weapon=None):
    return f"Fighting {monster} with {weapon}" 

def trade(npc=None, offer_item=None, request_item=None):
    return f"Trading {offer_item} with {npc} for {request_item}" 

def talk(npc=None, message=""):
    return f"Talking to {npc}. Message: {message}" 

def use(item=None):
    return f"Using {item}" 

def take(item=None):
    return f"Taking {item}" 

def leave(item=None):
    return f"Leaving {item}" 

def examine_item(item=None):
    return f"study item {item}" 

def examine_room(room=None):
    return f"study room {room}."

def solve_puzzle(riddle = None, magic=None):
    return f"Solving the {riddle} with {magic}"

# Function to process player input
def process_input(user_input):
     # Determine the action
    chosen_action = find_similar_action(user_input)
    # Extract entities
    entities_by_type = extract_entities(user_input)

       # Call the corresponding function based on the action and extracted entities
    if chosen_action == "go":
        return go(entities_by_type.get("directions"))
    if chosen_action == "attack":
        return fight(entities_by_type.get("monsters"), entities_by_type.get("weapons"))
    elif chosen_action == "trade":
        npc = entities_by_type.get("npcs")
        offer_item = entities_by_type.get("my_items")
        request_item = entities_by_type.get("npc_items")
        return trade(npc, offer_item, request_item)
    elif chosen_action == "talk":
        return talk(entities_by_type.get("npcs"), user_input)
    elif chosen_action == "use":
        return use(entities_by_type.get("magic_items"))
    elif chosen_action == "take":
        return take(entities_by_type.get("room_items"))
    elif chosen_action == "leave":
        return leave(entities_by_type.get("my_items"))
    elif chosen_action == "solve":
        return solve_puzzle( entities_by_type.get("riddles"), entities_by_type.get("my_items"))
    elif chosen_action == "study":
        # First check for a room to examine
        room = entities_by_type.get("room_desc")
        if room:
            return examine_room(room)
        else:
            # If no room is found, examine the item
            item = entities_by_type.get("room_items")
            return examine_item(item)
    else:
        return "Action not recognized or insufficient data for action."

# Define action synonyms if you want to compare to an average vector... i thtink this is a mistake. 

# Example usage
example_inputs = [
    "Kill the dragon with a sword.",
    "Throw the spear at the goblin.",
    "Fight the dragon with a sword.",
    "Trade gold coins with Merchant Zac for a magic elixir.",
    "Talk to King Arthur.",
    "Use the healing potion.",
    "Take the golden crown.",
    "Leave the torch.",
    "Examine the ancient artifact.",
    "Examine the mysterious cave.",
    "Examine the shield.",
    "trade gold for silver with zach",
    "Tell zac to give me a potion",
    "apply the potion",
    "solve the riddle with the magic wand",
    "figure out the puzzle with the magic wand",
    "ask zac to trade a potion for a sword",
    "hit the ugly dragon with the big sword",
    "trade the gold for the silver with zach",
    "tell zac I want to trade potion for magic wand",
    "solve the brain teaser with the purple magic wand",
    "go west",
    "move south",
    "walk east",
    "head north",
    "travel west",
    "kill the elk with the spear",
    "throw the spear at the elk",
    "kill the dinasour with the spear",
    "throw the spear at the dinasour",
    "stab the goblin with the sword",
    "shove the knife into the goblin",
    "attack the goblin with the sword",
    "trade the silver for the gold with jacob",
    "trade with jacob your gold for his silver",
    "trade my silver for jacob's gold",
]

for user_input in example_inputs:
    result = process_input(user_input)
    print(f"Player: \"{user_input}\"")
    print(f"Game: \"{result}\"\n")

while True:
    user_input = input("Enter a command>")
    result = process_input(user_input)
    print(f"Player:{user_input}")
    print(f"Game: {result}\n\n")
