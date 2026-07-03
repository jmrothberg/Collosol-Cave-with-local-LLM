"""Runtime feature flags (env-overridable)."""

import os

ADVANCED_DIRECTIVES = False

ADV_VERBOSE_DEBUG = os.environ.get("ADV_VERBOSE_DEBUG", "").lower() in {"1", "true", "yes"}

ADV_INJECT_ENGINE_NOTES = os.environ.get("ADV_INJECT_ENGINE_NOTES", "").lower() in {
    "1",
    "true",
    "yes",
}

# CHANGE (organize May 2026): optional atomic Pass-2 sub-passes for small/Ollama models.
USE_ATOMIC_PASS2 = os.environ.get("USE_ATOMIC_PASS2", "").lower() in {"1", "true", "yes"}

DEFAULT_IMAGE_THEME = (
    "Studio Ghibli background art, dark fantasy cavern, painterly matte painting, "
    "soft rim light, volumetric fog, by Studio Ghibli"
)

OLLAMA_CHOICE_PREFIX = "ollama:"
OLLAMA_PREFERRED_MODEL = "qwen3.6:27b-coding-mxfp8"

PRESET_THEMES = {
    "Tolkien Cave Adventure": (
        "An adventure game starting at the entrance to a giant cave system, "
        "featuring Tolkien-inspired characters like wizards, hobbits, dwarves, and elves. "
        "Each character and object should play a meaningful part in completing the quest. "
        "Art style: detailed fantasy illustration with warm torchlight, ancient stone architecture, "
        "and atmospheric mist. Environments should feel lived-in and mysterious."
    ),
    "Star Wars Universe": (
        "An adventure game set in the Star Wars universe with iconic characters like Jedi, "
        "droids, and aliens. Features lightsabers, the Force, and galactic technology. "
        "Each character and item should be essential to completing the mission. "
        "Art style: cinematic sci-fi with neon glows, metallic surfaces, holographic displays, "
        "and dramatic lighting contrasts between light and dark side aesthetics."
    ),
    "Colossal Cave (Classic)": (
        "A classic text adventure inspired by the original Colossal Cave Adventure. "
        "Features underground caverns, twisty passages, treasure hunting, and puzzle solving. "
        "Characters include a helpful hermit, mysterious spirits, and cave-dwelling creatures. "
        "Art style: nostalgic fantasy illustration with glowing crystals, underground rivers, "
        "ancient ruins, and atmospheric cave lighting."
    ),
    "Zork-style Mystery": (
        "A mysterious adventure in the style of Zork, featuring a sprawling underground empire, "
        "ancient artifacts, and clever puzzles. Characters are enigmatic and locations are interconnected. "
        "Art style: dark fantasy with ornate architecture, magical glows, brass and copper machinery, "
        "and deep shadows hiding secrets."
    ),
    "Myst Island": (
        "A puzzle-focused adventure on a mysterious island with strange machines and hidden passages. "
        "Minimal characters but rich environmental storytelling. Each mechanism and clue connects to the solution. "
        "Art style: surreal landscapes, steampunk machinery, crystalline structures, "
        "ethereal lighting, and dreamlike atmosphere."
    ),
    "Pirate Treasure Hunt": (
        "A swashbuckling adventure seeking buried treasure on a tropical island. "
        "Features pirates, sea caves, ancient maps, and hidden troves. "
        "Characters include a helpful parrot, ghostly pirates, and island natives. "
        "Art style: vibrant tropical colors, weathered wood and rope, golden treasure glints, "
        "moonlit beaches, and mysterious jungle ruins."
    ),
    "Cyberpunk Heist": (
        "A high-tech adventure in a neon-lit dystopian city. "
        "Features hackers, corporate secrets, AI companions, and cybernetic enhancements. "
        "Each contact and gadget serves a purpose in the heist. "
        "Art style: neon-soaked cityscapes, holographic interfaces, rain-slicked streets, "
        "chrome and glass architecture, and dramatic urban lighting."
    ),
    "Custom (type below)": "",
}
