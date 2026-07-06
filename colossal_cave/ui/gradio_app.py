"""Gradio web UI for Colossal Cave — separated from game engine for testability."""

import glob
import os

import gradio as gr


def _default_play_llm_choice(list_of_llms):
    """Prefer gemma-4-12B-it-8bit when listed."""
    preferred = os.environ.get("COLOSSAL_CAVE_PLAY_LLM", "gemma-4-12B-it-8bit")
    if preferred in list_of_llms:
        return preferred
    return list_of_llms[0]


def build_and_launch_ui(
    talk_to_functions,
    list_of_saved_games,
    list_of_llms,
    list_of_diffusers,
    ensure_mflux_installed_fn,
    game_dir=None,
):
    """
    Build Gradio Blocks and launch the web interface.
    Dependencies injected from colossal_cave_July_5_26.py to avoid circular imports.
    """
    if not list_of_saved_games:
        list_of_saved_games = ["No saved games"]
    if not list_of_llms or list_of_llms == ["No Ollama models found"]:
        list_of_llms = ["No Ollama models found - please install models with 'ollama pull <model>'"]
    if not list_of_diffusers:
        list_of_diffusers = ["No saved diffusers"]

    play_default = _default_play_llm_choice(list_of_llms)
    creation_default = list_of_llms[0]

    # CHANGE (UX): detect quicksave for Resume button
    has_quicksave = False
    if game_dir and os.path.isdir(game_dir):
        has_quicksave = any("quicksave" in f.lower() for f in os.listdir(game_dir) if f.endswith(".pkl"))

    # CHANGE (UX): fixed-height scrollable adventure log below gallery, above command prompt
    adventure_log_css = """
    #adventure-log-scroll textarea {
        height: 220px !important;
        max-height: 220px !important;
        overflow-y: auto !important;
        resize: none !important;
    }
    """
    # CHANGE (UX): auto-scroll adventure log on update — head script only (never js= on Gradio events)
    adventure_log_head = """
    <script>
    (function() {
        function scrollAdventureLog() {
            const root = document.getElementById('adventure-log-scroll');
            const el = root && root.querySelector('textarea');
            if (el) { el.scrollTop = el.scrollHeight; }
        }
        const obs = new MutationObserver(scrollAdventureLog);
        function watch() {
            const root = document.getElementById('adventure-log-scroll');
            if (!root) { setTimeout(watch, 300); return; }
            obs.observe(root, {subtree: true, childList: true, characterData: true});
            scrollAdventureLog();
        }
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', watch);
        } else {
            watch();
        }
    })();
    </script>
    """
    ui_theme = gr.themes.Default(font=[gr.themes.GoogleFont("IBM Plex Mono")])

    with gr.Blocks() as web_interface:
        with gr.Row():
            gr.Markdown("<h1 style='text-align: center;'>JMR's Colossal Cave Adventure</h1>")
        with gr.Row():
            with gr.Column():
                image_output = gr.Gallery(label="Room Image")
                # CHANGE (UX): status summary below gallery, above adventure log
                status_header_output = gr.Markdown(label="Status")
                adventure_log_output = gr.Textbox(
                    label="Adventure Log",
                    lines=10,
                    max_lines=10,
                    interactive=False,
                    elem_id="adventure-log-scroll",
                )
                text_input = gr.Textbox(autofocus=True, label="What do you want to do>", placeholder="Press Enter to act")
                difficulty_selector = gr.Radio(["Cheat", "Easy", "Medium", "Hard"], value="Easy", label="Difficulty")
                game_state_selector = gr.Radio(["Play-Game", "Load-Game", "New-Game", "Save-Game"], label="Game State")

                with gr.Row():
                    game_selector = gr.Dropdown(list_of_saved_games, visible=False, label="Select a saved game to load", value=list_of_saved_games[0])
                    name_of_game = gr.Textbox(visible=False, label="Name of game")
                    number_of_rooms = gr.Textbox(visible=False, label="Number of rooms")
                    art_style = gr.Textbox(visible=False, label="Art Style")
                with gr.Row():
                    room_npc_desc_art = gr.Checkbox(visible=False, value=True, label="Room & NPC Art and Description")
                    generate_videos = gr.Checkbox(label="Generate Videos", value=False, interactive=True)
                    monster_desc_art = gr.Checkbox(visible=False, value=True, label="Monster Art & Description")
                    item_desc_art = gr.Checkbox(visible=False, value=True, label="Item Art and Description")
                with gr.Row():
                    # CHANGE (speed): creation LLM for New-Game; play LLM for turns
                    creation_llm_select = gr.Dropdown(
                        list_of_llms, visible=False, label="Creation LLM (new game)", value=creation_default,
                    )
                    play_llm_select = gr.Dropdown(
                        list_of_llms, visible=False, label="Play LLM (fast)", value=play_default,
                    )
                    diff_select = gr.Dropdown(list_of_diffusers, visible=False, label="Select a Diffuser", value=list_of_diffusers[0])

                submit_button = gr.Button("Submit", visible=True)
                resume_button = gr.Button("Resume last run", visible=has_quicksave, interactive=has_quicksave)
            with gr.Column():
                # CHANGE (UX): room video at top of right column
                video_output = gr.Video(label="Room Video", format="mp4")
                # CHANGE (UX): map + rooms explored + inventory above room description
                with gr.Row():
                    map_image_output = gr.Image(label=" ")
                    with gr.Column():
                        rooms_explored = gr.Textbox(label="Rooms Explored")
                        inventory_out = gr.Textbox(label="Inventory")
                room_desc_output = gr.Textbox(label="Room Description")
                contents_output = gr.Textbox(label="Room Contents")
                console_output = gr.Textbox(label="Console")
                status_output = gr.Textbox(label="Health, Wealth, Points, Killed")
                long_description_output = gr.Textbox(label="Room Long Description")

        def update_game_selector(game_state):
            if game_state == "Load-Game":
                return (gr.Dropdown(visible=True), gr.Textbox(visible=False), gr.Textbox(visible=False),
                        gr.Checkbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False),
                        gr.Button(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=True),
                        gr.Dropdown(visible=True), gr.Textbox(visible=True), gr.Checkbox(visible=False))
            elif game_state == "Save-Game":
                return (gr.Dropdown(visible=False), gr.Textbox(visible=True), gr.Textbox(visible=False),
                        gr.Checkbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False),
                        gr.Button(visible=True), gr.Dropdown(visible=False), gr.Dropdown(visible=False),
                        gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False))
            elif game_state == "New-Game":
                return (gr.Dropdown(visible=False), gr.Textbox(visible=True), gr.Textbox(visible=True),
                        gr.Checkbox(visible=True), gr.Checkbox(visible=True), gr.Checkbox(visible=True),
                        gr.Button(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=True),
                        gr.Dropdown(visible=True), gr.Textbox(visible=True), gr.Checkbox(visible=True, interactive=True))
            elif game_state == "Play-Game":
                return (gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False),
                        gr.Checkbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False),
                        gr.Button(visible=True), gr.Dropdown(visible=False), gr.Dropdown(visible=True),
                        gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False))
            else:
                return (gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False),
                        gr.Checkbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False),
                        gr.Button(visible=False), gr.Dropdown(visible=False), gr.Dropdown(visible=False),
                        gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Checkbox(visible=False))

        game_state_selector.change(
            update_game_selector, inputs=[game_state_selector],
            outputs=[game_selector, name_of_game, number_of_rooms, room_npc_desc_art, monster_desc_art,
                     item_desc_art, submit_button, creation_llm_select, play_llm_select, diff_select,
                     art_style, generate_videos],
        )

        submit_inputs = [
            text_input, game_selector, difficulty_selector, game_state_selector, name_of_game,
            number_of_rooms, room_npc_desc_art, monster_desc_art, item_desc_art,
            creation_llm_select, diff_select, art_style, generate_videos, play_llm_select,
        ]
        submit_outputs = [
            image_output, map_image_output, room_desc_output, contents_output, console_output,
            status_output, rooms_explored, inventory_out, long_description_output,
            game_state_selector, video_output, adventure_log_output, status_header_output,
        ]

        def resume_last_run(difficulty, creation_llm, diff_name, art_st, gen_vid, play_llm):
            """Load the most recent quicksave if present."""
            if not game_dir or not os.path.isdir(game_dir):
                return talk_to_functions("", "", difficulty, "Load-Game", "", "", False, False, False,
                                         creation_llm, diff_name, art_st, gen_vid, play_llm)
            quicksaves = sorted(
                [f for f in os.listdir(game_dir) if f.startswith("Adv_quicksave") and f.endswith(".pkl")],
                key=lambda f: os.path.getmtime(os.path.join(game_dir, f)),
                reverse=True,
            )
            if not quicksaves:
                return talk_to_functions("", "", difficulty, "Load-Game", "", "", False, False, False,
                                         creation_llm, diff_name, art_st, gen_vid, play_llm)
            inner = quicksaves[0][4:-4]
            return talk_to_functions("", inner, difficulty, "Load-Game", "", "", False, False, False,
                                     creation_llm, diff_name, art_st, gen_vid, play_llm)

        text_input.submit(fn=talk_to_functions, inputs=submit_inputs, outputs=submit_outputs)
        submit_button.click(fn=talk_to_functions, inputs=submit_inputs, outputs=submit_outputs)
        resume_button.click(
            fn=resume_last_run,
            inputs=[difficulty_selector, creation_llm_select, diff_select, art_style, generate_videos, play_llm_select],
            outputs=submit_outputs,
        )

    ensure_mflux_installed_fn()
    launch_kw = dict(css=adventure_log_css, theme=ui_theme, head=adventure_log_head)
    try:
        web_interface.launch(share=True, **launch_kw)
    except Exception:
        web_interface.launch(share=False, **launch_kw)
