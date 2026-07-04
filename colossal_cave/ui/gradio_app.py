"""Gradio web UI for Colossal Cave — separated from game engine for testability."""

import glob
import os

import gradio as gr


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
    Dependencies injected from Colossal_Cave_Aug_2_25.py to avoid circular imports.
    """
    if not list_of_saved_games:
        list_of_saved_games = ["No saved games"]
    if not list_of_llms or list_of_llms == ["No Ollama models found"]:
        list_of_llms = ["No Ollama models found - please install models with 'ollama pull <model>'"]
    if not list_of_diffusers:
        list_of_diffusers = ["No saved diffusers"]

    # CHANGE (UX): detect quicksave for Resume button
    has_quicksave = False
    if game_dir and os.path.isdir(game_dir):
        has_quicksave = any("quicksave" in f.lower() for f in os.listdir(game_dir) if f.endswith(".pkl"))

    # CHANGE (UX): fixed-height scrollable adventure log below image/video, above command prompt
    adventure_log_css = """
    #adventure-log-scroll textarea {
        height: 220px !important;
        max-height: 220px !important;
        overflow-y: auto !important;
        resize: none !important;
    }
    """

    with gr.Blocks(
        theme=gr.themes.Default(font=[gr.themes.GoogleFont("IBM Plex Mono")]),
        css=adventure_log_css,
    ) as web_interface:
        with gr.Row():
            gr.Markdown("<h1 style='text-align: center;'>JMR's Colossal Cave Adventure</h1>")
        with gr.Row():
            with gr.Column():
                status_header_output = gr.Markdown(label="Status")
                image_output = gr.Gallery(label="Room Image")
                video_output = gr.Video(label="Room Video", format="mp4")
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
                    llm_select = gr.Dropdown(list_of_llms, visible=False, label="Select LLM", value=list_of_llms[0])
                    diff_select = gr.Dropdown(list_of_diffusers, visible=False, label="Select a Diffuser", value=list_of_diffusers[0])

                submit_button = gr.Button("Submit", visible=True)
                resume_button = gr.Button("Resume last run", visible=has_quicksave, interactive=has_quicksave)
            with gr.Column():
                room_desc_output = gr.Textbox(label="Room Description")
                contents_output = gr.Textbox(label="Room Contents")
                console_output = gr.Textbox(label="Console")
                status_output = gr.Textbox(label="Health, Wealth, Points, Killed")
                with gr.Row():
                    map_image_output = gr.Image(label=" ")
                    with gr.Column():
                        rooms_explored = gr.Textbox(label="Rooms Explored")
                        inventory_out = gr.Textbox(label="Inventory")
                long_description_output = gr.Textbox(label="Room Long Description")

        def update_game_selector(game_state):
            if game_state == "Load-Game":
                return (gr.Dropdown(visible=True), gr.Textbox(visible=False), gr.Textbox(visible=False),
                        gr.Checkbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False),
                        gr.Button(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=True),
                        gr.Textbox(visible=True), gr.Checkbox(visible=False))
            elif game_state == "Save-Game":
                return (gr.Dropdown(visible=False), gr.Textbox(visible=True), gr.Textbox(visible=False),
                        gr.Checkbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False),
                        gr.Button(visible=True), gr.Dropdown(visible=False), gr.Dropdown(visible=False),
                        gr.Textbox(visible=False), gr.Checkbox(visible=False))
            elif game_state == "New-Game":
                return (gr.Dropdown(visible=False), gr.Textbox(visible=True), gr.Textbox(visible=True),
                        gr.Checkbox(visible=True), gr.Checkbox(visible=True), gr.Checkbox(visible=True),
                        gr.Button(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=True),
                        gr.Textbox(visible=True), gr.Checkbox(visible=True, interactive=True))
            elif game_state == "Play-Game":
                return (gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False),
                        gr.Checkbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False),
                        gr.Button(visible=True), gr.Dropdown(visible=False), gr.Dropdown(visible=False),
                        gr.Textbox(visible=False), gr.Checkbox(visible=False))
            else:
                return (gr.Dropdown(visible=False), gr.Textbox(visible=False), gr.Textbox(visible=False),
                        gr.Checkbox(visible=False), gr.Checkbox(visible=False), gr.Checkbox(visible=False),
                        gr.Button(visible=False), gr.Dropdown(visible=False), gr.Dropdown(visible=False),
                        gr.Textbox(visible=False), gr.Checkbox(visible=False))

        game_state_selector.change(
            update_game_selector, inputs=[game_state_selector],
            outputs=[game_selector, name_of_game, number_of_rooms, room_npc_desc_art, monster_desc_art,
                     item_desc_art, submit_button, llm_select, diff_select, art_style, generate_videos],
        )

        submit_inputs = [
            text_input, game_selector, difficulty_selector, game_state_selector, name_of_game,
            number_of_rooms, room_npc_desc_art, monster_desc_art, item_desc_art,
            llm_select, diff_select, art_style, generate_videos,
        ]
        submit_outputs = [
            image_output, map_image_output, room_desc_output, contents_output, console_output,
            status_output, rooms_explored, inventory_out, long_description_output,
            game_state_selector, video_output, adventure_log_output, status_header_output,
        ]

        def resume_last_run(difficulty, llm_name, diff_name, art_st, gen_vid):
            """Load the most recent quicksave if present."""
            if not game_dir or not os.path.isdir(game_dir):
                return talk_to_functions("", "", difficulty, "Load-Game", "", "", False, False, False, llm_name, diff_name, art_st, gen_vid)
            quicksaves = sorted(
                [f for f in os.listdir(game_dir) if f.startswith("Adv_quicksave") and f.endswith(".pkl")],
                key=lambda f: os.path.getmtime(os.path.join(game_dir, f)),
                reverse=True,
            )
            if not quicksaves:
                return talk_to_functions("", "", difficulty, "Load-Game", "", "", False, False, False, llm_name, diff_name, art_st, gen_vid)
            # saved_files() strips Adv_ and .pkl — rebuild the name key
            inner = quicksaves[0][4:-4]
            return talk_to_functions("", inner, difficulty, "Load-Game", "", "", False, False, False, llm_name, diff_name, art_st, gen_vid)

        text_input.submit(fn=talk_to_functions, inputs=submit_inputs, outputs=submit_outputs)
        submit_button.click(fn=talk_to_functions, inputs=submit_inputs, outputs=submit_outputs)
        resume_button.click(
            fn=resume_last_run,
            inputs=[difficulty_selector, llm_select, diff_select, art_style, generate_videos],
            outputs=submit_outputs,
        )

    ensure_mflux_installed_fn()
    try:
        web_interface.launch(share=True)
    except Exception:
        web_interface.launch(share=False)
