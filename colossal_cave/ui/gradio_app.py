"""Gradio web UI for Colossal Cave — separated from game engine for testability."""

import gradio as gr


def build_and_launch_ui(
    talk_to_functions,
    list_of_saved_games,
    list_of_llms,
    list_of_diffusers,
    ensure_mflux_installed_fn,
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

    with gr.Blocks(theme=gr.themes.Default(font=[gr.themes.GoogleFont("IBM Plex Mono")])) as web_interface:
        with gr.Row():
            gr.Markdown("<h1 style='text-align: center;'>JMR's Colossal Cave Adventure</h1>")
        with gr.Row():
            with gr.Column():
                image_output = gr.Gallery(label="Room Image")
                video_output = gr.Video(label="Room Video", format="mp4")
                text_input = gr.Textbox(autofocus=True, label="What do you want to do>")
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

                submit_button = gr.Button("Submit", visible=False)
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
            game_state_selector, video_output,
        ]

        text_input.submit(fn=talk_to_functions, inputs=submit_inputs, outputs=submit_outputs)
        submit_button.click(fn=talk_to_functions, inputs=submit_inputs, outputs=submit_outputs)

    ensure_mflux_installed_fn()
    try:
        web_interface.launch(share=True)
    except Exception:
        web_interface.launch(share=False)
