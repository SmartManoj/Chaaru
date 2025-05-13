import json
import os
import shutil
import tempfile
import time
import uuid
from io import BytesIO
from threading import Timer
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from e2b_desktop import Sandbox
from gradio_modal import Modal
from huggingface_hub import login, upload_folder
from PIL import Image
from smolagents import CodeAgent, InferenceClientModel
from smolagents.gradio_ui import GradioUI

from e2bqwen import E2BVisionAgent, get_agent_summary_erase_images
from gradio_script import stream_to_gradio
from scripts_and_styling import (
    CUSTOM_JS,
    FOOTER_HTML,
    SANDBOX_CSS_TEMPLATE,
    SANDBOX_HTML_TEMPLATE,
    apply_theme,
)

load_dotenv(override=True)


EXAMPLES = [
    "Use Google Maps to find the Hugging Face HQ in Paris",
    "Go to Wikipedia and find what happened on April 4th",
    "Find out the travel time by train from Bern to Basel on Google Maps",
    "Go to Hugging Face Spaces and then find the Space flux.1 schnell. Use the space to generate an image with the prompt 'a field of gpus'",
]

E2B_API_KEY = os.getenv("E2B_API_KEY")
SANDBOXES: dict[str, Sandbox] = {}
SANDBOX_METADATA: dict[str, dict[str, Any]] = {}
SANDBOX_TIMEOUT = 300
WIDTH = 1280
HEIGHT = 960
TMP_DIR = "./tmp/"
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
login(token=hf_token)

custom_css = SANDBOX_CSS_TEMPLATE.replace("<<WIDTH>>", str(WIDTH + 15)).replace(
    "<<HEIGHT>>", str(HEIGHT + 10)
)

sandbox_html_template = SANDBOX_HTML_TEMPLATE.replace(
    "<<WIDTH>>", str(WIDTH + 15)
).replace("<<HEIGHT>>", str(HEIGHT + 10))


def upload_to_hf_and_remove(folder_paths: list[str]):
    repo_id = "smolagents/computer-agent-logs-2"

    with tempfile.TemporaryDirectory(dir=TMP_DIR) as temp_dir:
        print(f"Preparing to upload {len(folder_paths)} folders to {repo_id}...")

        # Copy all folders into the temporary directory
        for folder_path in folder_paths:
            folder_name = os.path.basename(os.path.normpath(folder_path))
            target_path = os.path.join(temp_dir, folder_name)
            print(f"Copying {folder_path} to temporary directory...")
            shutil.copytree(folder_path, target_path)
            # Remove the original folder after copying
            shutil.rmtree(folder_path)
            print(f"Original folder {folder_path} removed.")

        # Upload the entire temporary directory
        print(f"Uploading all folders to {repo_id}...")
        upload_folder(
            folder_path=temp_dir,
            repo_id=repo_id,
            repo_type="dataset",
            ignore_patterns=[".git/*", ".gitignore"],
        )
        print("Upload complete.")

        return f"Successfully uploaded {len(folder_paths)} folders to {repo_id}"


def cleanup_sandboxes():
    """Remove sandboxes that haven't been accessed for longer than SANDBOX_TIMEOUT"""
    current_time = time.time()
    sandboxes_to_remove = []

    for session_id, metadata in SANDBOX_METADATA.items():
        if current_time - metadata["last_accessed"] > SANDBOX_TIMEOUT:
            sandboxes_to_remove.append(session_id)

    for session_id in sandboxes_to_remove:
        if session_id in SANDBOXES:
            try:
                # Upload data before removing if needed
                data_dir = os.path.join(TMP_DIR, session_id)
                if os.path.exists(data_dir):
                    upload_to_hf_and_remove(data_dir)

                # Close the sandbox
                SANDBOXES[session_id].kill()
                del SANDBOXES[session_id]
                del SANDBOX_METADATA[session_id]
                print(f"Cleaned up sandbox for session {session_id}")
            except Exception as e:
                print(f"Error cleaning up sandbox {session_id}: {str(e)}")


def get_or_create_sandbox(session_uuid):
    current_time = time.time()

    if (
        session_uuid in SANDBOXES
        and session_uuid in SANDBOX_METADATA
        and current_time - SANDBOX_METADATA[session_uuid]["created_at"]
        < SANDBOX_TIMEOUT
    ):
        print(f"Reusing Sandbox for  {session_uuid}")
        SANDBOX_METADATA[session_uuid]["last_accessed"] = current_time
        return SANDBOXES[session_uuid]
    else:
        print("No sandbox found, creating a new one")

    if session_uuid in SANDBOXES:
        try:
            print(f"Closing expired sandbox for session {session_uuid}")
            SANDBOXES[session_uuid].kill()
        except Exception as e:
            print(f"Error closing expired sandbox: {str(e)}")

    print(f"Creating new sandbox for session {session_uuid}")
    desktop = Sandbox(
        api_key=E2B_API_KEY,
        resolution=(WIDTH, HEIGHT),
        dpi=96,
        timeout=SANDBOX_TIMEOUT,
        template="k0wmnzir0zuzye6dndlw",
    )
    desktop.stream.start(require_auth=True)
    setup_cmd = """sudo mkdir -p /usr/lib/firefox-esr/distribution && echo '{"policies":{"OverrideFirstRunPage":"","OverridePostUpdatePage":"","DisableProfileImport":true,"DontCheckDefaultBrowser":true}}' | sudo tee /usr/lib/firefox-esr/distribution/policies.json > /dev/null"""
    desktop.commands.run(setup_cmd)

    print(f"Sandbox ID for session {session_uuid} is {desktop.sandbox_id}.")

    SANDBOXES[session_uuid] = desktop
    SANDBOX_METADATA[session_uuid] = {
        "created_at": current_time,
        "last_accessed": current_time,
    }
    return desktop


def update_html(interactive_mode: bool, session_uuid):
    desktop = get_or_create_sandbox(session_uuid)
    auth_key = desktop.stream.get_auth_key()
    base_url = desktop.stream.get_url(auth_key=auth_key)
    stream_url = base_url if interactive_mode else f"{base_url}&view_only=true"

    status_class = "status-interactive" if interactive_mode else "status-view-only"
    status_text = "Interactive" if interactive_mode else "Agent running..."
    creation_time = (
        SANDBOX_METADATA[session_uuid]["created_at"]
        if session_uuid in SANDBOX_METADATA
        else time.time()
    )

    sandbox_html_content = sandbox_html_template.format(
        stream_url=stream_url,
        status_class=status_class,
        status_text=status_text,
    )
    sandbox_html_content += f'<div id="sandbox-creation-time" style="display:none;" data-time="{creation_time}" data-timeout="{SANDBOX_TIMEOUT}"></div>'
    return sandbox_html_content


def generate_interaction_id(session_uuid):
    return f"{session_uuid}_{int(time.time())}"


def save_final_status(folder, status: str, summary, error_message=None) -> None:
    with open(os.path.join(folder, "metadata.jsonl"), "a") as output_file:
        output_file.write(
            "\n"
            + json.dumps(
                {"status": status, "summary": summary, "error_message": error_message},
            )
        )


def extract_browser_uuid(js_uuid):
    print(f"[BROWSER] Got browser UUID from JS: {js_uuid}")
    return js_uuid


def initialize_session(interactive_mode, browser_uuid):
    if not browser_uuid:
        new_uuid = str(uuid.uuid4())
        print(f"[LOAD] No UUID from browser, generating: {new_uuid}")
        return update_html(interactive_mode, new_uuid), new_uuid
    else:
        print(f"[LOAD] Got UUID from browser: {browser_uuid}")
        return update_html(interactive_mode, browser_uuid), browser_uuid


def create_agent(data_dir, desktop):
    model = InferenceClientModel(
        model_id="https://n5wr7lfx6wp94tvl.us-east-1.aws.endpoints.huggingface.cloud",
        token=hf_token,
    )

    # model = OpenAIServerModel(
    #     "gpt-4o",api_key=os.getenv("OPENAI_API_KEY")
    # )
    return E2BVisionAgent(
        model=model,
        data_dir=data_dir,
        desktop=desktop,
        max_steps=20,
        verbosity_level=2,
        # planning_interval=10,
        use_v1_prompt=True,
    )


INTERACTION_IDS = {}


class EnrichedGradioUI(GradioUI):
    def log_user_message(self, text_input):
        import gradio as gr

        return (
            text_input,
            gr.Button(interactive=False),
        )

    def interact_with_agent(
        self,
        task_input,
        stored_messages,
        session_state,
        session_uuid,
        consent_storage,
        request: gr.Request,
    ):
        interaction_id = generate_interaction_id(session_uuid)
        desktop = get_or_create_sandbox(session_uuid)
        INTERACTION_IDS[interaction_id] = session_uuid

        data_dir = os.path.join(TMP_DIR, interaction_id)

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Always re-create an agent from scratch, else Qwen-VL gets confused with past history
        session_state["agent"] = create_agent(data_dir=data_dir, desktop=desktop)

        try:
            stored_messages.append(
                gr.ChatMessage(
                    role="user", content=task_input, metadata={"status": "done"}
                )
            )
            yield stored_messages

            with open(os.path.join(data_dir, "metadata.jsonl"), "w") as output_file:
                output_file.write(
                    json.dumps(
                        {"task": task_input},
                    )
                )

            screenshot_bytes = session_state["agent"].desktop.screenshot(format="bytes")
            initial_screenshot = Image.open(BytesIO(screenshot_bytes))
            for msg in stream_to_gradio(
                session_state["agent"],
                task=task_input,
                reset_agent_memory=False,
                task_images=[initial_screenshot],
            ):
                if (
                    hasattr(session_state["agent"], "last_marked_screenshot")
                    and isinstance(msg, gr.ChatMessage)
                    and msg.content == "-----"
                ):  # Append the last screenshot before the end of step
                    stored_messages.append(
                        gr.ChatMessage(
                            role="assistant",
                            content={
                                "path": session_state[
                                    "agent"
                                ].last_marked_screenshot.to_string(),
                                "mime_type": "image/png",
                            },
                            metadata={"status": "done"},
                        )
                    )
                if isinstance(msg, gr.ChatMessage):
                    stored_messages.append(msg)
                elif isinstance(msg, str):  # Then it's only a completion delta
                    try:
                        if stored_messages[-1].metadata["status"] == "pending":
                            stored_messages[-1].content = msg
                        else:
                            stored_messages.append(
                                gr.ChatMessage(
                                    role="assistant",
                                    content=msg,
                                    metadata={"status": "pending"},
                                )
                            )
                    except Exception as e:
                        raise e
                yield stored_messages

            status = "completed"
            yield stored_messages

        except Exception as e:
            error_message = f"Error in interaction: {str(e)}"
            print(error_message)
            stored_messages.append(
                gr.ChatMessage(
                    role="assistant", content="Run failed:\n" + error_message
                )
            )
            status = "failed"
            yield stored_messages
        finally:
            if consent_storage and task_input not in EXAMPLES:
                summary = get_agent_summary_erase_images(session_state["agent"])
                save_final_status(
                    data_dir, status, summary=summary, error_message=error_message
                )


theme = gr.themes.Default(
    font=["Oxanium", "sans-serif"], primary_hue="amber", secondary_hue="blue"
)

# Create a Gradio app with Blocks
with gr.Blocks(theme=theme, css=custom_css, js=CUSTOM_JS) as demo:
    # Storing session hash in a state variable
    session_uuid_state = gr.State(None)
    print("Starting the app!")
    with gr.Row():
        sandbox_html = gr.HTML(
            value=sandbox_html_template.format(
                stream_url="",
                status_class="status-interactive",
                status_text="Interactive",
            ),
            label="Output",
        )
        with gr.Sidebar(position="left"):
            with Modal(visible=True) as modal:
                gr.Markdown("""### Welcome to smolagent's Computer agent demo 🖥️
In this app, you'll be able to interact with an agent powered by [smolagents](https://github.com/huggingface/smolagents) and [Qwen-VL](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct).

👉 Type a task in the left sidebar, click the button, and watch the agent solving your task. ✨

_Please note that we store the task logs by default so **do not write any personal information**; you can uncheck the logs storing on the task bar._
""")
            task_input = gr.Textbox(
                placeholder="Find me pictures of cute puppies",
                label="Enter your task below:",
                elem_classes="primary-color-label",
            )

            run_btn = gr.Button("Let's go!", variant="primary")

            gr.Examples(
                examples=EXAMPLES,
                inputs=task_input,
                label="Example Tasks",
                examples_per_page=4,
            )

            session_state = gr.State({})
            stored_messages = gr.State([])

            minimalist_toggle = gr.Checkbox(label="Innie/Outie", value=False)

            consent_storage = gr.Checkbox(
                label="Store task and agent trace?", value=True
            )

            gr.Markdown(
                """
- **Data**: To opt-out of storing your trace, uncheck the box above.
- **Be patient**: The agent's first step can take a few seconds.
- **Captcha**: Sometimes the VMs get flagged for weird behaviour and are blocked with a captcha. Best is then to interrupt the agent and solve the captcha manually.
- **Restart**: If your agent seems stuck, the simplest way to restart is to refresh the page.
                """.strip()
            )

            # Hidden HTML element to inject CSS dynamically
            theme_styles = gr.HTML(apply_theme(False), visible=False)
            minimalist_toggle.change(
                fn=apply_theme, inputs=[minimalist_toggle], outputs=[theme_styles]
            )

            footer = gr.HTML(value=FOOTER_HTML, label="Footer")

    chatbot_display = gr.Chatbot(
        elem_id="chatbot",
        label="Agent's execution logs",
        type="messages",
        avatar_images=(
            None,
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
        ),
        resizable=True,
    )

    agent_ui = EnrichedGradioUI(
        CodeAgent(tools=[], model=None, name="ok", description="ok")
    )

    stop_btn = gr.Button("Stop the agent!", variant="huggingface")

    def read_log_content(log_file, tail=4):
        """Read the contents of a log file for a specific session"""
        if not log_file:
            return "Waiting for session..."

        if not os.path.exists(log_file):
            return "Waiting for machine from the future to boot..."

        try:
            with open(log_file, "r") as f:
                lines = f.readlines()
                return "".join(lines[-tail:] if len(lines) > tail else lines)
        except Exception as e:
            return f"Guru meditation: {str(e)}"

    # Function to set view-only mode
    def clear_and_set_view_only(task_input, session_uuid):
        return update_html(False, session_uuid)

    def set_interactive(session_uuid):
        return update_html(True, session_uuid)

    def reactivate_stop_btn():
        return gr.Button("Stop the agent!", variant="huggingface")

    is_interactive = gr.Checkbox(value=True, visible=False)

    # Chain the events
    run_event = (
        run_btn.click(
            fn=clear_and_set_view_only,
            inputs=[task_input, session_uuid_state],
            outputs=[sandbox_html],
        )
        .then(
            agent_ui.interact_with_agent,
            inputs=[
                task_input,
                stored_messages,
                session_state,
                session_uuid_state,
                consent_storage,
            ],
            outputs=[chatbot_display],
        )
        .then(fn=set_interactive, inputs=[session_uuid_state], outputs=[sandbox_html])
        .then(fn=reactivate_stop_btn, outputs=[stop_btn])
    )

    def interrupt_agent(session_state):
        if not session_state["agent"].interrupt_switch:
            session_state["agent"].interrupt()
            print("Stopping agent...")
            return gr.Button("Stopping agent... (could take time)", variant="secondary")
        else:
            return gr.Button("Stop the agent!", variant="huggingface")

    stop_btn.click(fn=interrupt_agent, inputs=[session_state], outputs=[stop_btn])

    demo.load(
        fn=lambda: True,  # dummy to trigger the load
        outputs=[is_interactive],
    ).then(
        fn=initialize_session,
        js="() => localStorage.getItem('gradio-session-uuid') || (() => { const id = self.crypto.randomUUID(); localStorage.setItem('gradio-session-uuid', id); return id })()",
        inputs=[is_interactive],
        outputs=[sandbox_html, session_uuid_state],
    )

    def upload_interaction_logs():
        data_dirs = []
        for interaction_id in list(INTERACTION_IDS.keys()):
            data_dir = os.path.join(TMP_DIR, interaction_id)
            if os.path.exists(data_dir):
                data_dirs.append(data_dir)
                INTERACTION_IDS.pop(interaction_id)

        upload_to_hf_and_remove(data_dirs)

    demo.unload(fn=upload_interaction_logs)

# Launch the app
if __name__ == "__main__":
    Timer(60, cleanup_sandboxes).start()  # Run every minute
    demo.launch()
