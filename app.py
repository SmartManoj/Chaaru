import gradio as gr
import os
import json
import shutil
import traceback
from textwrap import dedent
import time
from threading import Timer
from huggingface_hub import upload_folder, login
from e2b_desktop import Sandbox

from smolagents import CodeAgent
from smolagents.monitoring import LogLevel
from smolagents.gradio_ui import GradioUI, stream_to_gradio
from model_replay import FakeModelReplayLog

from e2bqwen import QwenVLAPIModel, E2BVisionAgent

E2B_API_KEY = os.getenv("E2B_API_KEY")
SANDBOXES = {}
SANDBOX_METADATA = {}
SANDBOX_TIMEOUT = 600
WIDTH = 1280
HEIGHT = 960
TMP_DIR = './tmp/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

hf_token = os.getenv("HUGGINGFACE_API_KEY")
login(token=hf_token)

custom_css = """
.sandbox-container {
    position: relative;
    width: 910px;
    overflow: hidden;
    margin: auto;
}
.sandbox-container {
    height: 800px;
}
.sandbox-frame {
    display: none;
    position: absolute;
    top: 0;
    left: 0;
    width: 910px;
    height: 800px;
    pointer-events:none;
}

.sandbox-iframe, .bsod-image {
    position: absolute;
    width: <<WIDTH>>px;
    height: <<HEIGHT>>px;
    border: 4px solid #444444;
    transform-origin: 0 0;
}

/* Colored label for task textbox */
.primary-color-label label span {
    font-weight: bold;
    color: var(--color-accent);
}

/* Status indicator light */
.status-bar {
    display: flex;
    flex-direction: row;
    align-items: center;
    flex-align:center;
    z-index: 100;
}

.status-indicator {
    width: 15px;
    height: 15px;
    border-radius: 50%;
}

.status-text {
    font-size: 16px;
    font-weight: bold;
    padding-left: 8px;
    text-shadow: none;
}

.status-interactive {
    background-color: #2ecc71;
    animation: blink 2s infinite;  
}

.status-view-only {
    background-color: #e74c3c;
}

.status-error {
    background-color: #e74c3c;
    animation: blink-error 1s infinite;
}

@keyframes blink-error {
    0% { background-color: rgba(231, 76, 60, 1); }
    50% { background-color: rgba(231, 76, 60, 0.4); }
    100% { background-color: rgba(231, 76, 60, 1); }
}

@keyframes blink {
    0% { background-color: rgba(46, 204, 113, 1); }  /* Green at full opacity */
    50% { background-color: rgba(46, 204, 113, 0.4); }  /* Green at 40% opacity */
    100% { background-color: rgba(46, 204, 113, 1); }  /* Green at full opacity */
}

.logo-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    width: 100%;
    box-sizing: border-box;
    gap: 5px;

.logo-item {
    display: flex;
    align-items: center;
    padding: 0 30px;
    gap: 10px;
    text-decoration: none!important;
    color: #f59e0b;
    font-size:17px;
}
.logo-item:hover {
    color: #935f06!important;
}
""".replace("<<WIDTH>>", str(WIDTH+15)).replace("<<HEIGHT>>", str(HEIGHT+10))

footer_html="""
<h3 style="text-align: center; margin-top:50px;"><i>Powered by open source:</i></h2>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
<div class="logo-container">
    <a class="logo-item" href="https://github.com/huggingface/smolagents"><i class="fa fa-github"></i>smolagents</a>
    <a class="logo-item" href="https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct"><i class="fa fa-github"></i>Qwen2-VL-72B</a>
    <a class="logo-item" href="https://github.com/e2b-dev/desktop"><i class="fa fa-github"></i>E2B Desktop</a>
</div>
"""
sandbox_html_template = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Oxanium:wght@200..800&display=swap');
</style>
<h1 style="color:var(--color-accent);margin:0;">Computer Agent - Input your task and run your personal assistant!<h1>
<div class="sandbox-container" style="margin:0;">
    <div class="status-bar">
        <div class="status-indicator {status_class}"></div>
        <div class="status-text">{status_text}</div>
    </div>
    <iframe id="sandbox-iframe"
        src="{stream_url}" 
        class="sandbox-iframe"
        style="display: block;"
        allowfullscreen>
    </iframe>
    <img src="https://huggingface.co/datasets/mfarre/servedfiles/resolve/main/blue_screen_of_death.gif" class="bsod-image" style="display: none;"/>
    <img src="https://huggingface.co/datasets/m-ric/images/resolve/main/HUD_thom.png" class="sandbox-frame" />
</div>
""".replace("<<WIDTH>>", str(WIDTH+15)).replace("<<HEIGHT>>", str(HEIGHT+10))

custom_js = """function() {
    document.body.classList.add('dark');

    // Function to check if sandbox is timing out
    const checkSandboxTimeout = function() {
        const timeElement = document.getElementById('sandbox-creation-time');
        
        if (timeElement) {
            const creationTime = parseFloat(timeElement.getAttribute('data-time'));
            const timeoutValue = parseFloat(timeElement.getAttribute('data-timeout'));
            const currentTime = Math.floor(Date.now() / 1000); // Current time in seconds
            
            const elapsedTime = currentTime - creationTime;
            console.log("Sandbox running for: " + elapsedTime + " seconds of " + timeoutValue + " seconds");
            
            // If we've exceeded the timeout, show BSOD
            if (elapsedTime >= timeoutValue) {
                console.log("Sandbox timeout! Showing BSOD");
                showBSOD('Error');
                // Don't set another timeout, we're done checking
                return;
            }
        }
        
        // Continue checking every 5 seconds
        setTimeout(checkSandboxTimeout, 5000);
    };
    
    const showBSOD = function(statusText = 'Error') {
        console.log("Showing BSOD with status: " + statusText);
        const iframe = document.getElementById('sandbox-iframe');
        const bsod = document.getElementById('bsod-image');
        
        if (iframe && bsod) {
            iframe.style.display = 'none';
            bsod.style.display = 'block';

            // Update status indicator
            const statusIndicator = document.querySelector('.status-indicator');
            const statusTextElem = document.querySelector('.status-text');

            if (statusIndicator) {
                statusIndicator.className = 'status-indicator status-error';
            }
            
            if (statusTextElem) {
                statusTextElem.innerText = statusText;
            }
        }
    };

    const resetBSOD = function() {
        console.log("Resetting BSOD display");
        const iframe = document.getElementById('sandbox-iframe');
        const bsod = document.getElementById('bsod-image');
        
        if (iframe && bsod) {
            if (bsod.style.display === 'block') {
                // BSOD is currently showing, reset it
                iframe.style.display = 'block';
                bsod.style.display = 'none';
                console.log("BSOD reset complete");
                return true; // Indicates reset was performed
            }
        }
        return false; // No reset needed
    };
    
    // Function to monitor for error messages
    const monitorForErrors = function() {
        console.log("Error monitor started");
        const resultsInterval = setInterval(function() {
            const resultsElements = document.querySelectorAll('textarea, .output-text');
            for (let elem of resultsElements) {
                const content = elem.value || elem.innerText || '';
                if (content.includes('Error running agent')) {
                    console.log("Error detected!");
                    showBSOD('Error');
                    clearInterval(resultsInterval);
                    break;
                }
            }
        }, 1000);
    };
    
    
    // Start monitoring for timeouts immediately
    checkSandboxTimeout();
    
    // Start monitoring for errors
    setTimeout(monitorForErrors, 3000);
    
    // Also monitor for errors after button clicks
    document.addEventListener('click', function(e) {
        if (e.target.tagName === 'BUTTON') {
            if (e.target.innerText === "Let's go!") {
                resetBSOD();
            }
            setTimeout(monitorForErrors, 3000);
        }
    });

    // Set up an interval to click the refresh button every 5 seconds
    setInterval(function() {
        const btn = document.getElementById('refresh-log-btn');
        if (btn) btn.click();
    }, 5000);

    // Force dark mode
    const params = new URLSearchParams(window.location.search);
    if (!params.has('__theme')) {
        params.set('__theme', 'dark');
        window.location.search = params.toString();
    }
}
"""

def write_to_console_log(log_file_path, message):
    """
    Appends a message to the specified log file with a newline character.
    
    Parameters:
        log_file_path (str): Path to the log file
        message (str): Message to append to the log file
    """
    if log_file_path is None:
        return False
    try:
        # Open the file in append mode
        with open(log_file_path, 'a') as log_file:
            # Write the message followed by a newline
            log_file.write(f"{message}\n")
        return True
    except Exception as e:
        print(f"Error writing to log file: {str(e)}")
        return False
    
def upload_to_hf_and_remove(folder_path):

    repo_id = "smolagents/computer-agent-logs"    
    try:
        folder_name = os.path.basename(os.path.normpath(folder_path))
        
        # Upload the folder to Huggingface
        print(f"Uploading {folder_path} to {repo_id}/{folder_name}...")
        url = upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=folder_name,
            ignore_patterns=[".git/*", ".gitignore"]
        )
        
        # Remove the local folder after successful upload
        print(f"Upload complete. Removing local folder {folder_path}...")
        shutil.rmtree(folder_path)
        print("Local folder removed successfully.")
        
        return url
    
    except Exception as e:
        print(f"Error during upload or cleanup: {str(e)}")
        raise

def cleanup_sandboxes():
    """Remove sandboxes that haven't been accessed for more than 5 minutes"""
    current_time = time.time()
    sandboxes_to_remove = []
    
    for session_id, metadata in SANDBOX_METADATA.items():
        if current_time - metadata['last_accessed'] > SANDBOX_TIMEOUT:
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


def get_or_create_sandbox(session_hash):
    current_time = time.time()
    print("======")
    print(":=======")
    print("Session hash:", session_hash)
    print("Sandboxes:", SANDBOXES.keys())
    print("Session hash in SANDBOXES:", session_hash in SANDBOXES)
    print("Session hash in SANDBOX_METADATA:", session_hash in SANDBOX_METADATA)
    if session_hash in SANDBOX_METADATA:
        print("Session not timeout:", current_time - SANDBOX_METADATA[session_hash]['created_at'] < SANDBOX_TIMEOUT)

    # Check if sandbox exists and is still valid
    if (session_hash in SANDBOXES and 
        session_hash in SANDBOX_METADATA and
        current_time - SANDBOX_METADATA[session_hash]['created_at'] < SANDBOX_TIMEOUT):
        
        # Update last accessed time
        SANDBOX_METADATA[session_hash]['last_accessed'] = current_time
        return SANDBOXES[session_hash]
    
    # Close existing sandbox if it exists but is too old
    if session_hash in SANDBOXES:
        try:
            print(f"Closing expired sandbox for session {session_hash}")
            SANDBOXES[session_hash].kill()
        except Exception as e:
            print(f"Error closing expired sandbox: {str(e)}")
    
    # Create new sandbox
    print(f"Creating new sandbox for session {session_hash}")
    desktop = Sandbox(api_key=E2B_API_KEY, resolution=(WIDTH, HEIGHT), dpi=96, timeout=SANDBOX_TIMEOUT)
    desktop.stream.start(require_auth=True)
    setup_cmd = """sudo mkdir -p /usr/lib/firefox-esr/distribution && echo '{"policies":{"OverrideFirstRunPage":"","OverridePostUpdatePage":"","DisableProfileImport":true,"DontCheckDefaultBrowser":true}}' | sudo tee /usr/lib/firefox-esr/distribution/policies.json > /dev/null"""
    desktop.commands.run(setup_cmd)
    
    # Store sandbox with metadata
    SANDBOXES[session_hash] = desktop
    SANDBOX_METADATA[session_hash] = {
        'created_at': current_time,
        'last_accessed': current_time
    }
    return desktop

def update_html(interactive_mode: bool, request: gr.Request):
    session_hash = request.session_hash
    desktop = get_or_create_sandbox(session_hash)
    auth_key = desktop.stream.get_auth_key()
    
    # Add view_only parameter based on interactive_mode
    base_url = desktop.stream.get_url(auth_key=auth_key)
    stream_url = base_url if interactive_mode else f"{base_url}&view_only=true"
    
    # Set status indicator class and text
    status_class = "status-interactive" if interactive_mode else "status-view-only"
    status_text = "Interactive" if interactive_mode else "Agent running..."
    
    creation_time = SANDBOX_METADATA[session_hash]['created_at'] if session_hash in SANDBOX_METADATA else time.time()

    sandbox_html_content = sandbox_html_template.format(
        stream_url=stream_url,
        status_class=status_class,
        status_text=status_text,
    )

    # Add hidden field with creation time for JavaScript to use
    sandbox_html_content += f'<div id="sandbox-creation-time" style="display:none;" data-time="{creation_time}" data-timeout="{SANDBOX_TIMEOUT}"></div>'

    return sandbox_html_content

def generate_interaction_id(request):
    """Generate a unique ID combining session hash and timestamp"""
    return f"{request.session_hash}_{int(time.time())}"


def chat_message_to_json(obj):
    """Custom JSON serializer for ChatMessage and related objects"""
    if hasattr(obj, '__dict__'):
        # Create a copy of the object's __dict__ to avoid modifying the original
        result = obj.__dict__.copy()
        
        # Remove the 'raw' field which may contain non-serializable data
        if 'raw' in result:
            del result['raw']
            
        # Process the content or tool_calls if they exist
        if 'content' in result and result['content'] is not None:
            if hasattr(result['content'], '__dict__'):
                result['content'] = chat_message_to_json(result['content'])
        
        if 'tool_calls' in result and result['tool_calls'] is not None:
            result['tool_calls'] = [chat_message_to_json(tc) for tc in result['tool_calls']]
            
        return result
    elif isinstance(obj, (list, tuple)):
        return [chat_message_to_json(item) for item in obj]
    else:
        return obj


def save_final_status(folder, status: str, summary, error_message = None) -> None:
    metadata_path = os.path.join(folder, "metadata.json")
    output_file = open(metadata_path,"w")
    output_file.write(json.dumps({"status":status, "summary":summary, "error_message": error_message}, default=chat_message_to_json))
    output_file.close()

def initialize_session(interactive_mode, request: gr.Request):
    session_hash = request.session_hash
    # Return HTML and session hash
    return update_html(interactive_mode, request), session_hash


def create_agent(data_dir, desktop):
    model = QwenVLAPIModel(
        model_id="Qwen/Qwen2.5-VL-72B-Instruct",
        hf_token = hf_token,
    )
    return E2BVisionAgent(
        model=model,
        data_dir=data_dir,
        desktop=desktop,
        max_steps=200,
        verbosity_level=2,
        planning_interval=10,
    )


class EnrichedGradioUI(GradioUI):
    def log_user_message(self, text_input):
        import gradio as gr

        return (
            text_input,
            gr.Button(interactive=False),
        )

    def interact_with_agent(self, task_input, stored_messages, session_state, session_hash, request: gr.Request):
        import gradio as gr

        interaction_id = generate_interaction_id(request)
        desktop = get_or_create_sandbox(session_hash)
        
        # Create data directory for this session
        data_dir = os.path.join(TMP_DIR, interaction_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        if "agent" in session_state:
            session_state["agent"].data_dir = data_dir # Update data dir to new interaction
        else:
            session_state["agent"] = create_agent(data_dir=data_dir, desktop=desktop)

        if "replay_log" in session_state and session_state["replay_log"] is not None:
            original_model = session_state["agent"].model
            session_state["agent"].model = FakeModelReplayLog(session_state["replay_log"])

        
        try:
            stored_messages.append(gr.ChatMessage(role="user", content=task_input))
            yield stored_messages

            for msg in stream_to_gradio(session_state["agent"], task=task_input, reset_agent_memory=False):
                if hasattr(session_state["agent"], "last_screenshot") and msg.content == "-----": # Append the last screenshot before the end of step
                    stored_messages.append(gr.ChatMessage(
                        role="assistant",
                        content={"path": session_state["agent"].last_screenshot.to_string(), "mime_type": "image/png"},
                    ))
                stored_messages.append(msg)
                yield stored_messages

            yield stored_messages
            # THIS ERASES IMAGES FROM MEMORY, USE WITH CAUTION
            memory = session_state["agent"].memory
            for memory_step in memory.steps:
                if getattr(memory_step, "observations_images", None):
                    memory_step.observations_images = None
            summary = memory.get_succinct_steps()
            save_final_status(data_dir, "completed", summary = summary)
    
        # # TODO: uncomment below after testing
        except Exception as e:
            error_message=f"Error in interaction: {str(e)}"
            stored_messages.append(gr.ChatMessage(role="assistant", content=error_message))
            yield stored_messages
            raise e
            save_final_status(data_dir, "failed", summary=[], error_message=error_message)

        finally:
            if "replay_log" in session_state and session_state["replay_log"] is not None: # Replace the model with original model
                session_state["agent"].model = original_model
                session_state["replay_log"] = None
            upload_to_hf_and_remove(data_dir)

theme = gr.themes.Default(font=["Oxanium", "sans-serif"], primary_hue="amber", secondary_hue="blue")

# Create a Gradio app with Blocks
with gr.Blocks(theme=theme, css=custom_css, js=custom_js) as demo:
    #Storing session hash in a state variable
    session_hash_state = gr.State(None)

    with gr.Row():
        sandbox_html = gr.HTML(
            value=sandbox_html_template.format(
                stream_url="",
                status_class="status-interactive",
                status_text="Interactive"
            ),
            label="Output"
        )
        with gr.Sidebar(position="left"):
            task_input = gr.Textbox(
                value="Download a picture of a cute puppy",
                label="Enter your task below:",
                elem_classes="primary-color-label"
            )

            run_btn = gr.Button("Let's go!", variant="primary")

            gr.Examples(
                examples=[
                    "Check the commuting time between Bern and Zurich on Google maps",
                    "Write 'Hello World' in a text editor",
                    "Search a flight Paris - Berlin for tomorrow",
                    "Search for Château de Fontainebleau in Google Maps",
                    "What is the picture that appeared on the very first version (2004) of the english Wikipedia page for the Palace of Fontainebleau?",
                    "Download me a picture of a puppy from Google, then head to Hugging Face, find a Space dedicated to background removal, and use it to remove the puppy picture's background"
                ],
                inputs = task_input,
                label= "Example Tasks",
                examples_per_page=4
            )

            session_state = gr.State({})
            stored_messages = gr.State([])


            replay_btn = gr.Button("Replay an agent run")

            minimalist_toggle = gr.Checkbox(label="Innie/Outie", value=False)

            def apply_theme(minimalist_mode: bool):
                if not minimalist_mode:
                    return """
                        <style>
                        .sandbox-frame {
                            display: block!important;
                        }

                        .sandbox-iframe, .bsod-image {
                            /* top: 73px; */
                            top: 99px;
                            /* left: 74px; */
                            left: 110px;
                        }
                        .sandbox-iframe {
                            transform: scale(0.535);
                            /* transform: scale(0.59); */
                        }

                        .status-bar {
                            position: absolute;
                            bottom: 88px;
                            left: 355px;
                        }
                        .status-text {
                            color: #fed244;
                        }
                        </style>
                    """
                else:
                    return """
                        <style>
                        .sandbox-container {
                            height: 700px!important;
                        }
                        .sandbox-iframe {
                            transform: scale(0.65);
                        }
                        </style>
                    """

            # Hidden HTML element to inject CSS dynamically
            theme_styles = gr.HTML(apply_theme(False), visible=False)
            minimalist_toggle.change(
                fn=apply_theme,
                inputs=[minimalist_toggle],
                outputs=[theme_styles]
            )

            footer = gr.HTML(
                value=footer_html,
                label="Header"
            )

    stop_btn = gr.Button("Stop the agent!")

    chatbot_display = gr.Chatbot(
        label="Agent's execution logs",
        type="messages",
        avatar_images=(
            None,
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
        ),
        resizable=True,
    )

    agent_ui = EnrichedGradioUI(CodeAgent(tools=[], model=None, name="ok", description="ok"))

    def read_log_content(log_file, tail=4):
        """Read the contents of a log file for a specific session"""
        if not log_file:
            return "Waiting for session..."

        if not os.path.exists(log_file):
            return "Waiting for machine from the future to boot..."
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                return "".join(lines[-tail:] if len(lines) > tail else lines)
        except Exception as e:
            return f"Guru meditation: {str(e)}"

    # Function to set view-only mode
    def clear_and_set_view_only(task_input, request: gr.Request):
        # set view-only mode
        return update_html(False, request)

    def set_interactive(request: gr.Request):
        return update_html(True, request)

    is_interactive = gr.Checkbox(value=True, visible=False)

    # Chain the events
    run_event = run_btn.click(
        fn=clear_and_set_view_only,
        inputs=[task_input], 
        outputs=[sandbox_html]
    ).then(
        agent_ui.interact_with_agent,
        inputs=[task_input, stored_messages, session_state, session_hash_state],
        outputs=[chatbot_display]
    ).then(
        fn=set_interactive,
        inputs=[],
        outputs=[sandbox_html]
    )

    stop_btn.click(fn=None, inputs=None, outputs=None, cancels=[run_event])

    def set_logs_source(session_state):
        session_state["replay_log"] = "udupp2fyavq_1743170323"

    replay_btn.click(
        fn=clear_and_set_view_only,
        inputs=[task_input], 
        outputs=[sandbox_html]
    ).then(
        set_logs_source,
        inputs=[session_state]
    ).then(
        agent_ui.interact_with_agent,
        inputs=[task_input, stored_messages, session_state, session_hash_state],
        outputs=[chatbot_display]
    ).then(
        fn=set_interactive,
        inputs=[],
        outputs=[sandbox_html]
    )

    demo.load(
        fn=initialize_session,
        inputs=[is_interactive],
        outputs=[sandbox_html, session_hash_state],
    )

# Launch the app
if __name__ == "__main__":
    Timer(60, cleanup_sandboxes).start()  # Run every minute
    demo.launch()