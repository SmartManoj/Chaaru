import gradio as gr
import os
import json
import shutil
import traceback
from e2b_desktop import Sandbox
from huggingface_hub import upload_folder, login
from smolagents.monitoring import LogLevel
from smolagents.gradio_ui import GradioUI, stream_to_gradio
from textwrap import dedent
import time
from threading import Timer


from e2bqwen import QwenVLAPIModel, E2BVisionAgent

E2B_API_KEY = os.getenv("E2B_API_KEY")
SANDBOXES = {}
SANDBOX_METADATA = {}
SANDBOX_TIMEOUT = 300
WIDTH = 1280
HEIGHT = 960
TMP_DIR = './tmp/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

hf_token = os.getenv("HUGGINGFACE_API_KEY")
login(token=hf_token)
model = QwenVLAPIModel(hf_token = hf_token)


custom_css = """
/* Your existing CSS */
.sandbox-outer-wrapper {
    display: flex;
    justify-content: center;
    width: 100%;
    padding: 20px 0;
    overflow: hidden;
}

.sandbox-container {
    position: relative;
    width: 1024px;
    height: 811px;
    flex-shrink: 0;
    overflow: hidden;
}

.sandbox-background {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.sandbox-iframe {
    position: absolute;
    top: 7%;
    left: 18%;
    width: 1288px;
    height: 968px;
    border: 4px solid #444444;
    transform-origin: 0 0;
    transform: scale(0.51);
}

/* Status indicator light */
.status-indicator {
    position: absolute;
    bottom: 28.6%;
    left: 20.1%;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    border: 2px solid black;
    z-index: 100;
}

.status-text {
    position: absolute;
    bottom: 28.4%;
    left: 22.5%;
    font-size: 16px;
    font-weight: bold;
    color: black;
    background-color: white;
    padding: 0px 2px;
    border-radius: 3px;
    border: 2px solid black;
    text-shadow: none;
    z-index: 100;
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
"""


html_template = """
    <h2 style="text-align: center">Personal Computer Assistant</h2>
    <div class="sandbox-outer-wrapper">
      <div class="sandbox-container">
          <img src="https://huggingface.co/datasets/mfarre/servedfiles/resolve/main/desktop2.png" class="sandbox-background" />
          <div class="status-text">{status_text}</div>
          <div class="status-indicator {status_class}"></div>
          <iframe id="sandbox-iframe"
              src="{stream_url}" 
              class="sandbox-iframe"
              style="display: block;"
              allowfullscreen>
          </iframe>
          <img id="bsod-image"
              src="https://huggingface.co/datasets/mfarre/servedfiles/resolve/main/blue_screen_of_death.gif"
              class="bsod-image"
              style="display: none; position: absolute; top: 7%; left: 18%; width: 657px; height: 494px; border: 4px solid #444444;"
          />
      </div>
    </div>
"""

custom_js = """
function() {
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

    repo_id = "open-agents/os-agent-logs"    
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
    
    # Store sandbox with metadata
    SANDBOXES[session_hash] = desktop
    SANDBOX_METADATA[session_hash] = {
        'created_at': current_time,
        'last_accessed': current_time
    }
    
    return desktop

def update_html(interactive_mode, request: gr.Request):
    session_hash = request.session_hash
    desktop = get_or_create_sandbox(session_hash)
    auth_key = desktop.stream.get_auth_key()
    
    # Add view_only parameter based on interactive_mode
    base_url = desktop.stream.get_url(auth_key=auth_key)
    stream_url = base_url if interactive_mode else f"{base_url}&view_only=true"
    
    # Set status indicator class and text
    status_class = "status-interactive" if interactive_mode else "status-view-only"
    status_text = "Interactive" if interactive_mode else "View Only"
    
    creation_time = SANDBOX_METADATA[session_hash]['created_at'] if session_hash in SANDBOX_METADATA else time.time()

    html_content = html_template.format(
        stream_url=stream_url,
        status_class=status_class,
        status_text=status_text,
    )

    # Add hidden field with creation time for JavaScript to use
    html_content += f'<div id="sandbox-creation-time" style="display:none;" data-time="{creation_time}" data-timeout="{SANDBOX_TIMEOUT}"></div>'
    
    return html_content

def generate_interaction_id(request):
    """Generate a unique ID combining session hash and timestamp"""
    return f"{request.session_hash}_{int(time.time())}"

def save_final_status(folder, status, details = None):
    a = open(os.path.join(folder,"status.json"),"w")
    a.write(json.dumps({"status":status,"details":details}))
    a.close()

def get_log_file_path(session_hash):
    """
    Creates a log file path based on the session hash.
    Makes sure the directory exists.
    """
    log_dir = os.path.join(TMP_DIR, session_hash)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    return os.path.join(log_dir, 'console.log')

def initialize_session(interactive_mode, request: gr.Request):
    session_hash = request.session_hash
    # Create session-specific log file
    log_path = get_log_file_path(session_hash)
    # Initialize log file if it doesn't exist
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write(f"Ready to go...\n")
    # Return HTML and session hash
    return update_html(interactive_mode, request), session_hash

# Function to read log content that gets the path from session hash
def update_terminal_from_session(session_hash):
    if not session_hash:
        return "Waiting for session..."
    
    log_path = get_log_file_path(session_hash)
    return read_log_content(log_path)


def create_agent(data_dir, desktop, log_file):
    return E2BVisionAgent(
        model=model,
        data_dir=data_dir,
        desktop=desktop,
        max_steps=200,
        verbosity_level=LogLevel.INFO,
        planning_interval=5,
        log_file = log_file
    )

class EnrichedGradioUI(GradioUI):
    def log_user_message(self, text_input):
        import gradio as gr

        return (
            text_input,
            gr.Button(interactive=False),
        )
    def interact_with_agent(self, task_input, messages, session_state, session_hash, request: gr.Request):
        import gradio as gr

        interaction_id = generate_interaction_id(request)
        desktop = get_or_create_sandbox(session_hash)
        
        # Create data directory for this session
        data_dir = os.path.join(TMP_DIR, interaction_id)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        log_file = get_log_file_path(session_hash)
        
        if "agent" not in session_state:
            session_state["agent"] = create_agent(data_dir=data_dir, desktop=desktop, log_file=log_file)
        
        # Construct the full task with instructions
        full_task = task_input + dedent(f"""
            The desktop has a resolution of {WIDTH}x{HEIGHT}, take it into account to decide clicking coordinates.
            When clicking an element, always make sure to click THE MIDDLE of that element! Else you risk to miss it.
    
            Always analyze the latest screenshot carefully before performing actions. Make sure to:
            1. Look at elements on the screen to determine what to click or interact with
            2. Use precise coordinates for mouse movements and clicks
            3. Wait for page loads or animations to complete using the wait() tool
            4. Sometimes you may have missed a click, so never assume that you're on the right page, always make sure that your previous action worked In the screenshot you can see if the mouse is out of the clickable area. Pay special attention to this.
    
            When you receive a task, break it down into step-by-step actions. On each step, look at the current screenshot to validate if previous steps worked and decide the next action.
            We can only execute one action at a time. On each step, answer only a python blob with the action to perform
        """)

        try:
            messages.append(gr.ChatMessage(role="user", content=full_task))
            yield messages

            for msg in stream_to_gradio(session_state["agent"], task=full_task, reset_agent_memory=False):
                messages.append(msg)
                yield messages

            yield messages
            save_final_status(data_dir, "completed", details = str(session_state["agent"].memory.get_succinct_steps()))
        except Exception as e:
            error_message=f"Error in interaction: {str(e)}"
            messages.append(gr.ChatMessage(role="assistant", content=error_message))
            yield messages
            save_final_status(data_dir, "failed", details = str(error_message))
            error_result = "Error running agent - Model inference endpoints not ready. Try again later." if 'Both endpoints failed' in error_message else "Error running agent"
            yield gr.ChatMessage(role="assistant", content=error_result)

        finally:
            upload_to_hf_and_remove(data_dir)


# Create a Gradio app with Blocks
with gr.Blocks(css=custom_css, js=custom_js) as demo:
    #Storing session hash in a state variable
    session_hash_state = gr.State(None)

    html_output = gr.HTML(
        value=html_template.format(
            stream_url="",
            status_class="status-interactive",
            status_text="Interactive"
        ),
        label="Output"
    )
    with gr.Row():
        task_input = gr.Textbox(
            value="Find picture of cute puppies",
            label="Enter your command",
        )

        gr.Examples(
            examples=[
                "Check the commuting time between Bern and Zurich",
                "Write 'Hello World' in a text editor",
                "Search a flight Paris - Berlin for tomorrow"
            ],
            inputs = task_input,
            label= "Example Tasks",
            examples_per_page=4
        )
    
    # with gr.Group(visible=True) as terminal_container:

        #terminal = gr.Textbox(
        #    value="Initializing...",
        #    label='Console',
        #    lines=5,
        #    max_lines=10,
        #    interactive=False
        #)

        
        # Hidden refresh button
    refresh_btn = gr.Button("Refresh", visible=False, elem_id="refresh-log-btn")

    session_state = gr.State({})
    stored_messages = gr.State([])

    with gr.Group(visible=False) as results_container:
        results_output = gr.Textbox(
            label="Results",
            interactive=False,
            elem_id="results-output"
        )

    update_btn = gr.Button("Let's go!")

    chatbot = gr.Chatbot(
        label="Agent's execution logs",
        type="messages",
        avatar_images=(
            None,
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/smolagents/mascot_smol.png",
        ),
        resizeable=True,
        scale=1,
    )
    from smolagents import CodeAgent
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
        # First clear the results, then set view-only mode
        return "", update_html(False, request), gr.update(visible=False)

    # Function to set interactive mode
    def set_interactive_mode(request: gr.Request):
        return update_html(True, request)
    

    # Function to check result and conditionally set interactive mode
    def check_and_set_interactive(result, request: gr.Request):
        if result and not result.startswith("Error running agent"):
            # Only set interactive mode if no error
            return update_html(True, request)
        else:
            # Return the current HTML to avoid changing the display
            # This will keep the BSOD visible
            return gr.update()

    # Chain the events
    # 1. Set view-only mode when button is clicked and reset visibility
    view_only_event = update_btn.click(
        fn=clear_and_set_view_only,
        inputs=[task_input], 
        outputs=[results_output, html_output, results_container]
    ).then(            
        agent_ui.log_user_message,
        [task_input],
        [stored_messages, task_input],
    ).then(agent_ui.interact_with_agent, [stored_messages, chatbot, session_state, session_hash_state], [chatbot]).then(
        lambda: (
            gr.Textbox(
                interactive=True, placeholder="Enter your prompt here and press Shift+Enter or the button"
            ),
            gr.Button(interactive=True),
        ),
        None,
        [task_input],
    ).then(
        fn=check_and_set_interactive,
        inputs=[results_output],
        outputs=html_output
    )
 
    demo.load(
        fn=initialize_session,
        inputs=[gr.Checkbox(value=True, visible=False)],
        outputs=[html_output, session_hash_state]
    )
    
    # Connect refresh button to update terminal



# Launch the app
if __name__ == "__main__":
    Timer(60, cleanup_sandboxes).start()  # Run every minute
    demo.launch()