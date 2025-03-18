import gradio as gr
import os
import json
import shutil
import traceback
from e2b_desktop import Sandbox
from huggingface_hub import upload_folder, login
from smolagents.monitoring import LogLevel
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

model = QwenVLAPIModel()
login(token=os.getenv("HUGGINGFACE_API_KEY"))

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
    width: 800px;
    height: 500px;
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
    top: 10%;
    left: 25%;
    width: 1288px;
    height: 968px;
    border: 4px solid #444444;
    transform-origin: 0 0;
    transform: scale(0.312);
}

/* Status indicator light */
.status-indicator {
    position: absolute;
    bottom: 25.5%;
    left: 26.7%;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    border: 2px solid black;
    z-index: 100;
}

.status-text {
    position: absolute;
    bottom: 25.0%;
    left: 28.6%;
    font-size: 12px;
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
    <div class="sandbox-outer-wrapper">
      <div class="sandbox-container">
          <img src="https://huggingface.co/datasets/lvwerra/admin/resolve/main/desktop_scaled.png" class="sandbox-background" />
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
              style="display: none; position: absolute; top: 10%; left: 25%; width: 400px; height: 300px; border: 4px solid #444444;"
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
            setTimeout(monitorForErrors, 3000);
        }
    });
}
"""

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
    a.write(json.dumps({"status":status,"details":str(details)}))
    a.close()

def run_agent_task(task_input, request: gr.Request):
    session_hash = request.session_hash
    interaction_id = generate_interaction_id(request)
    desktop = get_or_create_sandbox(session_hash)
    
    # Create data directory for this session
    data_dir = os.path.join(TMP_DIR, interaction_id)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    

    # Create the agent
    agent = E2BVisionAgent(
        model=model,
        data_dir=data_dir,
        desktop=desktop,
        max_steps=200,
        verbosity_level=LogLevel.INFO,
        planning_interval=5,
    )
    
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

        # Run the agent
        result = agent.run(full_task)
        save_final_status(data_dir, "completed", details = result)        
        return f"Task completed: {result}"

    except Exception as e:
        error_message = f"Error running agent: {str(e)} Details {traceback.format_exc()}"
        save_final_status(data_dir, "failed", details = error_message)
        print(error_message)
        return "Error running agent"
    
    finally:
        upload_to_hf_and_remove(data_dir)

# Create a Gradio app with Blocks
with gr.Blocks(css=custom_css, js=custom_js) as demo:
    gr.HTML("""<h1 style="text-align: center">Personal Computer Assistant</h1>""")
    
    # HTML output with simulated image and iframe - default to interactive
    html_output = gr.HTML(
        value=html_template.format(
            stream_url="",
            status_class="status-interactive",
            status_text="Interactive"
        ),
        label="Output"
    )

    # Text input for task
    task_input = gr.Textbox(
        value="Find picture of cute puppies",
        label="Enter your command"
    )
    
    # Results output
    results_output = gr.Textbox(
        label="Results",
        interactive=False,
        elem_id="results-output"
    )

    # Update button
    update_btn = gr.Button("Let's go!")
    
    # Function to set view-only mode
    def set_view_only(task_input, request: gr.Request):
        return update_html(False, request)
    
    # Function to set interactive mode
    def set_interactive_mode(request: gr.Request):
        return update_html(True, request)
    
    # Chain the events
    # 1. Set view-only mode when button is clicked
    view_only_event = update_btn.click(
        fn=set_view_only,
        inputs=[task_input], 
        outputs=html_output
    )
    
    # 2. Then run the agent task
    task_result = view_only_event.then(
        fn=run_agent_task,
        inputs=[task_input],
        outputs=results_output
    )
    
    # 3. Then set back to interactive mode
    task_result.then(
        fn=set_interactive_mode,
        inputs=None,  # No inputs needed here
        outputs=html_output
    )
    
    # Load the sandbox on app start with initial HTML
    demo.load(
        fn=update_html,
        inputs=[gr.Checkbox(value=True, visible=False)],  # Hidden checkbox with True value
        outputs=html_output
    )

# Launch the app
if __name__ == "__main__":
    Timer(60, cleanup_sandboxes).start()  # Run every minute
    demo.launch()