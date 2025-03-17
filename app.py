import gradio as gr
import os
import shutil
from e2b_desktop import Sandbox
from huggingface_hub import upload_folder
from smolagents.monitoring import LogLevel
from textwrap import dedent

from e2bqwen import QwenVLAPIModel, E2BVisionAgent

E2B_API_KEY = os.getenv("E2B_API_KEY")
SANDBOXES = {}
WIDTH = 1920
HEIGHT = 1440
TMP_DIR = './tmp/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

model = QwenVLAPIModel()

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
    width: 1928px;
    height: 1448px;
    border: 4px solid #444444;
    transform-origin: 0 0;
    transform: scale(0.207);
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
          <iframe 
              src="{stream_url}" 
              class="sandbox-iframe"
              allowfullscreen>
          </iframe>
      </div>
    </div>"""

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


def get_or_create_sandbox(session_hash):
    if session_hash not in SANDBOXES:
        print(f"Creating new sandbox for session {session_hash}")
        desktop = Sandbox(api_key=E2B_API_KEY, resolution=(WIDTH, HEIGHT), dpi=96)
        desktop.stream.start(require_auth=True)
        SANDBOXES[session_hash] = desktop
    return SANDBOXES[session_hash]

def update_html(interactive_mode, request: gr.Request):
    desktop = get_or_create_sandbox(request.session_hash)
    auth_key = desktop.stream.get_auth_key()
    
    # Add view_only parameter based on interactive_mode
    base_url = desktop.stream.get_url(auth_key=auth_key)
    stream_url = base_url if interactive_mode else f"{base_url}&view_only=true"
    
    # Set status indicator class and text
    status_class = "status-interactive" if interactive_mode else "status-view-only"
    status_text = "Interactive" if interactive_mode else "View Only"
    
    html_content = html_template.format(
        stream_url=stream_url,
        status_class=status_class,
        status_text=status_text
    )
    return html_content

def run_agent_task(task_input, interactive_mode, request: gr.Request):
    session_hash = request.session_hash
    desktop = get_or_create_sandbox(session_hash)
    
    # Create data directory for this session
    data_dir = os.path.join(TMP_DIR, session_hash)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    

    # Create the agent
    agent = E2BVisionAgent(
        model=model,
        data_dir=data_dir,
        desktop=desktop,
        max_steps=200,
        verbosity_level=LogLevel.DEBUG,
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
        
        interactive_mode = True        
        return f"Task completed: {result}", update_html(interactive_mode, request)
    except Exception as e:
        error_message = f"Error running agent: {str(e)}"
        print(error_message)
        interactive_mode = True
        return error_message, update_html(interactive_mode, request)
    finally:
        upload_to_hf_and_remove(data_dir)


# Create a Gradio app with Blocks
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""<h1 style="text-align: center">Personal Computer Assistant</h1>""")
    
    # HTML output with simulated image and iframe
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
    
    # Interactive mode checkbox
    interactive_mode = gr.Checkbox(
        value=False,
        label="Interactive Mode"
    )

    # Results output
    results_output = gr.Textbox(
        label="Results",
        interactive=False
    )

    # Update button
    update_btn = gr.Button("Let's go!")
    
    # Connect the components for displaying the sandbox
    interactive_mode.change(
        fn=update_html,
        inputs=[interactive_mode],
        outputs=[html_output]
    )
    
    # Connect the components for running the agent
    update_btn.click(
        fn=run_agent_task,
        inputs=[task_input, interactive_mode],
        outputs=[results_output]
    )

    # Load the sandbox on app start
    demo.load(update_html, [interactive_mode], html_output)

# Launch the app
if __name__ == "__main__":
    demo.launch()