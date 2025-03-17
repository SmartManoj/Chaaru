import gradio as gr
import os
from e2b_desktop import Sandbox

E2B_API_KEY = os.getenv("E2B_API_KEY")
SANDBOXES = {}
TMP_DIR = './tmp/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)

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
    width: 1032px;
    height: 776px;
    border: 4px solid #444444;
    transform-origin: 0 0;
    transform: scale(0.392);
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

def update_html(interactive_mode, request: gr.Request):
    if request.session_hash not in SANDBOXES: # str necessary to run locally when hash is None
        print("No sandbox found, creating new one", request.session_hash) 
        desktop = Sandbox(api_key=E2B_API_KEY, resolution=(1024, 768), dpi=96)
        desktop.stream.start(require_auth=True)
        SANDBOXES[request.session_hash] = desktop
    
    desktop = SANDBOXES[request.session_hash]
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

    # Text input for placeholder text
    placeholder_input = gr.Textbox(
        value="Find picture of cute puppies",
        label="Enter your command"
    )
    
    # Interactive mode checkbox
    interactive_mode = gr.Checkbox(
        value=True,
        label="Interactive Mode"
    )

    # Update button
    update_btn = gr.Button("Let's go!")
    
    # Connect the components
    update_btn.click(
        fn=update_html,
        inputs=[interactive_mode],
        outputs=[html_output]
    )
    
    # Also update when interactive mode changes
    interactive_mode.change(
        fn=update_html,
        inputs=[interactive_mode],
        outputs=[html_output]
    )

    demo.load(update_html, [interactive_mode], html_output)
# Launch the app
if __name__ == "__main__":
    demo.launch()