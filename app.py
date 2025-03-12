import gradio as gr
import os
from e2b_desktop import Sandbox


E2B_API_KEY = os.getenv("E2B_API_KEY")
DEFAULT_MAX_TOKENS = 512
SANDBOXES = {}
TMP_DIR = './tmp/'
if not os.path.exists(TMP_DIR):
    os.makedirs(TMP_DIR)


custom_css = """
/* Outer wrapper to handle centering without scrolling */
.sandbox-outer-wrapper {
    display: flex;
    justify-content: center;
    width: 100%;
    padding: 20px 0;
    overflow: hidden; /* Changed from overflow-x: auto */
}

.sandbox-container {
    position: relative;
    width: 800px;
    height: 500px;
    flex-shrink: 0; /* Prevents container from shrinking */
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
"""


def update_placeholder_text():
    desktop = Sandbox(api_key=E2B_API_KEY, resolution=(1024, 768), dpi=96)
    desktop.stream.start(require_auth=True)
    auth_key = desktop.stream.get_auth_key()
    stream_url = desktop.stream.get_url(auth_key=auth_key)
    
    html_content = f"""
    <div class="sandbox-outer-wrapper">
      <div class="sandbox-container">
          <img src="https://huggingface.co/datasets/lvwerra/admin/resolve/main/desktop_scaled.png" class="sandbox-background" />
          <iframe 
              src="{stream_url}" 
              class="sandbox-iframe"
              allowfullscreen>
          </iframe>
      </div>
    </div>
    """
    return html_content

# Create a Gradio app with Blocks
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""<h1 style="text-align: center">Personal Computer Assistant</h1>""")
    
    # HTML output with simulated image and iframe
    html_output = gr.HTML(
        value=update_placeholder_text(),
        label="Output"
    )

    # Text input for placeholder text
    placeholder_input = gr.Textbox(
        value="Find picture of cute puppies",
        label="Enter your command"
    )
    
    # Update button
    update_btn = gr.Button("Let's go!")
    
    # Connect the components
    update_btn.click(
        fn=update_placeholder_text,
        inputs=None,
        outputs=[html_output]
    )
    

# Launch the app
if __name__ == "__main__":
    demo.launch()