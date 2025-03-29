import os
import time
import base64
from io import BytesIO
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple
import json
import unicodedata

# HF API params
from huggingface_hub import InferenceClient

# E2B imports
from e2b_desktop import Sandbox
from PIL import Image

# SmolaAgents imports
from smolagents import CodeAgent, tool, HfApiModel
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, MessageRole, Model
from smolagents.monitoring import LogLevel
from smolagents.agent_types import AgentImage
from PIL import ImageDraw

E2B_SYSTEM_PROMPT_TEMPLATE = """You are a desktop automation assistant that can control a remote desktop environment.
<action process>
You willbe given a task to solve in several steps. At each step you will perform an action.
After each action, you'll receive an updated screenshot. 
Then you will proceed as follows, with these sections: don't skip any!

Short term goal: ...
Where I am: ...
What I see: ...
Reflection: ...
Action: ...
Code:
```python
click(250, 300)
```<end_code>
</action_process>

<tools>
On top of performing computations in the Python code snippets that you create, you only have access to these tools to interact with the desktop, no additional ones:
{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
{%- endfor %}

The desktop has a resolution of <<resolution_x>>x<<resolution_y>>, take it into account to decide clicking coordinates.
If you clicked somewhere in the previous action, a red crosshair will appear at the exact location of the previous click.
The image might have change since then but the cross stays at the previous click. If your click seems to have changed nothing, check that this location is exactly where you intended to click. Otherwise correct the click coordinates.
</tools>

<code_format>
Always format your actions as Python code blocks, as shown below:
Code:
```python
click(250, 300)
```<end_code>
</code_format>

<task_resolution_example>
For a task like "Open a text editor and type 'Hello World'":
Step 1:
Short term goal: I want to open a text editor.
Where I am: I am on the homepage of my desktop.
What I see: I see the applications
Reflection: I think that a notes application would fit in the Applications menu, let's open it. 
Action: I'll click it, carefully clicking in the middle of the text 'Applications'/
Code:
```python
click(50, 10) 
```<end_code>

Step 2:
Short term goal: I want to open a text editor.
Where I am: I am on the homepage of my desktop, with the applications menu open.
What I see: I see an Accessories section, I see it is a section in the menu thanks to the tiny white triangle after the text accessories.
Reflection: I think that a notes application would fit the Accessories section. I SHOULD NOT try to move through the menus with scroll, it won't work:.
Action: I'll look for Accessories and click on it being very precise, clicking in the middle of the text 'Accessories'.
Code:
```python
click(76, 195) 
```<end_code>

Step 3:
Short term goal: I want to open a text editor.
Where I am: I am under the Accessories menu.
What I see: under the open submenu Accessories, I've found 'Text Editor'.
Reflection: This must be my notes app. I remember that menus are navigated through clicking.
Action: I will now click on it being very precise, clicking in the middle of the text 'Text Editor'.
Code:
```python
click(251, 441) 
```<end_code>

Step 4:
Short term goal: I want to open a text editor.
Where I am: I am still under the Accessories menu.
What I see: Nothing has changed compared to previous screenshot. Under the open submenu Accessories, I still see 'Text Editor'. The red crosshair is off from the element.
Reflection: My last click must have been off. Let's correct this.
Action: I will click the correct place, right in the middle of the element.
Code:
```python
click(241, 441) 
```<end_code>

Step 5:
Short term goal: I want to type 'Hello World'.
Where I am: I have opened a Notepad.
What I see: The Notepad app is open on an empty page
Reflection: Now Notepad is open as intended, time to type text.
Action: I will type the requested text.
Code:
```python
type_text("Hello World")
```<end_code>

Step 6:
Short term goal: I want to type 'Hello World'.
Where I am: I have opened a Notepad.
What I see: The Notepad app displays 'Hello World'
Reflection: Now that I've 1. Opened the notepad and 2. typed 'Hello World', and 3. the result seems correct, I think the Task is completed.
Action: I will return a confirmation that the task is completed.
Code:
```python
final_answer("Done")
```<end_code>
</task_resolution_example>

<click_guidelines>
Look at elements on the screen to determine what to click or interact with.
Use precise coordinates for mouse movements and clicks. When clicking an element, ALWAYS CLICK THE MIDDLE of that element, not UNDER OR ABOVE! Else you risk to miss it.
Sometimes you may have missed a click, so never assume that you're on the right page, always make sure that your previous action worked. In the screenshot you can see if the mouse is out of the clickable area. Pay special attention to this.
Remember the tools that you have as those can save you time, for example open_url to enter a website rather than searching for the browser in the OS.
Whenever you click, MAKE SURE to click in the middle of the button, text, link or any other clickable element. Not under, not on the side. IN THE MIDDLE. In menus it is always better to click in the middle of the text rather than in the tiny icon. Calculate extremelly well the coordinates. A mistake here can make the full task fail.
</click_guidelines>

<general_guidelines>
You can wait for appropriate loading times using the wait() tool. But don't wait forever, sometimes you've just misclicked and the process didn't launch.
Use precise coordinates based on the current screenshot
Execute one action at a time: don't try to pack a click and typing in one action.
On each step, look at the last screenshot and action to validate if previous steps worked and decide the next action. If you repeated an action already without effect, it means that this action is useless: don't repeat it and try something else.
Use click to move through menus on the desktop and scroll for web and specific applications.
Always analyze the latest screenshot carefully before performing actions. Make sure to:
To navigate the desktop you should open menus and click. Menus usually expand with more options, the tiny triangle next to some text in a menu means that menu expands. For example in Office in the Applications menu expands showing presentation or writing applications. 
Always analyze the latest screenshot carefully before performing actions.
</general_guidelines>
"""

def draw_marker_on_image(image, click_coordinates):
    x, y = click_coordinates
    draw = ImageDraw.Draw(image)
    cross_size, linewidth = 10, 3
    # Draw red cross lines
    draw.line((x - cross_size, y, x + cross_size, y), fill="red", width=linewidth)
    draw.line((x, y - cross_size, x, y + cross_size), fill="red", width=linewidth)
    # Add a circle around it for better visibility
    draw.ellipse((x - cross_size * 2, y - cross_size * 2, x + cross_size * 2, y + cross_size * 2), outline="red", width=linewidth)

class E2BVisionAgent(CodeAgent):
    """Agent for e2b desktop automation with Qwen2.5VL vision capabilities"""
    def __init__(
        self,
        model: HfApiModel,
        data_dir: str,
        desktop: Sandbox,
        tools: List[tool] = None,
        max_steps: int = 200,
        verbosity_level: LogLevel = 2,
        planning_interval: int = 10,
        **kwargs
    ):
        self.desktop = desktop
        self.data_dir = data_dir
        self.planning_interval = planning_interval
        # Initialize Desktop
        self.width, self.height = self.desktop.get_screen_size()
        print(f"Screen size: {self.width}x{self.height}")

        # Set up temp directory
        os.makedirs(self.data_dir, exist_ok=True)
        print(f"Screenshots and steps will be saved to: {self.data_dir}")

        # Initialize base agent
        super().__init__(
            tools=tools or [],
            model=model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            planning_interval = self.planning_interval,
            **kwargs
        )
        self.prompt_templates["system_prompt"] = E2B_SYSTEM_PROMPT_TEMPLATE.replace("<<resolution_x>>", str(self.width)).replace("<<resolution_y>>", str(self.height))


        # Add screen info to state
        self.state["screen_width"] = self.width
        self.state["screen_height"] = self.height


        # Add default tools
        self.logger.log("Setting up agent tools...")
        self._setup_desktop_tools()
        self.step_callbacks.append(self.take_screenshot_callback)

    def _setup_desktop_tools(self):
        """Register all desktop tools"""
        @tool
        def click(x: int, y: int) -> str:
            """
            Performs a left-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            self.desktop.move_mouse(x, y)
            self.desktop.left_click()
            self.click_coordinates = [x, y]
            self.logger.log(f"Clicked at coordinates ({x}, {y})")
            return f"Clicked at coordinates ({x}, {y})"

        @tool
        def right_click(x: int, y: int) -> str:
            """
            Performs a right-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            self.desktop.move_mouse(x, y)
            self.desktop.right_click()
            self.click_coordinates = [x, y]
            self.logger.log(f"Right-clicked at coordinates ({x}, {y})")
            return f"Right-clicked at coordinates ({x}, {y})"

        @tool
        def double_click(x: int, y: int) -> str:
            """
            Performs a double-click at the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            self.desktop.move_mouse(x, y)
            self.desktop.double_click()
            self.click_coordinates = [x, y]
            self.logger.log(f"Double-clicked at coordinates ({x}, {y})")
            return f"Double-clicked at coordinates ({x}, {y})"

        @tool
        def move_mouse(x: int, y: int) -> str:
            """
            Moves the mouse cursor to the specified coordinates
            Args:
                x: The x coordinate (horizontal position)
                y: The y coordinate (vertical position)
            """
            self.desktop.move_mouse(x, y)
            self.logger.log(f"Moved mouse to coordinates ({x}, {y})")
            return f"Moved mouse to coordinates ({x}, {y})"

        def normalize_text(text):
            return ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))

        @tool
        def type_text(text: str, delay_in_ms: int = 75) -> str:
            """
            Types the specified text at the current cursor position.
            Args:
                text: The text to type
                delay_in_ms: Delay between keystrokes in milliseconds
            """
            clean_text = normalize_text(text)
            self.desktop.write(clean_text, delay_in_ms=delay_in_ms)
            self.logger.log(f"Typed text: '{clean_text}'")
            return f"Typed text: '{clean_text}'"

        @tool
        def press_key(key: str) -> str:
            """
            Presses a keyboard key
            Args:
                key: The key to press (e.g. "enter", "space", "backspace", etc.).
            """
            self.desktop.press(key)
            self.logger.log(f"Pressed key: {key}")
            return f"Pressed key: {key}"

        @tool
        def go_back() -> str:
            """
            Goes back to the previous page in the browser.
            Args:
            """
            self.desktop.press(["alt", "left"])
            self.logger.log("Went back one page")
            return "Went back one page"

        @tool
        def drag_and_drop(x1: int, y1: int, x2: int, y2: int) -> str:
            """
            Clicks [x1, y1], drags mouse to [x2, y2], then release click.
            Args:
                x1: origin x coordinate
                y1: origin y coordinate
                x2: end x coordinate
                y2: end y coordinate
            """
            self.desktop.drag([x1, y1], [x2, y2])
            message = f"Dragged and dropped from [{x1}, {y1}] to [{x2}, {y2}]"
            self.logger.log(message)
            return message

        @tool
        def scroll(x: int, y: int, direction: str = "down", amount: int = 1) -> str:
            """
            Uses scroll button: this could scroll the page or zoom, depending on the app. DO NOT use scroll to move through linux desktop menus.
            Args:
                x: The x coordinate (horizontal position) of the element to scroll/zoom
                y: The y coordinate (vertical position) of the element to scroll/zoom
                direction: The direction to scroll ("up" or "down"), defaults to "down"
                amount: The amount to scroll. A good amount is 1 or 2.
            """
            self.desktop.scroll(direction=direction, amount=amount)
            self.logger.log(f"Scrolled {direction} by {amount}")
            return f"Scrolled {direction} by {amount}"

        @tool
        def wait(seconds: float) -> str:
            """
            Waits for the specified number of seconds. Very useful in case the prior order is still executing (for example starting very heavy applications like browsers or office apps)
            Args:
                seconds: Number of seconds to wait, generally 3 is enough.
            """
            time.sleep(seconds)
            self.logger.log(f"Waited for {seconds} seconds")
            return f"Waited for {seconds} seconds"

        @tool
        def open_url(url: str) -> str:
            """
            Directly opens a browser with the specified url, saves time compared to clicking in a browser and going through the initial setup wizard.
            Args:
                url: The URL to open
            """
            # Make sure URL has http/https prefix
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            self.desktop.open(url)
            # Give it time to load
            time.sleep(2)
            self.logger.log(f"Opening URL: {url}")
            return f"Opened URL: {url}"


        # Register the tools
        self.tools["click"] = click
        self.tools["right_click"] = right_click
        self.tools["double_click"] = double_click
        self.tools["move_mouse"] = move_mouse
        self.tools["type_text"] = type_text
        self.tools["press_key"] = press_key
        self.tools["scroll"] = scroll
        self.tools["wait"] = wait
        self.tools["open_url"] = open_url
        self.tools["go_back"] = go_back
        self.tools["drag_and_drop"] = drag_and_drop


    def take_screenshot_callback(self, memory_step: ActionStep, agent=None) -> None:
        """Callback that takes a screenshot + memory snapshot after a step completes"""
        self.logger.log("Analyzing screen content...")

        current_step = memory_step.step_number

        time.sleep(2.0)  # Let things happen on the desktop
        screenshot_bytes = self.desktop.screenshot()
        image = Image.open(BytesIO(screenshot_bytes))

        if getattr(self, "click_coordinates", None):
            # If a click was performed in the last action, mark it on the image
            x, y = self.click_coordinates
            draw = ImageDraw.Draw(image)
            cross_size, linewidth = 10, 3
            # Draw red cross lines
            draw.line((x - cross_size, y, x + cross_size, y), fill="red", width=linewidth)
            draw.line((x, y - cross_size, x, y + cross_size), fill="red", width=linewidth)
            # Add a circle around it for better visibility
            draw.ellipse((x - cross_size * 2, y - cross_size * 2, x + cross_size * 2, y + cross_size * 2), outline="red", width=linewidth)

        # Create a filename with step number
        screenshot_path = os.path.join(self.data_dir, f"step_{current_step:03d}.png")
        image.save(screenshot_path)
        self.last_screenshot = AgentImage(screenshot_path)
        print(f"Saved screenshot for step {current_step} to {screenshot_path}")

        for (
            previous_memory_step
        ) in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number <= current_step - 2
            ):
                previous_memory_step.observations_images = None

            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number <= current_step - 1
            ):
                if previous_memory_step.tool_calls[0].arguments == memory_step.tool_calls[0].arguments:
                    memory_step.observations += "\nWARNING: You've executed the same action several times in a row. MAKE SURE TO NOT USELESSLY REPEAT ACTIONS."

        # Add to the current memory step
        memory_step.observations_images = [image.copy()]

        # memory_step.observations_images = [screenshot_path] # IF YOU USE THIS INSTEAD OF ABOVE, LAUNCHING A SECOND TASK BREAKS

        self.click_coordinates = None # Reset click marker


    def close(self):
        """Clean up resources"""
        if self.desktop:
            print("Stopping e2b stream and killing sandbox...")
            self.desktop.stream.stop()
            self.desktop.kill()
            print("E2B sandbox terminated")


class QwenVLAPIModel(Model):
    """Model wrapper for Qwen2.5VL API with fallback mechanism"""
    
    def __init__(
        self, 
        model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        hf_token: str = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.base_model = HfApiModel(
            model_id="https://n5wr7lfx6wp94tvl.us-east-1.aws.endpoints.huggingface.cloud",
            token=hf_token,
        )
        self.fallback_model = HfApiModel(
            model_id,
            provider="hyperbolic",
            token=hf_token,
        )
        
    def __call__(
        self, 
        messages: List[Dict[str, Any]], 
        stop_sequences: Optional[List[str]] = None, 
        **kwargs
    ) -> ChatMessage:
        
        try:
            message = self.base_model(messages, stop_sequences, **kwargs)
            return message
        except Exception as e:
            print(f"Base model failed with error: {e}. Calling fallback model.")
                
        # Continue to fallback
        try:
            message = self.fallback_model(messages, stop_sequences, **kwargs)
            return message
        except Exception as e:
            raise Exception(f"Both endpoints failed. Last error: {e}")

# class QwenVLAPIModel(Model):
#     """Model wrapper for Qwen2.5VL API with fallback mechanism"""
    
#     def __init__(
#         self, 
#         model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct",
#         provider: str = "hyperbolic",
#         hf_token: str = None,
#         hf_base_url: str = "https://n5wr7lfx6wp94tvl.us-east-1.aws.endpoints.huggingface.cloud"
#     ):
#         super().__init__()
#         self.model_path = model_path
#         self.model_id = model_path
#         self.provider = provider
#         self.hf_token = hf_token
#         self.hf_base_url = hf_base_url
        
#         # Initialize hyperbolic client
#         self.hyperbolic_client = InferenceClient(
#             provider=self.provider,
#         )

#         assert not self.hf_base_url.endswith("/v1/"), "Enter your base url without '/v1/' suffix."

#         # Initialize HF OpenAI-compatible client if token is provided
#         self.hf_client = None
#         from openai import OpenAI
#         self.hf_client = OpenAI(
#             base_url=self.hf_base_url + "/v1/",
#             api_key=self.hf_token
#         )
        
#     def __call__(
#         self, 
#         messages: List[Dict[str, Any]], 
#         stop_sequences: Optional[List[str]] = None, 
#         **kwargs
#     ) -> ChatMessage:
#         """Convert a list of messages to an API request with fallback mechanism"""
        
#         # Format messages once for both APIs
#         formatted_messages = self._format_messages(messages)
        
#         # First try the HF endpoint if available - THIS ALWAYS FAILS SO SKIPPING
#         try:
#             completion = self._call_hf_endpoint(
#                 formatted_messages, 
#                 stop_sequences, 
#                 **kwargs
#             )
#             print("SUCCESSFUL call of inference endpoint")
#             return ChatMessage(role=MessageRole.ASSISTANT, content=completion)
#         except Exception as e:
#             print(f"HF endpoint failed with error: {e}. Falling back to hyperbolic.")
#             # Continue to fallback
        
#         # Fallback to hyperbolic
#         try:
#             return self._call_hyperbolic(formatted_messages, stop_sequences, **kwargs)
#         except Exception as e:
#             raise Exception(f"Both endpoints failed. Last error: {e}")
    
#     def _format_messages(self, messages: List[Dict[str, Any]]):
#         """Format messages for API requests - works for both endpoints"""
        
#         formatted_messages = []
        
#         for msg in messages:
#             role = msg["role"]
#             content = []
            
#             if isinstance(msg["content"], list):
#                 for item in msg["content"]:
#                     if item["type"] == "text":
#                         content.append({"type": "text", "text": item["text"]})
#                     elif item["type"] == "image":
#                         # Handle image path or direct image object
#                         if isinstance(item["image"], str):
#                             # Image is a path
#                             with open(item["image"], "rb") as image_file:
#                                 base64_image = base64.b64encode(image_file.read()).decode("utf-8")
#                         else:
#                             # Image is a PIL image or similar object
#                             img_byte_arr = BytesIO()
#                             base64_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")

#                         content.append({
#                             "type": "image_url",
#                             "image_url": {
#                                 "url": f"data:image/png;base64,{base64_image}"
#                             }
#                         })
#             else:
#                 # Plain text message
#                 content = [{"type": "text", "text": msg["content"]}]
            
#             formatted_messages.append({"role": role, "content": content})
        
#         return formatted_messages

#     def _call_hf_endpoint(self, formatted_messages, stop_sequences=None, **kwargs):
#         """Call the Hugging Face OpenAI-compatible endpoint"""

#         # Extract parameters with defaults
#         max_tokens = kwargs.get("max_new_tokens", 4096)
#         temperature = kwargs.get("temperature", 0.7)
#         top_p = kwargs.get("top_p", 0.9)
#         stream = kwargs.get("stream", False)
        
#         completion = self.hf_client.chat.completions.create(
#             model="tgi",  # Model name for the endpoint
#             messages=formatted_messages,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             stream=stream,
#             stop=stop_sequences
#         )
        
#         if stream:
#             # For streaming responses, return a generator
#             def stream_generator():
#                 for chunk in completion:
#                     yield chunk.choices[0].delta.content or ""
#             return stream_generator()
#         else:
#             # For non-streaming, return the full text
#             return completion.choices[0].message.content
    
#     def _call_hyperbolic(self, formatted_messages, stop_sequences=None, **kwargs):
#         """Call the hyperbolic API"""
        
#         completion = self.hyperbolic_client.chat.completions.create(
#             model=self.model_path,
#             messages=formatted_messages,
#             max_tokens=kwargs.get("max_new_tokens", 4096),
#             temperature=kwargs.get("temperature", 0.7),
#             top_p=kwargs.get("top_p", 0.9),
#             stop=stop_sequences
#         )
        
#         # Extract the response text
#         output_text = completion.choices[0].message.content
        
#         return ChatMessage(role=MessageRole.ASSISTANT, content=output_text)
    
#     def to_dict(self) -> Dict[str, Any]:
#         """Convert the model to a dictionary"""
#         return {
#             "class": self.__class__.__name__,
#             "model_path": self.model_path,
#             "provider": self.provider,
#             "hf_base_url": self.hf_base_url,
#             # We don't save the API keys for security reasons
#         }
    
#     @classmethod
#     def from_dict(cls, data: Dict[str, Any]) -> "QwenVLAPIModel":
#         """Create a model from a dictionary"""
#         return cls(
#             model_path=data.get("model_path", "Qwen/Qwen2.5-VL-72B-Instruct"),
#             provider=data.get("provider", "hyperbolic"),
#             hf_base_url=data.get("hf_base_url", "https://s41ydkv0iyjeokyj.us-east-1.aws.endpoints.huggingface.cloud"),
#         )