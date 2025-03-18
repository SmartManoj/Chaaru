import os
import time
import base64
from io import BytesIO
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple
import json

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


class E2BVisionAgent(CodeAgent):
    """Agent for e2b desktop automation with Qwen2.5VL vision capabilities"""
    def __init__(
        self,
        model: HfApiModel,
        data_dir: str,
        desktop: Sandbox,
        tools: List[tool] = None,
        max_steps: int = 200,
        verbosity_level: LogLevel = 4,
        planning_interval: int = 15,
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
        print(f"Verbosity level set to {verbosity_level}")


        # Initialize base agent
        super().__init__(
            tools=tools or [],
            model=model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            planning_interval = self.planning_interval,
            **kwargs
        )


        # Add screen info to state
        self.state["screen_width"] = self.width
        self.state["screen_height"] = self.height


        # Add default tools
        self._setup_desktop_tools()
        self.step_callbacks.append(self.take_snapshot_callback)


    def initialize_system_prompt(self):
        return """You are a desktop automation assistant that can control a remote desktop environment.
You only have access to the following tools to interact with the desktop, no additional ones:

- click(x, y): Performs a left-click at the specified coordinates
- right_click(x, y): Performs a right-click at the specified coordinates
- double_click(x, y): Performs a double-click at the specified coordinates
- move_mouse(x, y): Moves the mouse cursor to the specified coordinates
- type_text(text): Types the specified text at the current cursor position
- press_key(key): Presses a keyboard key (e.g., "Return", "tab", "ctrl+c")
- scroll(direction, amount): Scrolls a website in a browser or a document (direction can be "up" or "down", a common amount is 1 or 2 scroll("down",1) ). DO NOT use scroll to move through linux desktop menus.
- wait(seconds): Waits for the specified number of seconds. Very useful in case the prior order is still executing (for example starting very heavy applications like browsers or office apps)
- open_url(url): Directly opens a browser with the specified url, saves time compared to clicking in a browser and going through the initial setup wizard.
- final_answer("YOUR FINAL ANSWER TEXT"): Announces that the task requested is completed and provides a final text

The desktop has a resolution of {resolution_x}x{resolution_y}.

IMPORTANT:
- Remember the tools that you have as those can save you time, for example open_url to enter a website rather than searching for the browser in the OS.
- Whenever you click, MAKE SURE to click in the middle of the button, text, link or any other clickable element. Not under, not on the side. IN THE MIDDLE. In menus it is always better to click in the middle of the text rather than in the tiny icon. Calculate extremelly well the coordinates. A mistake here can make the full task fail.
- To navigate the desktop you should open menus and click. Menus usually expand with more options, the tiny triangle next to some text in a menu means that menu expands. For example in Office in the Applications menu expands showing presentation or writing applications. 
- Always analyze the latest screenshot carefully before performing actions. If you clicked somewhere in the previous action and in the screenshot nothing happened, make sure the mouse is where it should be. Otherwise you can see that the coordinates were wrong.

You must proceed step by step:
1. Understand the task thoroughly
2. Break down the task into logical steps
3. For each step:
   a. Analyze the current screenshot to identify UI elements
   b. Plan the appropriate action with precise coordinates
   c. Execute ONE action at a time using the proper tool
   d. Wait for the action to complete before proceeding

After each action, you'll receive an updated screenshot. Review it carefully before your next action.

COMMAND FORMAT:
Always format your actions as Python code blocks. For example:

```python
click(250, 300)
```<end_code>


TASK EXAMPLE:
For a task like "Open a text editor and type 'Hello World'":
1- First, analyze the screenshot to find the Applications menu and click on it being very precise, clicking in the middle of the text 'Applications':
```python
click(50, 10) 
```<end_code>
2- Remembering that menus are navigated through clicking, after analyzing the screenshot with the applications menu open we see that a notes application probably fits in the Accessories section (we see it is a section in the menu thanks to the tiny white triangle after the text accessories). We look for Accessories and click on it being very precise, clicking in the middle of the text 'Accessories'. DO NOT try to move through the menus with scroll, it won't work:
```python
click(76, 195) 
```<end_code>
3- Remembering that menus are navigated through clicking, after analyzing the screenshot with the submenu Accessories open, look for 'Text Editor' and click on it being very precise, clicking in the middle of the text 'Text Editor':
```python
click(241, 441) 
```<end_code>
4- Once Notepad is open, type the requested text:
```python
type_text("Hello World")
```<end_code>

5- Task is completed:
```python
final_answer("Done")
```<end_code>

Remember to:

Always wait for appropriate loading times
Use precise coordinates based on the current screenshot
Execute one action at a time
Verify the result before proceeding to the next step
Use click to move through menus on the desktop and scroll for web and specific applications.
REMEMBER TO ALWAYS CLICK IN THE MIDDLE OF THE TEXT, NOT ON THE SIDE, NOT UNDER.
""".format(resolution_x=self.width, resolution_y=self.height)

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
            return f"Moved mouse to coordinates ({x}, {y})"

        @tool
        def type_text(text: str, delay_in_ms: int = 75) -> str:
            """
            Types the specified text at the current cursor position
            Args:
                text: The text to type
                delay_in_ms: Delay between keystrokes in milliseconds
            """
            self.desktop.write(text, delay_in_ms=delay_in_ms)
            return f"Typed text: '{text}'"

        @tool
        def press_key(key: str) -> str:
            """
            Presses a keyboard key
            Args:
                key: The key to press (e.g., "Return", "tab", "ctrl+c")
            """
            if key == "enter":
                key = "Return"
            self.desktop.press(key)
            return f"Pressed key: {key}"

        @tool
        def go_back() -> str:
            """
            Goes back to the previous page in the browser.
            Args:
            """
            self.desktop.press(["alt", "left"])
            return "Went back one page"

        @tool
        def scroll(direction: str = "down", amount: int = 1) -> str:
            """
            Scrolls the page
            Args:
                direction: The direction to scroll ("up" or "down"), defaults to "down"
                amount: The amount to scroll. A good amount is 1 or 2.
            """
            self.desktop.scroll(direction=direction, amount=amount)
            return f"Scrolled {direction} by {amount}"

        @tool
        def wait(seconds: float) -> str:
            """
            Waits for the specified number of seconds
            Args:
                seconds: Number of seconds to wait
            """
            time.sleep(seconds)
            return f"Waited for {seconds} seconds"

        @tool
        def open_url(url: str) -> str:
            """
            Opens the specified URL in the default browser
            Args:
                url: The URL to open
            """
            # Make sure URL has http/https prefix
            if not url.startswith(("http://", "https://")):
                url = "https://" + url

            self.desktop.open(url)
            # Give it time to load
            time.sleep(2)
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


    def store_metadata_to_file(self, agent) -> None:
        metadata_path = os.path.join(self.data_dir, "metadata.json")
        output = {}
        output_memory  = self.write_memory_to_messages()
        a = open(metadata_path,"w")
        a.write(json.dumps(output_memory))
        a.close()

    def write_memory_to_messages(self) -> List[Dict[str, Any]]:
        """Convert memory to messages for the model"""
        messages = [{"role": MessageRole.SYSTEM, "content": [{"type": "text", "text": self.system_prompt}]}]

        for memory_step in self.memory.steps:
            if hasattr(memory_step, "task") and memory_step.task:
                # Add task message if it exists
                messages.append({
                    "role": MessageRole.USER, 
                    "content": [{"type": "text", "text": memory_step.task}]
                })
                continue  # Skip to next step after adding task
                
            # Process model output message if it exists
            if hasattr(memory_step, "model_output") and memory_step.model_output:
                messages.append({
                    "role": MessageRole.ASSISTANT, 
                    "content": [{"type": "text", "text": memory_step.model_output}]
                })
            
            # Process observations and images
            observation_content = []
            

            # Add text observations if any
            if hasattr(memory_step, "observations") and memory_step.observations:
                observation_content.append({"type": "text", "text": f"Observation: {memory_step.observations}"})
            
            # Add error if present and didn't already add observations
            if hasattr(memory_step, "error") and memory_step.error:
                observation_content.append({"type": "text", "text": f"Error: {memory_step.error}"})
            
            # Add user message with content if we have any
            if observation_content:
                messages.append({
                    "role": MessageRole.USER,
                    "content": observation_content
                })
        
        return messages


    def write_memory_to_messages(self, summary_mode: Optional[bool] = False) -> List[Dict[str, Any]]:
        """Convert memory to messages for the model"""
        messages = [{"role": MessageRole.SYSTEM, "content": [{"type": "text", "text": self.system_prompt}]}]
        # Get the last memory step
        last_step = self.memory.steps[-1] if self.memory.steps else None

        for memory_step in self.memory.steps:
            if hasattr(memory_step, "task") and memory_step.task:
                # Add task message if it exists
                messages.append({
                    "role": MessageRole.USER, 
                    "content": [{"type": "text", "text": memory_step.task}]
                })
                continue  # Skip to next step after adding task
                
            # Process model output message if it exists
            if hasattr(memory_step, "model_output") and memory_step.model_output:
                messages.append({
                    "role": MessageRole.ASSISTANT, 
                    "content": [{"type": "text", "text": memory_step.model_output}]
                })
            
            # Process observations and images
            observation_content = []
            
            # Add screenshot image paths if present
            if memory_step is last_step and hasattr(memory_step, "observations_images") and memory_step.observations_images:
                self.logger.log(f"Found {len(memory_step.observations_images)} image paths in step", level=LogLevel.DEBUG)
                for img_path in memory_step.observations_images:
                    if isinstance(img_path, str) and os.path.exists(img_path):
                        observation_content.append({"type": "image", "image": img_path})
                    elif isinstance(img_path, Image.Image):
                        screenshot_path = f"screenshot_{int(time.time() * 1000)}.png"
                        img_path.save(screenshot_path)
                        observation_content.append({"type": "image", "image": screenshot_path})
                    else:
                        self.logger.log(f"  - Skipping invalid image: {type(img_path)}", level=LogLevel.ERROR)
            
            # Add text observations if any
            if hasattr(memory_step, "observations") and memory_step.observations:
                self.logger.log(f"  - Adding text observation", level=LogLevel.DEBUG)
                observation_content.append({"type": "text", "text": f"Observation: {memory_step.observations}"})
            
            # Add error if present and didn't already add observations
            if hasattr(memory_step, "error") and memory_step.error:
                self.logger.log(f"  - Adding error message", level=LogLevel.DEBUG)
                observation_content.append({"type": "text", "text": f"Error: {memory_step.error}"})
            
            # Add user message with content if we have any
            if observation_content:
                self.logger.log(f"  - Adding user message with {len(observation_content)} content items", level=LogLevel.DEBUG)
                messages.append({
                    "role": MessageRole.USER,
                    "content": observation_content
                })
        
        # # Check for images in final message list
        # image_count = 0
        # for msg in messages:
        #     if isinstance(msg.get("content"), list):
        #         for item in msg["content"]:
        #             if isinstance(item, dict) and item.get("type") == "image":
        #                 image_count += 1
        
        # print(f"Created {len(messages)} messages with {image_count} image paths")

        return messages
    
    def take_snapshot_callback(self, memory_step: ActionStep, agent=None) -> None:
        """Callback that takes a screenshot + memory snapshot after a step completes"""
        current_step = memory_step.step_number
        print(f"Taking screenshot for step {current_step}")
        try:
            time.sleep(2.0) # Let things happen on the desktop
            screenshot_bytes = self.desktop.screenshot()
            image = Image.open(BytesIO(screenshot_bytes))

            # Create a filename with step number
            screenshot_path = os.path.join(self.data_dir, f"step_{current_step:03d}.png")
            image.save(screenshot_path)
            print(f"Saved screenshot to {screenshot_path}")

            for previous_memory_step in agent.memory.steps:  # Remove previous screenshots from logs for lean processing
                if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                    previous_memory_step.observations_images = None

            # Add to the current memory step
            # memory_step.observations_images = [image.copy()] # This takes the original image directly.
            memory_step.observations_images = [screenshot_path]

            #Storing memory and metadata to file:
            self.store_metadata_to_file(agent)
            

        except Exception as e:
            print(f"Error taking screenshot: {e}")

    def close(self):
        """Clean up resources"""
        if self.desktop:
            print("Stopping e2b stream...")
            self.desktop.stream.stop()

            print("Killing e2b sandbox...")
            self.desktop.kill()
            print("E2B sandbox terminated")



class QwenVLAPIModel(Model):
    """Model wrapper for Qwen2.5VL API"""
    
    def __init__(
        self, 
        model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct", 
        provider: str = "hyperbolic"
    ):
        super().__init__()
        self.model_path = model_path
        self.model_id = model_path
        self.provider = provider
        
        self.client = InferenceClient(
            provider=self.provider,
        )
        
    def __call__(
        self, 
        messages: List[Dict[str, Any]], 
        stop_sequences: Optional[List[str]] = None, 
        **kwargs
    ) -> ChatMessage:
        """Convert a list of messages to an API request and return the response"""
        # # Count images in messages - debug
        # image_count = 0
        # for msg in messages:
        #     if isinstance(msg.get("content"), list):
        #         for item in msg["content"]:
        #             if isinstance(item, dict) and item.get("type") == "image":
        #                 image_count += 1
        
        # print(f"QwenVLAPIModel received {len(messages)} messages with {image_count} images")
        
        # Format the messages for the API

        formatted_messages = []
        
        for msg in messages:
            role = msg["role"]
            if isinstance(msg["content"], list):
                content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        content.append({"type": "text", "text": item["text"]})
                    elif item["type"] == "image":
                        # Handle image path or direct image object
                        if isinstance(item["image"], str):
                            # Image is a path
                            with open(item["image"], "rb") as image_file:
                                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                        else:
                            # Image is a PIL image or similar object
                            img_byte_arr = io.BytesIO()
                            item["image"].save(img_byte_arr, format="PNG")
                            base64_image = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
                        
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        })
            else:
                content = [{"type": "text", "text": msg["content"]}]
                
            formatted_messages.append({"role": role, "content": content})
        
        # Make the API request
        completion = self.client.chat.completions.create(
            model=self.model_path, 
            messages=formatted_messages, 
            max_tokens=kwargs.get("max_new_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
        )
        
        # Extract the response text
        output_text = completion.choices[0].message.content
        
        return ChatMessage(role=MessageRole.ASSISTANT, content=output_text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary"""
        return {
            "class": self.__class__.__name__,
            "model_path": self.model_path,
            "provider": self.provider,
            # We don't save the API key for security reasons
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QwenVLAPIModel":
        """Create a model from a dictionary"""
        return cls(
            model_path=data.get("model_path", "Qwen/Qwen2.5-VL-72B-Instruct"),
            provider=data.get("provider", "hyperbolic"),
        )
