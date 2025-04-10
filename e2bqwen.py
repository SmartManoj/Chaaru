import os
import time
from io import BytesIO
from typing import Any, Dict, List, Optional
import unicodedata

# E2B imports
from e2b_desktop import Sandbox
from PIL import Image

# SmolaAgents imports
from smolagents import CodeAgent, tool, HfApiModel
from smolagents.memory import ActionStep, TaskStep
from smolagents.models import ChatMessage, Model
from smolagents.agents import populate_template
from smolagents.monitoring import LogLevel
from smolagents.agent_types import AgentImage
from PIL import ImageDraw
from datetime import datetime

E2B_SYSTEM_PROMPT_TEMPLATE = """You are a desktop automation assistant that can control a remote desktop environment.
The current date is <<current_date>>.
<action process>
You will be given a task to solve in several steps. At each step you will perform an action.
After each action, you'll receive an updated screenshot. 
Then you will proceed as follows, with these sections: don't skip any!

Short term goal: ...
Where I am: ...
What I see: ...
Reflection: ...
Action: ...
Code:
```python
click(254, 308)
```<end_code>
</action_process>

<tools>
On top of performing computations in the Python code snippets that you create, you only have access to these tools to interact with the desktop, no additional ones:
{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
    Takes inputs: {{tool.inputs}}
    Returns an output of type: {{tool.output_type}}
{%- endfor %}

The desktop has a resolution of <<resolution_x>>x<<resolution_y>> pixels, take it into account to decide clicking coordinates.
If you clicked somewhere in the previous action, a green crosshair will appear at the exact location of the previous click.
The image might have change since then but the cross stays at the previous click. If your click seems to have changed nothing, check that this location is exactly where you intended to click. Otherwise correct the click coordinates.
</tools>

<code_format>
Always format your actions as Python code blocks, as shown below:
Code:
```python
click(254, 308)
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
click(51, 8) 
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
What I see: Nothing has changed compared to previous screenshot. Under the open submenu Accessories, I still see 'Text Editor'. The green cross is off from the element.
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
Use precise coordinates based on the current screenshot for mouse movements and clicks. 
Whenever you click, MAKE SURE to click in the middle of the button, text, link or any other clickable element. Not under, not on the side. IN THE MIDDLE, else you risk to miss it.
In menus it is always better to click in the middle of the text rather than in the tiny icon. Calculate extremelly well the coordinates. A mistake here can make the full task fail.
The desktop has a resolution of <<resolution_x>>x<<resolution_y>> pixels: NEVER USE HYPOTHETIC OR ASSUMED COORDINATES, USE TRUE COORDINATES that you can see from the screenshot.
Sometimes you may have missed a click, so never assume that you're on the right page, always make sure that your previous action worked. In the screenshot you can see if the mouse is out of the clickable area. Pay special attention to this.
</click_guidelines>

<general_guidelines>
Always analyze the latest screenshot carefully before performing actions.
You can wait for appropriate loading times using the wait() tool. But don't wait forever, sometimes you've just misclicked and the process didn't launch.
Execute one action at a time: don't try to pack a click and typing in one action.
On each step, look at the last screenshot and action to validate if previous steps worked and decide the next action. If you repeated an action already without effect, it means that this action is useless: don't repeat it and try something else.
Use click to move through menus on the desktop and scroll for web and specific applications.
Always analyze the latest screenshot carefully before performing actions.
Desktop menus usually expand with more options, the tiny triangle next to some text in a menu means that menu expands. For example in Office in the Applications menu expands showing presentation or writing applications. 
NEVER CLICK THE WEB BROWSER ICON TO OPEN THE WEB BROWSER: use open_url directly.
In browser, ignore any sign in popups while they don't interfere with your usage of the browser.
</general_guidelines>
""".replace("<<current_date>>", datetime.now().strftime("%A, %d-%B-%Y"))


def draw_marker_on_image(image_copy, click_coordinates):
    x, y = click_coordinates
    draw = ImageDraw.Draw(image_copy)
    cross_size, linewidth = 10, 3
    # Draw cross
    draw.line((x - cross_size, y, x + cross_size, y), fill="green", width=linewidth)
    draw.line((x, y - cross_size, x, y + cross_size), fill="green", width=linewidth)
    # Add a circle around it for better visibility
    draw.ellipse(
        (
            x - cross_size * 2,
            y - cross_size * 2,
            x + cross_size * 2,
            y + cross_size * 2,
        ),
        outline="green",
        width=linewidth,
    )
    return image_copy


def get_agent_summary_erase_images(agent):
    for memory_step in agent.memory.steps:
        if hasattr(memory_step, "observations_images"):
            memory_step.observations_images = None
        if hasattr(memory_step, "task_images"):
            memory_step.task_images = None
    return agent.write_memory_to_messages()


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
        planning_interval: int = None,
        use_v1_prompt: bool = False,
        **kwargs,
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

        self.use_v1_prompt = use_v1_prompt
        # Initialize base agent
        super().__init__(
            tools=tools or [],
            model=model,
            max_steps=max_steps,
            verbosity_level=verbosity_level,
            planning_interval=self.planning_interval,
            **kwargs,
        )
        self.prompt_templates["system_prompt"] = E2B_SYSTEM_PROMPT_TEMPLATE.replace(
            "<<resolution_x>>", str(self.width)
        ).replace("<<resolution_y>>", str(self.height))

        # Add screen info to state
        self.state["screen_width"] = self.width
        self.state["screen_height"] = self.height

        # Add default tools
        self.logger.log("Setting up agent tools...")
        self._setup_desktop_tools()
        self.step_callbacks.append(self.take_screenshot_callback)

    def initialize_system_prompt(self) -> str:
        if False:
            return """You are a desktop automation assistant that can control a remote desktop environment.
You only have access to the following tools to interact with the desktop, no additional ones:
- click(x, y): Performs a left-click at the specified coordinates
- right_click(x, y): Performs a right-click at the specified coordinates
- double_click(x, y): Performs a double-click at the specified coordinates
- move_mouse(x, y): Moves the mouse cursor to the specified coordinates
- type_text(text): Types the specified text at the current cursor position
- press_key(key): Presses a keyboard key (e.g., "Return", "tab", "ctrl+c")
- scroll(x, y, direction, amount): Scrolls a website in a browser or a document (direction can be "up" or "down", a common amount is 1 or 2 scroll("down",1) ). DO NOT use scroll to move through linux desktop menus. x, y, is the mouse position to scroll on.
- wait(seconds): Waits for the specified number of seconds. Very useful in case the prior order is still executing (for example starting very heavy applications like browsers or office apps)
- open_url(url): Directly opens a browser with the specified url, saves time compared to clicking in a browser and going through the initial setup wizard.
- drag_and_drop(x1, y1, x2, y2): Clicks [x1, y1], drags mouse to [x2, y2], then releases click.
- find_on_page_ctrl_f(search_string): Scroll the viewport to the first occurrence of the search string. This is equivalent to Ctrl+F.
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
click(250, 304)
```<end_code>
TASK EXAMPLE:
For a task like "Open a text editor and type 'Hello World'":
1- First, analyze the screenshot to find the Applications menu and click on it being very precise, clicking in the middle of the text 'Applications':
```python
click(52, 10) 
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
        else:
            print("USING v2 prompt")
            system_prompt = populate_template(
                self.prompt_templates["system_prompt"],
                variables={
                    "tools": self.tools,
                    "managed_agents": self.managed_agents,
                    "authorized_imports": (
                        "You can import from any package you want."
                        if "*" in self.authorized_imports
                        else str(self.authorized_imports)
                    ),
                },
            )
            assert system_prompt != self.prompt_templates["system_prompt"], (
                "Populating prompt template failed"
            )
            return system_prompt

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
            return "".join(
                c
                for c in unicodedata.normalize("NFD", text)
                if not unicodedata.combining(c)
            )

        @tool
        def type_text(text: str) -> str:
            """
            Types the specified text at the current cursor position.
            Args:
                text: The text to type
            """
            clean_text = normalize_text(text)
            self.desktop.write(clean_text, delay_in_ms=75)
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
            Goes back to the previous page in the browser. If using this tool doesn't work, just click the button directly.
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
        def scroll(x: int, y: int, direction: str = "down", amount: int = 2) -> str:
            """
            Moves the mouse to selected coordinates, then uses the scroll button: this could scroll the page or zoom, depending on the app. DO NOT use scroll to move through linux desktop menus.
            Args:
                x: The x coordinate (horizontal position) of the element to scroll/zoom
                y: The y coordinate (vertical position) of the element to scroll/zoom
                direction: The direction to scroll ("up" or "down"), defaults to "down". For zoom, "up" zooms in, "down" zooms out.
                amount: The amount to scroll. A good amount is 1 or 2.
            """
            self.desktop.move_mouse(x, y)
            self.desktop.scroll(direction=direction, amount=amount)
            message = f"Scrolled {direction} by {amount}"
            self.logger.log(message)
            return message

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
            Directly opens a browser with the specified url: use this at start of web searches rather than trying to click the browser.
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

        @tool
        def find_on_page_ctrl_f(search_string: str) -> str:
            """
            Scroll the browser viewport to the first occurrence of the search string. This is equivalent to Ctrl+F. Use this to search on a pdf for instance.
            Args:
                search_string: The string to search for on the page.
            """
            self.desktop.press(["ctrl", "f"])
            time.sleep(0.3)
            clean_text = normalize_text(search_string)
            self.desktop.write(clean_text, delay_in_ms=75)
            time.sleep(0.3)
            self.desktop.press("enter")
            time.sleep(0.3)
            self.desktop.press("esc")
            output_message = f"Scrolled to the first occurrence of '{clean_text}'"
            self.logger.log(output_message)
            return output_message

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
        self.tools["find_on_page_ctrl_f"] = find_on_page_ctrl_f

    def take_screenshot_callback(self, memory_step: ActionStep, agent=None) -> None:
        """Callback that takes a screenshot + memory snapshot after a step completes"""
        self.logger.log("Analyzing screen content...")

        current_step = memory_step.step_number

        time.sleep(3.0)  # Let things happen on the desktop
        screenshot_bytes = self.desktop.screenshot(format="bytes")
        image = Image.open(BytesIO(screenshot_bytes))

        # Create a filename with step number
        screenshot_path = os.path.join(self.data_dir, f"step_{current_step:03d}.png")
        image.save(screenshot_path)

        image_copy = image.copy()

        if getattr(self, "click_coordinates", None):
            print("DRAWING MARKER")
            image_copy = draw_marker_on_image(image_copy, self.click_coordinates)

        self.last_marked_screenshot = AgentImage(screenshot_path)
        print(f"Saved screenshot for step {current_step} to {screenshot_path}")

        for previous_memory_step in (
            agent.memory.steps
        ):  # Remove previous screenshots from logs for lean processing
            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number <= current_step - 1
            ):
                previous_memory_step.observations_images = None
            elif isinstance(previous_memory_step, TaskStep):
                previous_memory_step.observations_images = None

            if (
                isinstance(previous_memory_step, ActionStep)
                and previous_memory_step.step_number == current_step - 1
            ):
                if (
                    previous_memory_step.tool_calls
                    and getattr(previous_memory_step.tool_calls[0], "arguments", None)
                    and memory_step.tool_calls
                    and getattr(memory_step.tool_calls[0], "arguments", None)
                ):
                    if (
                        previous_memory_step.tool_calls[0].arguments
                        == memory_step.tool_calls[0].arguments
                    ):
                        memory_step.observations += "\nWARNING: You've executed the same action several times in a row. MAKE SURE TO NOT UNNECESSARILY REPEAT ACTIONS."

        # Add the marker-edited image to the current memory step
        memory_step.observations_images = [image_copy]

        # memory_step.observations_images = [screenshot_path] # IF YOU USE THIS INSTEAD OF ABOVE, LAUNCHING A SECOND TASK BREAKS

        self.click_coordinates = None  # Reset click marker

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
            model_id="https://ahbeihft09ulicbf.us-east-1.aws.endpoints.huggingface.cloud",
            token=hf_token,
            max_tokens=4096,
        )
        self.fallback_model = HfApiModel(
            model_id,
            provider="nebius",
            token=hf_token,
            max_tokens=4096,
        )

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
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
