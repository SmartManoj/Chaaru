import os
import time
import subprocess
import platform
import tempfile
from io import BytesIO
from PIL import Image, ImageGrab
import pyautogui

class LocalDesktopStream:
    """
    A class to simulate the streaming functionality of E2B desktop
    but using the local desktop instead.
    """
    def __init__(self):
        self.is_running = False
        self.auth_key = "local"
        self.screenshot_path = os.path.join(os.getcwd(), "local_desktop_screenshot.png")
        self.update_screenshot()
    
    def update_screenshot(self):
        """Take a screenshot and save it to the screenshot path"""
        try:
            screenshot = ImageGrab.grab()
            screenshot.save(self.screenshot_path)
            return True
        except Exception as e:
            print(f"Error taking screenshot: {e}")
            return False
    
    def start(self, require_auth=False):
        """Start the local desktop stream"""
        self.is_running = True
        self.update_screenshot()
        return True
    
    def stop(self):
        """Stop the local desktop stream"""
        self.is_running = False
        if os.path.exists(self.screenshot_path):
            try:
                os.remove(self.screenshot_path)
            except:
                pass
        return True
    
    def get_auth_key(self):
        """Get the authentication key for the stream"""
        return self.auth_key
    
    def get_url(self, auth_key=None):
        """Get the URL for the stream - in this case, we'll use a local file path"""
        self.update_screenshot()
        return f"file:///{self.screenshot_path.replace(os.sep, '/')}"


class LocalDesktop:
    """
    A class to simulate the E2B Sandbox class but using the local desktop instead.
    This provides the same interface as the E2B Sandbox class but operates on the local machine.
    """
    def __init__(self, api_key=None, resolution=(1024, 768), dpi=96, timeout=300, template=None):
        self.sandbox_id = "local-desktop"
        self.resolution = resolution
        self.dpi = dpi
        self.timeout = timeout
        self.stream = LocalDesktopStream()
        self.last_screenshot = None
    
    def get_screen_size(self):
        """Get the screen size of the local desktop"""
        return self.resolution
    
    def screenshot(self, format="bytes"):
        """Take a screenshot of the local desktop"""
        # Capture the entire screen
        screenshot = ImageGrab.grab()
        
        # Resize to match the configured resolution if needed
        if screenshot.size != self.resolution:
            screenshot = screenshot.resize(self.resolution)
        
        if format == "bytes":
            # Convert to bytes
            img_byte_arr = BytesIO()
            screenshot.save(img_byte_arr, format='PNG')
            self.last_screenshot = img_byte_arr.getvalue()
            return self.last_screenshot
        else:
            # Save to a temporary file and return the path
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
            screenshot.save(temp_file.name)
            return temp_file.name
    
    def move_mouse(self, x, y):
        """Move the mouse to the specified coordinates"""
        try:
            pyautogui.moveTo(x, y)
            print(f"Moving mouse to ({x}, {y})")
            return True
        except Exception as e:
            print(f"Error moving mouse: {e}")
            return False
    
    def left_click(self):
        """Perform a left click at the current mouse position"""
        try:
            pyautogui.click()
            print("Left click")
            return True
        except Exception as e:
            print(f"Error left clicking: {e}")
            return False
    
    def right_click(self):
        """Perform a right click at the current mouse position"""
        try:
            pyautogui.rightClick()
            print("Right click")
            return True
        except Exception as e:
            print(f"Error right clicking: {e}")
            return False
    
    def double_click(self):
        """Perform a double click at the current mouse position"""
        try:
            pyautogui.doubleClick()
            print("Double click")
            return True
        except Exception as e:
            print(f"Error double clicking: {e}")
            return False
    
    def write(self, text, delay_in_ms=75):
        """Type the specified text"""
        try:
            interval = delay_in_ms / 1000  # Convert ms to seconds
            pyautogui.write(text, interval=interval)
            print(f"Typing: {text}")
            return True
        except Exception as e:
            print(f"Error typing text: {e}")
            return False
    
    def press(self, key):
        """Press the specified key or key combination"""
        try:
            # Handle key combinations (list of keys)
            if isinstance(key, list):
                # For key combinations, use hotkey
                pyautogui.hotkey(*key)
            else:
                # For single keys
                pyautogui.press(key)
            print(f"Pressing key: {key}")
            return True
        except Exception as e:
            print(f"Error pressing key: {e}")
            return False
    
    def drag(self, start_coords, end_coords):
        """Drag from start coordinates to end coordinates"""
        try:
            x1, y1 = start_coords
            x2, y2 = end_coords
            pyautogui.moveTo(x1, y1)
            pyautogui.dragTo(x2, y2, duration=0.5)
            print(f"Dragging from {start_coords} to {end_coords}")
            return True
        except Exception as e:
            print(f"Error dragging: {e}")
            return False
    
    def scroll(self, direction="down", amount=2):
        """Scroll in the specified direction"""
        try:
            # PyAutoGUI uses positive values for scrolling up, negative for down
            scroll_amount = -amount if direction.lower() == "down" else amount
            pyautogui.scroll(scroll_amount * 100)  # Multiply by 100 for more noticeable scrolling
            print(f"Scrolling {direction} by {amount}")
            return True
        except Exception as e:
            print(f"Error scrolling: {e}")
            return False
    
    def open(self, url):
        """Open a URL in the default browser"""
        try:
            print(f"Opening URL: {url}")
            # Make sure URL has http/https prefix
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
                
            # Use the default system browser to open the URL
            if platform.system() == 'Windows':
                os.system(f'start {url}')
            elif platform.system() == 'Darwin':  # macOS
                os.system(f'open {url}')
            else:  # Linux
                os.system(f'xdg-open {url}')
                
            # Give the browser time to open
            time.sleep(2)
            
            return True
        except Exception as e:
            print(f"Error opening URL: {e}")
            return False
    
    def commands(self):
        """Return a commands object with a run method"""
        return CommandsRunner()
    
    def kill(self):
        """Kill the sandbox"""
        self.stream.stop()
        print("Local desktop sandbox terminated")
        return True


class CommandsRunner:
    """A class to simulate the commands functionality of E2B desktop"""
    def run(self, command):
        """Run a command on the local machine"""
        print(f"Running command: {command}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout
        except Exception as e:
            print(f"Error running command: {e}")
            return str(e)