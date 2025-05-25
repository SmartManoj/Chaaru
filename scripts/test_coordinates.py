from PIL import ImageGrab
screenshot = ImageGrab.grab()

# Resize to match the configured resolution if needed
# if screenshot.size != (1024, 768):
    # screenshot = screenshot.resize((1024, 768))

screenshot.save('screenshot.png')

img = r'screenshot.png'
import litellm
from dotenv import load_dotenv
import pyautogui

load_dotenv()
system_prompt = f'''You are a desktop automation assistant that can control a remote desktop environment.
The screen resolution is {pyautogui.size()}
'''
prompt = '''just give the position and the coordinates of chrome icon in task bar from the screenshot.
format:
position: <<position>>
full resolution: <<full resolution>>
coordinates: <<coordinates>>
'''

image_url = fr'screenshot.png'
import base64
with open(image_url, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

image_url = f"data:image/png;base64,{encoded_string}"

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt
            },
            {
                "type": "image_url",
                "image_url": {"url": image_url}
            }
        ]
    }
]
response = litellm.completion(
    model='openrouter/Qwen/Qwen2.5-VL-72B-Instruct:free',
    # model='gemini/gemini-2.5-flash-preview-05-20',
    # model='vertex_ai/gemini-2.5-pro-preview-05-06',
    messages=messages,
    temperature=0,
    seed=42,
    drop_params=True

)

msg = response.choices[0].message.content
print(msg)
coordinates = msg.split('coordinates:')[1]
from pyautogui import moveTo
moveTo(*eval(coordinates))
