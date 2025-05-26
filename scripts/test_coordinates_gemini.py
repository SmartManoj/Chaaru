import asyncio
import os
from google import genai
from google.genai import types
from PIL import ImageGrab
import pyautogui
screenshot = ImageGrab.grab()

# Resize to match the configured resolution if needed
# if screenshot.size != (1024, 768):
    # screenshot = screenshot.resize((1024, 768))

screenshot.save('screenshot.png')

# Initialize the client
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)
image_url = fr'screenshot.png'
import base64
with open(image_url, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# image_url = f"data:image/png;base64,{encoded_string}"
image_url = {"mime_type": "image/png", "data": encoded_string}

CONFIG = types.LiveConnectConfig(
    response_modalities=[
        "TEXT",
    ],
    media_resolution="MEDIA_RESOLUTION_MEDIUM",
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Zephyr")
        )
    ),
    context_window_compression=types.ContextWindowCompressionConfig(
        trigger_tokens=25600,
        sliding_window=types.SlidingWindow(target_tokens=12800),
    ),
)
# Simple text completion using gemini-2.0-flash-001 (regular version)
async def main():
    async with client.aio.live.connect(
        model="models/gemini-2.0-flash-live-001",
        config=CONFIG,
    ) as session:
        
        msg=f"""just give the position index and the coordinates of chrome icon in task bar from the screenshot.
the screen resolution is {pyautogui.size()}
format:
# of position : <number>
full resolution: <width> <height>
coordinates: <x> <y>"""
        # await session.send_client_content(turns=[types.Content(role="user", parts=[types.Part(media=image_url)])])
        await session.send(input=image_url)
        await asyncio.sleep(3)
        await session.send_realtime_input(text=msg)
        print("sent")
        # wait till send is done
        async for event in session.receive():
            if event.text:
                print(event.text, end="")


if __name__ == "__main__":
    asyncio.run(main())
