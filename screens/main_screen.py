import os
import cv2
import time
import base64
import asyncio
import threading
import numpy as np
import sounddevice as sd
from io import BytesIO
from PIL import Image

import google.genai as genai
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.clock import Clock

# ------------ Helpers ------------
def encode_audio(data: np.ndarray) -> dict:
    return {
        "mime_type": "audio/pcm",
        "data": base64.b64encode(data.tobytes()).decode("UTF-8"),
    }

def encode_image(frame: np.ndarray) -> dict:
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image.save(output_bytes, "JPEG")
        bytes_data = output_bytes.getvalue()
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(bytes_data).decode("utf-8"),
    }

async def wait_for_item(queue: asyncio.Queue, timeout: float):
    try:
        return await asyncio.wait_for(queue.get(), timeout)
    except asyncio.TimeoutError:
        return None

# ------------ Gemini Handler ------------
class GeminiHandler:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        self.session = None
        self.audio_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()
        self.quit = asyncio.Event()
        self.last_frame_time = 0

    async def start(self):
        config = {"response_modalities": ["AUDIO"]}
        async with self.client.aio.live.connect(model="gemini-2.0-flash-exp", config=config) as session:
            self.session = session
            while not self.quit.is_set():
                turn = self.session.receive()
                try:
                    async for response in turn:
                        if data := response.data:
                            audio = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
                            await self.audio_queue.put(audio)
                        if response.text:  # optional text events
                            print("Agent:", response.text)
                except Exception as e:
                    print("Gemini error:", e)
                    break

    async def send_audio(self, array: np.ndarray):
        if self.session:
            await self.session.send(input=encode_audio(array))

    async def send_video(self, frame: np.ndarray):
        if self.session and (time.time() - self.last_frame_time) > 1:
            self.last_frame_time = time.time()
            await self.session.send(input=encode_image(frame))

    async def get_audio_reply(self):
        array = await wait_for_item(self.audio_queue, 0.01)
        if array is not None:
            return (24000, array)
        return None

    async def stop(self):
        if self.session:
            self.quit.set()
            await self.session.close()
            self.quit.clear()

# ------------ Kivy MainScreen ------------
class MainScreen(Screen):
    def __init__(self, **kwargs):
        super(MainScreen, self).__init__(**kwargs)
        self.handler = None
        self.loop = None
        self.thread = None
        self.cap = None
        self.mic_stream = None
        self.speaker = None

    def on_pre_enter(self):
        """Start streaming when entering main screen"""
        self.handler = GeminiHandler(api_key=os.getenv("GEMINI_API_KEY"))

        # Background asyncio loop
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run_agent())

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

        # Start camera preview updater
        Clock.schedule_interval(self.update_camera_widget, 1/30)

    async def run_agent(self):
        asyncio.create_task(self.handler.start())

        # Microphone -> Gemini
        def mic_callback(indata, frames, time_, status):
            if status:
                print("Mic:", status)
            array = (indata * 32767).astype(np.int16)
            asyncio.run_coroutine_threadsafe(self.handler.send_audio(array), self.loop)

        self.mic_stream = sd.InputStream(channels=1, samplerate=16000, blocksize=512, callback=mic_callback)
        self.mic_stream.start()

        # Speaker (Gemini -> User)
        self.speaker = sd.OutputStream(channels=1, samplerate=24000, blocksize=512)
        self.speaker.start()

        # Camera
        self.cap = cv2.VideoCapture(0)

        try:
            while not self.handler.quit.is_set():
                ret, frame = self.cap.read()
                if ret:
                    await self.handler.send_video(frame)

                # Play audio replies
                reply = await self.handler.get_audio_reply()
                if reply:
                    _, audio_np = reply
                    audio_np = audio_np.astype(np.float32) / 32767.0

                    # Ensure it's mono (N,1)
                    if audio_np.ndim == 1:  
                        audio_np = audio_np[:, np.newaxis]  # shape (N,1)
                    elif audio_np.ndim == 2 and audio_np.shape[0] == 1:
                        audio_np = audio_np.T  # convert (1,N) → (N,1)
                    elif audio_np.ndim == 2 and audio_np.shape[1] > 1:
                        # stereo → take left channel
                        audio_np = audio_np[:, 0:1]

                    self.speaker.write(audio_np)

                await asyncio.sleep(0.01)
        finally:
            if self.cap: self.cap.release()
            if self.mic_stream: self.mic_stream.stop()
            if self.speaker: self.speaker.stop()

    def update_camera_widget(self, dt):
        """Update camera_widget canvas with live feed"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 0)  
                buf = frame.tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.ids.camera_widget.texture = texture

    def on_leave(self):
        """Stop everything when leaving main screen"""
        if self.handler:
            asyncio.run_coroutine_threadsafe(self.handler.stop(), self.loop)
        if self.cap: self.cap.release()
        if self.mic_stream: self.mic_stream.stop()
        if self.speaker: self.speaker.stop()
        Clock.unschedule(self.update_camera_widget)
