import cv2
import time
import base64
import asyncio
import numpy as np
from io import BytesIO
from PIL import Image
import google.genai as genai

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

        # NEW: external memory handler (LangChain)
        self.memory = None

    def attach_memory(self, langchain_handler):
        """Attach a LangChain handler for long-term memory"""
        self.memory = langchain_handler    
    
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
                        if response.text: 
                            print(f"{response.text}")
                             # Store Gemini’s reply into memory
                            if self.memory:
                                await self.memory.update_conversation("assistant", response.text)
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
