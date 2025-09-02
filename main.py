import os
import time
import base64
import asyncio
import numpy as np
from io import BytesIO
from backend.routers import auth, user_profile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastrtc import Stream, AsyncAudioVideoStreamHandler, wait_for_item
import google.genai as genai
from PIL import Image

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
app.include_router(auth.router)
app.include_router(user_profile.router)

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helpers ----------
def encode_audio(data: np.ndarray) -> dict:
    return {
        "mime_type": "audio/pcm",
        "data": base64.b64encode(data.tobytes()).decode("UTF-8"),
    }

def encode_image(frame: np.ndarray) -> dict:
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(frame)
        pil_image.save(output_bytes, "JPEG")
        bytes_data = output_bytes.getvalue()
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(bytes_data).decode("utf-8")
    }

# ---------- Gemini Handler ----------
class GeminiHandler(AsyncAudioVideoStreamHandler):
    def __init__(self) -> None:
        super().__init__("mono", output_sample_rate=24000, input_sample_rate=16000)
        self.audio_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()
        self.session = None
        self.last_frame_time = 0
        self.quit = asyncio.Event()

    def copy(self):
        return GeminiHandler()

    async def start_up(self):
        client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_options={"api_version": "v1alpha"}
        )
        config = {"response_modalities": ["AUDIO"]}
        async with client.aio.live.connect(
            model="gemini-2.0-flash-exp",
            config=config
        ) as session:
            self.session = session
            while not self.quit.is_set():
                turn = self.session.receive()
                try:
                    async for response in turn:
                        if data := response.data:
                            audio = np.frombuffer(data, dtype=np.int16).reshape(1, -1)
                            self.audio_queue.put_nowait(audio)
                except Exception:
                    break

    async def video_receive(self, frame: np.ndarray):
        """Receive video frames from client and send to Gemini every 1s"""
        self.video_queue.put_nowait(frame)
        if self.session and (time.time() - self.last_frame_time) > 1:
            self.last_frame_time = time.time()
            await self.session.send(input=encode_image(frame))

    async def video_emit(self):
        """Emit last received frame or placeholder"""
        frame = await wait_for_item(self.video_queue, 0.01)
        if frame is not None:
            return frame
        else:
            return np.zeros((100, 100, 3), dtype=np.uint8)

    async def receive(self, frame: tuple[int, np.ndarray]) -> None:
        """Receive audio from client and send to Gemini"""
        _, array = frame
        array = array.squeeze()
        if self.session:
            await self.session.send(input=encode_audio(array))

    async def emit(self):
        """Emit audio from Gemini to client"""
        array = await wait_for_item(self.audio_queue, 0.01)
        if array is not None:
            return (self.output_sample_rate, array)
        return array

    async def shutdown(self) -> None:
        if self.session:
            self.quit.set()
            await self.session.close()
            self.quit.clear()

# ---------- Stream ----------
stream = Stream(
    handler=GeminiHandler(),
    modality="audio-video",
    mode="send-receive"
)
stream.mount(app)
    
# Run the application using uvicorn main:app --reload