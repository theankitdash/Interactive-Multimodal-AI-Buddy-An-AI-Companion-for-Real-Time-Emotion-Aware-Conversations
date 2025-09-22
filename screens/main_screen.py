import os
import re
import cv2
import time
import base64
import asyncio
import threading
import numpy as np
import sounddevice as sd
from io import BytesIO
from PIL import Image
from utils.db_connect import connect_db
from datetime import datetime

import google.genai as genai
from kivy.uix.screenmanager import Screen
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
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
    def __init__(self, api_key: str, system_prompt: str = None):
        self.client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        self.session = None
        self.audio_queue = asyncio.Queue()
        self.video_queue = asyncio.Queue()
        self.quit = asyncio.Event()
        self.last_frame_time = 0
        self.last_text = None
        self.system_prompt = system_prompt

    async def start(self):
        config = {"response_modalities": ["AUDIO"]}

        if self.system_prompt:
            config["system_instruction"] = self.system_prompt

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
                            self.last_text = response.text
                            print("Agent text event:", response.text)
                except Exception as e:
                    print("Gemini error (in receive loop):", e)
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
            try:
                await self.session.close()
            except Exception as e:
                print("Error closing session:", e)
            finally:
                self.quit.clear()
                self.session = None

# ------------------ Event Row ------------------
class EventRow(BoxLayout):
    date = StringProperty("")   
    title = StringProperty("")      

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

        self.username = None
        self.full_name = None
        self.events = []
        self.knowledge = []

    async def load_user_data(self):
        conn = await connect_db()
        try:
            # --- Load full name ---
            row = await conn.fetchrow(
                "SELECT name FROM user_details WHERE username=$1",
                self.username
            )
            if row:
                self.full_name = row["name"]

            # --- Load events ---
            rows = await conn.fetch(
                "SELECT event_id, type, description, event_time, priority, status "
                "FROM events WHERE username=$1 ORDER BY event_time ASC",
                self.username
            )
            self.events = [dict(r) for r in rows]

            # --- Load knowledge ---
            rows = await conn.fetch(
                "SELECT fact, category, importance FROM user_knowledge "
                "WHERE username=$1 ORDER BY importance DESC",
                self.username
            )
            self.knowledge = [dict(r) for r in rows]

            print(f"✅ Loaded profile for {self.full_name}")
            print(f"Events: {len(self.events)}, Knowledge: {len(self.knowledge)}")

        finally:
            await conn.close()

    # ------------------ Add new event ------------------
    async def add_event(self, description, event_time, type="other", priority=3):
        conn = await connect_db()
        try:
            await conn.execute(
                """
                INSERT INTO events (username, type, description, event_time, priority)
                VALUES ($1, $2, $3, $4, $5)
                """,
                self.username, type, description, event_time, priority
            )
            print(f"✅ Event added: {description} at {event_time}")
        finally:
            await conn.close()

        self.events.append({
            "description": description,
            "event_time": event_time,
            "type": type,
            "priority": priority,
            "status": "pending"
        })
        self.ids.events.data = [
            {"date": str(e["event_time"].date()), "title": e["description"]}
            for e in self.events
        ]

    # ------------------ Add new knowledge ------------------
    async def add_knowledge(self, fact, category="other", importance=3):
        conn = await connect_db()
        try:
            await conn.execute(
                """
                INSERT INTO user_knowledge (username, fact, category, importance)
                VALUES ($1, $2, $3, $4)
                """,
                self.username, fact, category, importance
            )
            print(f"✅ Knowledge added: {fact}")
        finally:
            await conn.close()

        self.knowledge.insert(0, {"fact": fact, "category": category, "importance": importance})

    def build_context(self, user_input: str) -> str:
        """Dynamically build context with DB knowledge + events for Gemini."""
        context = f"You are talking to {self.full_name}. "
        
        if self.knowledge:
            top_facts = "; ".join([f"{k['fact']} (category: {k['category']})" for k in self.knowledge[:5]])
            context += f"Some facts about them: {top_facts}. "

        upcoming = [e for e in self.events if e["event_time"] > datetime.now()]
        if upcoming:
            upcoming_str = "; ".join(
                [f"{e['description']} at {e['event_time'].strftime('%H:%M')}" for e in upcoming[:5]]
            )
            context += f"Upcoming events: {upcoming_str}. "

        if user_input:
            context += f"User said: {user_input}"
        else:
            context += "Session priming message."

        return context  

    # ------------------ Handle Gemini commands ------------------
    async def handle_gemini_command(self, text):
        text = text.lower()
        if "remind me to" in text:
            m = re.search(r"remind me to (.+) at (\d{1,2}:\d{2})", text)
            if m:
                desc = m.group(1)
                time_str = m.group(2)
                now = datetime.now()
                event_time = datetime.strptime(f"{now.date()} {time_str}", "%Y-%m-%d %H:%M")
                await self.add_event(desc, event_time, type="reminder")
        elif "remember that" in text:
            fact = text.split("remember that", 1)[1].strip()
            if fact:
                await self.add_knowledge(fact)       

    def on_pre_enter(self):
        """Start streaming when entering main screen"""

        self.username = self.manager.current_user
        # Create event loop + background thread once (use this loop everywhere)
        if not self.loop:
            self.loop = asyncio.new_event_loop()
            def _run_loop(loop):
                asyncio.set_event_loop(loop)
                loop.run_forever()
            self.loop_thread = threading.Thread(target=_run_loop, args=(self.loop,), daemon=True)
            self.loop_thread.start()

        future = asyncio.run_coroutine_threadsafe(self.load_user_data(), self.loop)
        future.result() 

        # Update events in Kivy UI
        self.ids.events.data = [
            {"date": str(e["event_time"].date()), "title": e["description"]}
            for e in self.events
        ]

        system_prompt = f"You are a personalized assistant for {self.full_name}."
        self.handler = GeminiHandler(api_key=os.getenv("GEMINI_API_KEY"), system_prompt=system_prompt)

        # Start the agent on the background loop (do NOT create another loop)
        asyncio.run_coroutine_threadsafe(self.run_agent(), self.loop)

        # Start camera preview updater
        Clock.schedule_interval(self.update_camera_widget, 1/30)

    # ------------------ Start Gemini in background ------------------
    async def run_agent(self):
        # start the handler (this opens/blocks on the live session in handler.start())
        # run it as a background task inside the same loop
        handler_task = asyncio.create_task(self.handler.start())

        # wait until session is available (handler.start sets session when connected)
        max_wait = 10.0
        waited = 0.0
        while not self.handler.session and waited < max_wait:
            await asyncio.sleep(0.1)
            waited += 0.1

        if not self.handler.session:
            print("Failed to establish Gemini session within timeout.")
            return

        # Send initial context primer so model is grounded
        primer = self.build_context("")  # empty user_input for initial primer
        try:
            await self.handler.session.send(input=primer)
        except Exception as e:
            print("Error sending primer to session:", e)

        # Start camera/mic/speaker
        self.cap = cv2.VideoCapture(0)
        self.mic_stream = sd.InputStream(channels=1, samplerate=16000, blocksize=512, callback=self.mic_callback)
        self.mic_stream.start()
        self.speaker = sd.OutputStream(channels=1, samplerate=24000, blocksize=512)
        self.speaker.start()

        # Main loop: read audio replies, detect model text events to handle commands
        try:
            while not self.handler.quit.is_set():
                # Send video frames periodically
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        # optionally send frames (throttled in handler)
                        asyncio.create_task(self.handler.send_video(frame))

                # Play audio replies from Gemini
                reply = await self.handler.get_audio_reply()
                if reply:
                    _, audio_np = reply
                    audio_np = audio_np.astype(np.float32) / 32767.0
                    if audio_np.ndim == 1:
                        audio_np = audio_np[:, np.newaxis]
                    elif audio_np.ndim == 2 and audio_np.shape[0] == 1:
                        audio_np = audio_np.T
                    elif audio_np.ndim == 2 and audio_np.shape[1] > 1:
                        audio_np = audio_np[:, 0:1]
                    # write may block, consider try/except in production
                    self.speaker.write(audio_np)

                # If model produced textual event(s), react to them (e.g., add reminders/knowledge).
                # IMPORTANT: we do NOT send the model's own text back into the session (that causes loops).
                if self.handler.last_text:
                    # make a local copy then clear to avoid re-processing same text
                    text_event = self.handler.last_text
                    self.handler.last_text = None
                    # handle DB-affecting commands like "remember that" / "remind me to"
                    await self.handle_gemini_command(text_event)

                await asyncio.sleep(0.01)
        finally:
            # cleanup
            try:
                await self.handler.stop()
            except Exception:
                pass

    def mic_callback(self, indata, frames, time_, status):
        if status:
            print("Mic:", status)
        array = (indata * 32767).astype(np.int16)
        if self.loop and self.handler:
            try:
                asyncio.run_coroutine_threadsafe(self.handler.send_audio(array), self.loop)
            except Exception as e:
                print("Failed to submit audio send:", e)

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
        if self.cap: 
            self.cap.release()
        if self.mic_stream: 
            self.mic_stream.stop()
        if self.speaker: 
            self.speaker.stop()
        Clock.unschedule(self.update_camera_widget)
