from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty
from kivy.clock import Clock
import os
import cv2
import numpy as np
import asyncio
import threading
import sounddevice as sd
from utils.face_utils import get_embedding, mtcnn
from utils.db_connect import connect_db
from ai.gemini import GeminiHandler

class RootWidget(Screen):
    mode = StringProperty("login")  # login / register / assistant
    current_user = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # AI & camera
        self.handler = None
        self.loop = None
        self.thread = None
        self.cap = None
        self.mic_stream = None
        self.speaker = None

        # Queue for capturing frames for face recognition
        self.face_queue = asyncio.Queue()

        # Start camera preview immediately
        Clock.schedule_interval(self.update_camera_widget, 1/30)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # ---------------- AUTH ----------------
    def toggle_mode(self):
        self.mode = "register" if self.mode=="login" else "login"
        self.ids.switch_button.text = "Switch to Register" if self.mode=="login" else "Switch to Login"
        self.ids.action_button.text = "Login" if self.mode=="login" else "Register"

    def do_action(self):
        if self.mode == "login":
            asyncio.run(self.login_user())
        elif self.mode == "register":
            asyncio.run(self.register_user())

    async def capture_face_embedding(self, num_samples=5, timeout_per_frame=0.5):
        """Capture multiple frames for reliable face embedding"""
        samples = []
        attempts = 0
        max_attempts = num_samples * 5

        while len(samples) < num_samples and attempts < max_attempts:
            ret, frame = self.cap.read()
            if not ret:
                attempts += 1
                await asyncio.sleep(0.05)
                continue

            emb = get_embedding(frame)
            if emb is not None:
                samples.append(emb)

            attempts += 1
            await asyncio.sleep(0.05)

        if samples:
            return np.mean(np.vstack(samples), axis=0).tolist()
        return None

    async def register_user(self):
        username = self.ids.username_input.text.strip()
        fullname = self.ids.fullname_input.text.strip()
        if not username or not fullname:
            self.ids.status_label.text = "❌ Missing fields"
            return

        face_embedding = await self.capture_face_embedding()
        if not face_embedding:
            self.ids.status_label.text = "❌ Could not capture face. Try again."
            return

        conn = await connect_db()
        try:
            await conn.execute(
                """INSERT INTO user_details (username, name, face_embedding)
                   VALUES ($1,$2,$3)
                   ON CONFLICT (username) DO UPDATE SET name=$2, face_embedding=$3""",
                username, fullname, face_embedding
            )
        finally:
            await conn.close()

        self.current_user = username
        self.mode = "assistant"
        self.ids.status_label.text = f"✅ Registered {username}, ready!"
        self.start_assistant()

    async def login_user(self):
        conn = await connect_db()
        try:
            rows = await conn.fetch("SELECT username, face_embedding FROM user_details WHERE face_embedding IS NOT NULL")
            if not rows:
                self.ids.status_label.text = "❌ No registered users"
                return

            face_embedding = await self.capture_face_embedding()
            if not face_embedding:
                self.ids.status_label.text = "❌ No face detected. Try again."
                return

            # Cosine similarity
            def cos_sim(a, b):
                return float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))

            best_user, best_score = None, -1
            for row in rows:
                ref_emb = np.array(row["face_embedding"])
                score = cos_sim(face_embedding, ref_emb)
                if score > best_score:
                    best_user, best_score = row["username"], score

            if best_score > 0.75:
                self.current_user = best_user
                self.mode = "assistant"
                self.ids.status_label.text = f"✅ Welcome back {best_user}!"
                self.start_assistant()
            else:
                self.ids.status_label.text = "❌ Face not recognized"

        finally:
            await conn.close()

    # ---------------- GEMINI ----------------
    def start_assistant(self):
        """Start Gemini agent after login/registration"""
        if self.handler:
            return  # already running

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            self.ids.status_label.text = "❌ Missing GEMINI_API_KEY"
            return

        self.handler = GeminiHandler(api_key=api_key, username=self.current_user)

        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run_agent())

        self.thread = threading.Thread(target=run_loop, daemon=True)
        self.thread.start()

    async def run_agent(self):
        asyncio.create_task(self.handler.start())

        # Microphone → Gemini
        def mic_callback(indata, frames, time_, status):
            if status:
                print("Mic:", status)
            array = (indata * 32767).astype(np.int16)
            asyncio.run_coroutine_threadsafe(self.handler.send_audio(array), self.loop)

        self.mic_stream = sd.InputStream(channels=1, samplerate=16000, blocksize=512, callback=mic_callback)
        self.mic_stream.start()

        # Speaker (Gemini → User)
        self.speaker = sd.OutputStream(channels=1, samplerate=24000, blocksize=512)
        self.speaker.start()

        try:
            while not self.handler.quit.is_set():
                ret, frame = self.cap.read()
                if ret:
                    await self.handler.send_video(frame)

                reply = await self.handler.get_audio_reply()
                if reply:
                    _, audio_np = reply
                    audio_np = audio_np.astype(np.float32) / 32767.0

                    # Ensure mono (N,1)
                    if audio_np.ndim == 1:
                        audio_np = audio_np[:, np.newaxis]
                    elif audio_np.ndim == 2 and audio_np.shape[0] == 1:
                        audio_np = audio_np.T
                    elif audio_np.ndim == 2 and audio_np.shape[1] > 1:
                        audio_np = audio_np[:, 0:1]

                    self.speaker.write(audio_np)

                await asyncio.sleep(0.01)
        finally:
            self.stop_assistant()

    def update_camera_widget(self, dt):
        """Update camera preview in KV"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 0)
                buf = frame.tobytes()
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
                self.ids.camera_preview.texture = texture

    def stop_assistant(self):
        """Stop Gemini & release resources"""
        if self.handler:
            asyncio.run_coroutine_threadsafe(self.handler.stop(), self.loop)
            self.handler = None

        if self.mic_stream: self.mic_stream.stop()
        if self.speaker: self.speaker.stop()

        Clock.unschedule(self.update_camera_widget)
