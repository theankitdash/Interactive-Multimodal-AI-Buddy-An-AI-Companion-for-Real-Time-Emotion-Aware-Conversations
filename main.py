from kivy.app import App
from kivy.animation import Animation
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.properties import StringProperty, BooleanProperty
from kivy.clock import Clock
import os
import cv2
import numpy as np
import asyncio
import threading
import sounddevice as sd
from utils.face_utils import get_embedding, mtcnn
from utils.db_connect import connect_db
from ai.gemini_handler import GeminiHandler
from ai.langchain_handler import LangchainHandler

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class RootWidget(Screen):
    mode = StringProperty("login")  # login / register / assistant
    ai_state = StringProperty("listening")  # listening / thinking / speaking
    is_muted = BooleanProperty(False)
    is_camera_on = BooleanProperty(True)
    user_initials = StringProperty("")
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
        self.langchain_handler = None

        # Queue for capturing frames for face recognition
        self.face_queue = asyncio.Queue()

        # Start camera preview immediately
        Clock.schedule_interval(self.update_camera_widget, 1/30)
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # --- NEW helper for showing temporary status ---
    def show_status(self, message, duration=5, fade=True):
        self.ids.status_label.text = message
        self.ids.status_label.opacity = 1  # make sure visible

        # Schedule removal
        def clear_text(dt):
            if fade:
                anim = Animation(opacity=0, duration=1)
                anim.bind(on_complete=lambda *args: setattr(self.ids.status_label, 'text', ""))
                anim.start(self.ids.status_label)
            else:
                self.ids.status_label.text = ""

        Clock.schedule_once(clear_text, duration)
    
    # --- UI State Management ---
    def get_initials(self, name):
        """Extract 2-letter initials from user name"""
        if not name:
            return "??"
        words = name.strip().split()
        if len(words) >= 2:
            return (words[0][0] + words[-1][0]).upper()
        elif len(words) == 1:
            return words[0][:2].upper()
        return "??"
    
    def get_state_color(self):
        """Return RGBA color based on AI state"""
        colors = {
            "listening": (0.388, 0.4, 0.945, 1),    # #6366f1 blue
            "thinking": (0.961, 0.62, 0.043, 1),    # #f59e0b amber
            "speaking": (0.063, 0.725, 0.506, 1),   # #10b981 green
        }
        return colors.get(self.ai_state, colors["listening"])
    
    def cycle_ai_state(self):
        """Cycle through AI states for demo/testing"""
        states = ["listening", "thinking", "speaking"]
        current_index = states.index(self.ai_state)
        self.ai_state = states[(current_index + 1) % len(states)]
        self.start_pulse_animation()
    
    def toggle_mute(self):
        """Toggle audio mute state"""
        self.is_muted = not self.is_muted
        # TODO: Connect to actual audio stream muting
        print(f"Audio muted: {self.is_muted}")
    
    def toggle_camera(self):
        """Toggle camera on/off"""
        self.is_camera_on = not self.is_camera_on
        if not self.is_camera_on and self.cap:
            # Stop camera
            self.cap.release()
            self.cap = None
        elif self.is_camera_on and not self.cap:
            # Restart camera
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    def handle_logout(self):
        """Handle logout button click"""
        self.stop_assistant()
        self.current_user = None
        self.user_initials = ""
        self.mode = "login"
        self.ai_state = "listening"
        self.is_muted = False
        if not self.is_camera_on:
            self.is_camera_on = True
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    def start_pulse_animation(self):
        """Start pulse animation on AI state circles"""
        if self.mode != "assistant":
            return
        
        # Get the circle widgets
        try:
            glow = self.ids.ai_glow_outer
            main_circle = self.ids.ai_circle_main
            inner_circle = self.ids.ai_circle_inner
            
            # Create pulsing animation
            anim_out = Animation(size=("140dp", "140dp"), duration=1) + Animation(size=("128dp", "128dp"), duration=1)
            anim_out.repeat = True
            anim_out.start(glow)
            
            anim_main = Animation(opacity=0.9, duration=1) + Animation(opacity=0.8, duration=1)
            anim_main.repeat = True
            anim_main.start(main_circle)
        except:
            pass  # Widgets might not be ready yet    

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
            self.show_status("Missing fields")
            return

        face_embedding = await self.capture_face_embedding()
        if not face_embedding:
            self.show_status("Could not capture face. Try again.")
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
        self.user_initials = self.get_initials(fullname)
        self.mode = "assistant"
        self.show_status(f"Registered {username}, ready!")
        self.langchain_handler = LangchainHandler(username=self.current_user, NVIDIA_API_KEY=NVIDIA_API_KEY)
        Clock.schedule_once(lambda dt: self.start_pulse_animation(), 0.5)
        self.start_assistant()

    async def login_user(self):
        conn = await connect_db()
        try:
            rows = await conn.fetch("SELECT name, face_embedding FROM user_details WHERE face_embedding IS NOT NULL")
            if not rows:
                self.show_status("No registered users")
                return

            face_embedding = await self.capture_face_embedding()
            if not face_embedding:
                self.show_status("No face detected. Try again.")
                return

            # Cosine similarity
            def cos_sim(a, b):
                return float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))

            best_name, best_score = None, -1
            for row in rows:
                ref_emb = np.array(row["face_embedding"])
                score = cos_sim(face_embedding, ref_emb)
                if score > best_score:
                    best_name, best_score = row["name"], score

            if best_score > 0.75:
                self.current_user = best_name
                self.user_initials = self.get_initials(best_name)
                self.mode = "assistant"
                self.show_status(f"Welcome back {best_name}!")
                self.langchain_handler = LangchainHandler(username=self.current_user, NVIDIA_API_KEY=NVIDIA_API_KEY)
                Clock.schedule_once(lambda dt: self.start_pulse_animation(), 0.5)
                self.start_assistant()
            else:
                self.show_status("Face not recognized! Try again.")

        finally:
            await conn.close()

    # ---------------- GEMINI ----------------
    def start_assistant(self):
        """Start Gemini agent after login/registration"""
        if self.handler:
            return  # already running

        if not GEMINI_API_KEY:
            self.show_status("Missing GEMINI_API_KEY!")
            return

        self.handler = GeminiHandler(api_key=GEMINI_API_KEY)
        self.handler.attach_memory(self.langchain_handler)

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
                # Full screen background preview
                frame_bg = cv2.flip(frame, 0)
                buf_bg = frame_bg.tobytes()
                texture_bg = Texture.create(size=(frame_bg.shape[1], frame_bg.shape[0]), colorfmt='bgr')
                texture_bg.blit_buffer(buf_bg, colorfmt='bgr', bufferfmt='ubyte')
                self.ids.camera_preview.texture = texture_bg
                
                # Small preview in corner (bottom-right)
                if self.mode == "assistant" and self.is_camera_on:
                    try:
                        frame_small = cv2.flip(frame, 0)
                        buf_small = frame_small.tobytes()
                        texture_small = Texture.create(size=(frame_small.shape[1], frame_small.shape[0]), colorfmt='bgr')
                        texture_small.blit_buffer(buf_small, colorfmt='bgr', bufferfmt='ubyte')
                        self.ids.camera_preview_small.texture = texture_small
                    except:
                        pass  # Widget might not be ready yet

    def stop_assistant(self):
        """Stop Gemini & release resources"""

        if self.langchain_handler:
            asyncio.run_coroutine_threadsafe(self.langchain_handler.stop(), self.loop)
            self.langchain_handler = None  

        if self.handler and self.loop:
            asyncio.run_coroutine_threadsafe(self.handler.stop(), self.loop)
            self.handler = None

            # stop event loop safely
            if self.loop.is_running():
                self.loop.call_soon_threadsafe(self.loop.stop)
            self.loop = None

        if self.mic_stream:
            self.mic_stream.stop()
            self.mic_stream.close()
            self.mic_stream = None

        if self.speaker:
            self.speaker.stop()
            self.speaker.close()
            self.speaker = None

        if self.cap:
            self.cap.release()
            self.cap = None

        Clock.unschedule(self.update_camera_widget)


class MultiModalBuddy(App):
    def build(self):
        Builder.load_file("ui/kv_layout.kv")
        return RootWidget()

    def on_stop(self):
        if self.root:
            self.root.stop_assistant()    

if __name__ == "__main__":
    MultiModalBuddy().run()
