from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from db_connect import connect_db
import asyncio

import cv2
import numpy as np
import torch
import os
import mediapipe as mp
from facenet_pytorch import MTCNN, InceptionResnetV1

Window.clearcolor = (0.07, 0.07, 0.08, 1)  # dark background
Window.size = (1280, 720)

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
facenet = InceptionResnetV1(pretrained='vggface2').eval()


KV = """
RootWidget:
    AuthScreen:
    MainScreen:

<AuthScreen>:
    name: "auth"
    BoxLayout:
        orientation: "vertical"
        padding: "32dp"
        spacing: "20dp"

        # Camera preview
        Image:
            id: camera_preview
            size_hint_y: 0.6

        # Username input (only in registration)
        TextInput:
            id: username_input
            hint_text: "Username"
            size_hint_y: None
            height: "44dp"
            background_normal: ""
            background_color: (0.12,0.12,0.15,1)
            foreground_color: (1,1,1,1)
            padding: ["12dp","10dp"]
            radius: [12,]
            opacity: 1 if root.mode=='register' else 0
            disabled: False if root.mode=='register' else True

        # Full Name input (only in registration)
        TextInput:
            id: fullname_input
            hint_text: "Full Name"
            size_hint_y: None
            height: "44dp"
            background_normal: ""
            background_color: (0.12,0.12,0.15,1)
            foreground_color: (1,1,1,1)
            padding: ["12dp","10dp"]
            radius: [12,]
            opacity: 1 if root.mode=='register' else 0
            disabled: False if root.mode=='register' else True

        # Action button
        Button:
            id: action_button
            text: root.action_text
            size_hint_y: None
            height: "48dp"
            background_normal: ""
            background_color: (0.25,0.45,1,1)
            color: (1,1,1,1)
            font_size: "16sp"
            bold: True
            radius: [14,]
            on_release: root.do_action()

        # Toggle Login/Register
        Button:
            id: switch_button
            text: "Switch to Register"
            size_hint_y: None
            height: "44dp"
            background_normal: ""
            background_color: (0.18,0.18,0.22,1)
            color: (0.8,0.8,0.85,1)
            font_size: "14sp"
            radius: [12,]
            on_release: root.toggle_mode()

<MainScreen>:
    name: "main"
    BoxLayout:
        orientation: "horizontal"
        spacing: "20dp"
        padding: "24dp"

        # Left side - events with modern scrollbar
        ScrollView:
            id: scroll_events
            bar_width: 6
            scroll_type: ['bars', 'content']
            effect_cls: "ScrollEffect"
            bar_color: (0.3,0.5,1,0.8)  # soft blue scrollbar
            bar_inactive_color: (0.3,0.5,1,0.3)  # faded when not moving
            canvas.before:
                Color:
                    rgba: (0,0,0,0)
                Rectangle:
                    pos: self.pos
                    size: self.size

            RecycleView:
                id: events
                viewclass: "EventRow"
                bar_width: 0
                RecycleBoxLayout:
                    default_size: None, dp(54)
                    default_size_hint: 1, None
                    size_hint_y: None
                    height: self.minimum_height
                    orientation: "vertical"
                    spacing: "8dp"

        # Right side - camera
        Widget:
            id: camera_widget
            size_hint_y: None
            height: self.width
            canvas.before:
                Color:
                    rgba: (0.15, 0.15, 0.18, 1)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [20,]

<EventRow@BoxLayout>:
    size_hint_y: None
    height: "54dp"
    padding: "10dp"
    spacing: "16dp"
    canvas.before:
        Color:
            rgba: (0.12, 0.12, 0.15, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [14,]
    Label:
        text: root.date
        color: (0.7,0.7,0.7,1)
        font_size: "14sp"
    Label:
        text: root.title
        color: (1,1,1,1)
        font_size: "16sp"
        bold: True
"""
def get_embedding(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    if face is None: return None
    with torch.no_grad():
        return facenet(face.unsqueeze(0)).cpu().numpy()
        
class AuthScreen(Screen):
    mode = "login"  # 'login' or 'register'

    def on_enter(self):
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update_camera, 1/30)

    def on_leave(self):
        Clock.unschedule(self.update_camera)
        if hasattr(self, "capture"):
            self.capture.release()

    def update_camera(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, _ = mtcnn.detect(frame_rgb)

        # 🔹 Draw rectangle if a face is found
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(c) for c in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert to texture for Kivy Image
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.ids.camera_preview.texture = texture         

    @property
    def action_text(self):
        return "Login" if self.mode == "login" else "Register"

    def toggle_mode(self):
        if self.mode == "login":
            self.mode = "register"
        else:
            self.mode = "login"

        # Update input fields visibility
        is_register = self.mode == "register"

        self.ids.username_input.disabled = not is_register
        self.ids.username_input.opacity = 1 if is_register else 0

        self.ids.fullname_input.disabled = not is_register
        self.ids.fullname_input.opacity = 1 if is_register else 0

        # Update action button text
        self.ids.action_button.text = self.action_text

        self.ids.switch_button.text = "Switch to Register" if self.mode=="login" else "Switch to Login"

    def do_action(self):
        if self.mode == "login":
            asyncio.run(self.login_user())
        else:
            asyncio.run(self.register_user())
    
    def capture_face_embedding(self, num_samples=10):
        samples = []
        attempts = 0
        while len(samples) < num_samples and attempts < num_samples * 3:
            ret, frame = self.capture.read()
            if not ret:
                continue

            emb = get_embedding(frame)  
            if emb is not None:
                samples.append(emb)

        if samples:
            mean_emb = np.mean(np.vstack(samples), axis=0).tolist()
            return mean_emb
        return None       

    async def register_user(self):
        username = self.ids.username_input.text.strip()
        fullname = self.ids.fullname_input.text.strip()

        if not username or not fullname:
            print("❌ Missing fields")
            return
        
        face_embedding = self.capture_face_embedding()
        if face_embedding is None:
            print("❌ Face not captured, registration aborted")
            return

        conn = await connect_db()
        try:
            await conn.execute(
                """
                INSERT INTO user_details (username, name, face_embedding)
                VALUES ($1, $2, $3)
                ON CONFLICT (username) DO UPDATE SET name = $2, face_embedding = $3
                """,
                username, fullname, face_embedding
            )
            print(f"✅ Registered {username}")
            self.manager.current = "main"
        finally:
            await conn.close()

    async def login_user(self):
        conn = await connect_db()
        try:
            # Fetch all users and embeddings
            rows = await conn.fetch("SELECT username, face_embedding FROM user_details WHERE face_embedding IS NOT NULL")
            if not rows:
                print("❌ No registered users with face data.")
                return

            # Capture one frame
            ret, frame = self.capture.read()
            if not ret:
                print("❌ Could not access camera.")
                return

            # Get embedding
            emb = get_embedding(frame)
            if emb is None:
                print("❌ No face detected.")
                return

            # Cosine similarity
            def cos_sim(a, b):
                return float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))

            best_user, best_score = None, -1
            for row in rows:
                ref_emb = np.array(row["face_embedding"])
                score = cos_sim(emb, ref_emb)
                if score > best_score:
                    best_user, best_score = row["username"], score

            if best_score > 0.75:  # ✅ threshold
                print(f"✅ Recognized as {best_user} (similarity={best_score:.3f})")
                self.manager.current = "main"
            else:
                print("❌ Face not recognized.")
        finally:
            await conn.close()         

class MainScreen(Screen):
    def talk_with_agent(self, message):
        if not message.strip():
            return
        print("🤖 Agent received:", message)
        # Here you can call your LLM / agent logic
        # Example: store in DB under user_knowledge

class RootWidget(ScreenManager):
    pass

class MinimalUIApp(App):
    def build(self):
        return Builder.load_string(KV)

if __name__ == "__main__":
    MinimalUIApp().run()
