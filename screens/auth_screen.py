from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np
import asyncio
from utils.face_utils import get_embedding, mtcnn
from utils.db_connect import connect_db

class AuthScreen(Screen):
    mode = "login"  # 'login' or 'register'

    def on_enter(self):
        self.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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