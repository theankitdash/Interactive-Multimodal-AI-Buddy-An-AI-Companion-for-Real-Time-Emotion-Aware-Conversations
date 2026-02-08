import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Server configuration
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000

# Face recognition settings
FACE_RECOGNITION_THRESHOLD = 0.75
FACE_SAMPLES_FOR_AUTH = 5

# Liveness detection settings
LIVENESS_ENABLED = False
LIVENESS_TIMEOUT = 15  # seconds per challenge
LIVENESS_BLINK_THRESHOLD = 0.22  # EAR threshold
LIVENESS_HEAD_MOVEMENT_THRESHOLD = 40  # pixels
LIVENESS_SMILE_WIDTH_RATIO = 0.45
LIVENESS_SMILE_HEIGHT_RATIO = 0.03

# Face registration
FACE_REGISTRATION_SAMPLES = 50  # Number of samples to capture during registration

# Gemini settings
GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GEMINI_FRAME_RATE = 1  # Send 1 frame per second to Gemini
