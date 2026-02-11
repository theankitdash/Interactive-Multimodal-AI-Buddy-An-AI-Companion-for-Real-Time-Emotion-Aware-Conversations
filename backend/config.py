import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Server configuration
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000

# Face recognition settings
FACE_RECOGNITION_THRESHOLD = 0.75
FACE_SAMPLES_FOR_AUTH = 5

# Face registration
FACE_REGISTRATION_SAMPLES = 50  # Number of samples to capture during registration

# Gemini settings
GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GEMINI_FRAME_RATE = 1  # Send 1 frame per second to Gemini
