import os

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Server configuration
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000

# Face recognition settings
FACE_RECOGNITION_THRESHOLD = 0.75
FACE_SAMPLES_FOR_AUTH = 5

# Face registration
FACE_REGISTRATION_SAMPLES = 50 

# Gemini settings (live audio streaming only)
GEMINI_MODEL = "gemini-2.5-flash-native-audio-preview-12-2025"
GEMINI_FRAME_RATE = 1 

# NVIDIA Mistral settings (text reasoning & generation)
NVIDIA_MODEL = "mistralai/mistral-7b-instruct-v0.3"
NVIDIA_TEMPERATURE = 0.2
NVIDIA_TOP_P = 0.7
NVIDIA_MAX_TOKENS = 1024
