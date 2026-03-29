import os

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

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

# Local Mistral settings (base model from HuggingFace, improved over time via DPO)
# On first run, downloads from HuggingFace Hub. After fine-tuning, point to local checkpoint.
LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH", "mistralai/Mistral-7B-Instruct-v0.3")
MISTRAL_TEMPERATURE = 0.2
MISTRAL_TOP_P = 0.7
MISTRAL_MAX_TOKENS = 1024
MISTRAL_QUANTIZE_4BIT = True  # 4-bit quantization for ~6 GB VRAM
