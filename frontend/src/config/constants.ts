// Centralized configuration constants

export const BACKEND_URL = 'http://127.0.0.1:8000';
export const BACKEND_WS_URL = 'ws://127.0.0.1:8000';

// Health check configuration
export const HEALTH_CHECK_ENDPOINT = `${BACKEND_URL}/health`;
export const HEALTH_CHECK_TIMEOUT = 30000; // 30 seconds
export const HEALTH_CHECK_INTERVAL = 1000; // Check every second

// Audio configuration
export const AUDIO_SAMPLE_RATE = 16000;
export const AUDIO_CHANNEL_COUNT = 1;
export const AUDIO_BUFFER_SIZE = 512;

// Video configuration
export const VIDEO_WIDTH = 1280;
export const VIDEO_HEIGHT = 720;
export const VIDEO_FRAME_INTERVAL = 1000; // 1 frame per second
export const IMAGE_QUALITY = 0.8; // JPEG quality

// Face capture configuration
export const FACE_CAPTURE_COUNT = 5;
export const FACE_CAPTURE_DELAY = 500; // ms between captures
