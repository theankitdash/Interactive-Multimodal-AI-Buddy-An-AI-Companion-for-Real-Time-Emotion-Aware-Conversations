import asyncio
import base64
import logging
import time
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import google.genai as genai
from google.genai import types
from config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Initialize Gemini client for vision analysis
vision_client = genai.Client(api_key=GEMINI_API_KEY)

VISION_ANALYSIS_INTERVAL = 3.0  # Analyze every 3 seconds (balances responsiveness vs API cost)
VISION_MODEL = "gemini-2.5-flash"  # Text+vision model

VISION_PROMPT = """Describe what you see in this image in 1-2 concise sentences. 
Focus on: the person (appearance, expression, activity), their environment, and any notable objects.
Be factual and brief. Do not speculate beyond what is visible."""


class VisionAnalyzer: 
    def __init__(self):
        self._camera_on = False
        self._latest_frame: np.ndarray | None = None
        self._latest_description: str = "Camera is off. No visual data available."
        self._lock = asyncio.Lock()
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_analysis_time: float = 0
        
    @property
    def camera_on(self) -> bool:
        return self._camera_on
    
    @property
    def latest_description(self) -> str:
        return self._latest_description
    
    def set_camera_state(self, on: bool):
        self._camera_on = on
        if not on:
            self._latest_frame = None
            self._latest_description = "Camera is off. No visual data available."
            logger.info("[Vision] Camera turned OFF — description cleared")
        else:
            self._latest_description = "Camera just turned on. Waiting for first frame..."
            logger.info("[Vision] Camera turned ON — awaiting frames")
    
    def update_frame(self, frame: np.ndarray):
        if self._camera_on:
            self._latest_frame = frame.copy()
    
    def start(self):
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._analysis_loop())
            logger.info("[Vision] Analysis loop started")
    
    def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        logger.info("[Vision] Analysis loop stopped")
    
    async def _analysis_loop(self):
        while self._running:
            try:
                await asyncio.sleep(VISION_ANALYSIS_INTERVAL)
                
                if not self._camera_on or self._latest_frame is None:
                    continue
                
                # Analyze the current frame
                async with self._lock:
                    frame = self._latest_frame
                
                if frame is not None:
                    description = await self._analyze_frame(frame)
                    if description:
                        self._latest_description = description
                        logger.debug(f"[Vision] Scene: {description[:80]}...")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Vision] Analysis loop error: {e}")
                await asyncio.sleep(1)  # Back off on error
    
    async def _analyze_frame(self, frame: np.ndarray) -> str | None:
        try:
            # Encode frame to JPEG bytes
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG", quality=70)
            image_bytes = buffer.getvalue()
            
            # Call Gemini Vision
            response = await vision_client.aio.models.generate_content(
                model=VISION_MODEL,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {"text": VISION_PROMPT},
                            {"inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64.b64encode(image_bytes).decode("utf-8")
                            }}
                        ]
                    }
                ],
                config=types.GenerateContentConfig(
                    response_modalities=["TEXT"],
                    max_output_tokens=100
                )
            )
            
            # Extract text, skipping thought parts
            text = ""
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if getattr(part, 'thought', False):
                        continue
                    if part.text:
                        text += part.text
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            logger.error(f"[Vision] Frame analysis error: {e}")
            return None
    
    async def analyze_now(self, frame: np.ndarray) -> str:
        result = await self._analyze_frame(frame)
        if result:
            self._latest_description = result
        return self._latest_description
