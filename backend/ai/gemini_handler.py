"""
Gemini Live Audio Handler

Handles real-time audio/video streaming with Gemini's Live API.
Based on the fastrtc reference pattern for bidirectional streaming.
"""
import cv2
import time
import base64
import asyncio
import numpy as np
import logging
from io import BytesIO
from PIL import Image
import google.genai as genai
from google.genai import types

from google.genai.types import (
    LiveConnectConfig,
    PrebuiltVoiceConfig,
    SpeechConfig,
    VoiceConfig,
)
from config import GEMINI_MODEL

logger = logging.getLogger(__name__)

# Constants
INPUT_SAMPLE_RATE = 16000   # Input audio from client
OUTPUT_SAMPLE_RATE = 24000  # Gemini's native output sample rate


def encode_audio(data: np.ndarray) -> str:
    """Encode audio data to base64 string for streaming to Gemini."""
    if data.dtype != np.int16:
        data = np.clip(data, -32768, 32767).astype(np.int16)
    return base64.b64encode(data.tobytes()).decode("UTF-8")


def encode_image(frame: np.ndarray) -> dict:
    """Encode video frame for Gemini input."""
    with BytesIO() as output_bytes:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        pil_image.save(output_bytes, "JPEG")
        bytes_data = output_bytes.getvalue()
    return {
        "mime_type": "image/jpeg",
        "data": base64.b64encode(bytes_data).decode("utf-8"),
    }


class GeminiHandler:
    """
    Handler for Gemini Live API with bidirectional audio streaming.
    
    Follows the pattern from fastrtc reference implementation:
    - Audio input via async generator (base64 encoded strings)
    - Audio output as numpy arrays at 24kHz
    """
    
    def __init__(self, api_key: str, voice_name: str = "Puck"):
        self.api_key = api_key
        self.voice_name = voice_name
        self.client = genai.Client(
            api_key=api_key,
            http_options={"api_version": "v1alpha"}
        )
        
        self.session = None
        self.input_queue: asyncio.Queue = asyncio.Queue()
        self.output_queue: asyncio.Queue = asyncio.Queue()
        self.transcription_queue: asyncio.Queue = asyncio.Queue()  # User input transcriptions
        self.event_queue: asyncio.Queue = asyncio.Queue()  # Events for Cognition Socket
        self.quit: asyncio.Event = asyncio.Event()
        self.session_ready: asyncio.Event = asyncio.Event()
        self.last_frame_time = 0
        self._is_model_speaking = False  # Track if model is currently speaking
    
    def copy(self):
        """Create a copy of this handler (for handler instantiation pattern)."""
        return GeminiHandler(
            api_key=self.api_key,
            voice_name=self.voice_name
        )
    
    async def start(self, system_instruction: str = None):
        """
        Start the Gemini Live session and begin audio streaming.
        
        NOTE: System instruction is intentionally minimal/None for codec-only mode.
        Intelligence/personality handled by Cognition Socket (Mistral).
        
        Args:
            system_instruction: Optional system prompt (ignored in codec mode)
        """
        try:
            # CODEC MODE: No system instruction, personality-neutral
            # All intelligence comes from Cognition Socket (Mistral)
            config = LiveConnectConfig(
                response_modalities=["AUDIO"],  # Audio-only responses
                speech_config=SpeechConfig(
                    voice_config=VoiceConfig(
                        prebuilt_voice_config=PrebuiltVoiceConfig(
                            voice_name=self.voice_name,
                        )
                    )
                ),
                system_instruction=types.Content(parts=[{"text": system_instruction}]) if system_instruction else None
            )
            
            logger.info(f"[Gemini] Starting live session with voice: {self.voice_name}")
            
            async with self.client.aio.live.connect(
                model=GEMINI_MODEL, 
                config=config
            ) as session:
                self.session = session
                self.session_ready.set()
                logger.info("[Gemini] Session connected successfully")
                
                try:
                    # Use start_stream for bidirectional audio
                    # The stream() method yields base64-encoded audio strings
                    async for response in session.start_stream(
                        stream=self._audio_input_stream(),
                        mime_type="audio/pcm"
                    ):
                        if self.quit.is_set():
                            break
                        
                        # Manually extract audio data from parts to avoid SDK warning
                        # The warning occurs when using response.data with mixed content types
                        
                        # Check for user input transcription (input_transcription)
                        # This is what the USER said, not what the model said
                        if hasattr(response, 'server_content') and response.server_content:
                            # Check for user input transcription
                            if hasattr(response.server_content, 'input_transcription') and response.server_content.input_transcription:
                                user_text = response.server_content.input_transcription.strip()
                                if user_text:
                                    logger.info(f"[Gemini] User said: {user_text[:100]}")
                                    await self.transcription_queue.put(user_text)
                        
                        # Process model's audio response
                        if response.server_content and response.server_content.model_turn:
                            self._is_model_speaking = True
                            for part in response.server_content.model_turn.parts:
                                # Skip thought parts entirely
                                if getattr(part, 'thought', False):
                                    continue
                                
                                # Extract audio data from inline_data
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                        try:
                                            audio_array = np.frombuffer(part.inline_data.data, dtype=np.int16)
                                            await self.output_queue.put((OUTPUT_SAMPLE_RATE, audio_array))
                                        except Exception as e:
                                            logger.error(f"[Gemini] Audio decode error: {e}")
                                
                                # Log model's text output but DO NOT put in transcription queue
                                # The model's response should not trigger re-processing
                                elif hasattr(part, 'text') and part.text:
                                    text = part.text.strip()
                                    if text:
                                        logger.debug(f"[Gemini] Model said (not queued): {text[:100]}")
                        
                        # Check for turn completion to reset speaking flag
                        if hasattr(response.server_content, 'turn_complete') and response.server_content.turn_complete:
                            self._is_model_speaking = False
                            # Emit end-of-utterance event for Cognition Socket
                            await self.event_queue.put({
                                "event": "end_of_utterance",
                                "timestamp": time.time()
                            })
                
                except asyncio.CancelledError:
                    logger.info("[Gemini] Stream cancelled")
                except Exception as e:
                    logger.error(f"[Gemini] Streaming error: {e}")
        
        except Exception as e:
            logger.error(f"[Gemini] Session start error: {e}")
        finally:
            self.session = None
            self.session_ready.clear()
    
    async def _audio_input_stream(self):
        """
        Async generator that yields base64-encoded audio chunks.
        This is consumed by Gemini's start_stream method.
        """
        while not self.quit.is_set():
            try:
                # Wait for audio chunk with timeout
                audio_chunk = await asyncio.wait_for(
                    self.input_queue.get(), 
                    timeout=0.1
                )
                # Yield the base64 encoded string
                yield audio_chunk
                
            except (asyncio.TimeoutError, TimeoutError):
                # No audio available, continue waiting
                pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[Gemini] Input stream error: {e}")
    
    async def send_audio(self, array: np.ndarray):
        """
        Queue audio data for sending to Gemini.
        
        Args:
            array: Audio data as numpy array (int16)
        """
        try:
            if not self.session_ready.is_set():
                # Wait briefly for session to be ready
                if self.quit.is_set():
                    return
                await asyncio.sleep(0.1)
                if not self.session_ready.is_set():
                    return
            
            # Convert to base64 and queue
            audio_b64 = encode_audio(array)
            await self.input_queue.put(audio_b64)
            
        except Exception as e:
            logger.error(f"[Gemini] Send audio error: {e}")
    
    async def send_video(self, frame: np.ndarray):
        """
        Send a video frame to Gemini for visual context.
        Rate-limited to avoid overwhelming the API.
        
        Args:
            frame: Video frame as numpy array (BGR format)
        """
        try:
            # Rate limit: 1 frame per second
            current_time = time.time()
            if not self.session or not self.session_ready.is_set():
                logger.debug("[Gemini] Cannot send video: session not ready")
                return
            if (current_time - self.last_frame_time) < 1:
                logger.debug("[Gemini] Video frame skipped (rate limit)")
                return
            
            self.last_frame_time = current_time
            
            # Encode frame
            encoded_frame = encode_image(frame)
            frame_size = len(encoded_frame.get('data', ''))
            logger.info(f"[Gemini] Sending video frame: {frame.shape}, encoded size: {frame_size} bytes")
            
            # Send to Gemini
            await self.session.send(input=encoded_frame)
            logger.info("[Gemini] ✓ Video frame sent successfully")
            
        except Exception as e:
            logger.error(f"[Gemini] ✗ Send video error: {e}", exc_info=True)
    
    async def get_audio_reply(self):
        """
        Get an audio reply from Gemini if available.
        
        Returns:
            Tuple of (sample_rate, audio_array) or None
        """
        try:
            if self.output_queue.empty():
                return None
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        except Exception as e:
            logger.error(f"[Gemini] Get reply error: {e}")
            return None
    
    async def get_transcription(self):
        """
        Get captured transcription text for Mistral reasoning.
        This captures what Gemini heard or generated as text.
        
        Returns:
            str: Transcription text, or None if queue is empty
        """
        try:
            if self.transcription_queue.empty():
                return None
            return self.transcription_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
        except Exception as e:
            logger.error(f"[Gemini] Get transcription error: {e}")
            return None
    
    async def stop(self):
        """Stop the Gemini session and clean up resources."""
        try:
            logger.info("[Gemini] Stopping session...")
            self.quit.set()
            
            # Clear queues
            while not self.input_queue.empty():
                try:
                    self.input_queue.get_nowait()
                except:
                    pass
            
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except:
                    pass
            
            # Close session if still open
            if self.session:
                try:
                    await self.session.close()
                except Exception as e:
                    logger.warning(f"[Gemini] Session close warning: {e}")
            
            logger.info("[Gemini] Session stopped")
            
        except Exception as e:
            logger.error(f"[Gemini] Stop error: {e}")
        finally:
            self.session = None
            self.quit.clear()
            self.session_ready.clear()
