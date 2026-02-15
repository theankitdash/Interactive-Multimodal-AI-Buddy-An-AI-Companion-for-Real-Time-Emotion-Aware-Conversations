from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ai.gemini_handler import GeminiHandler
from ai.vision_analyzer import VisionAnalyzer
from graphs.agent_graph import app as agent_graph
from utils.memory import get_user_profile
from config import GEMINI_API_KEY
from session_registry import session_registry
import asyncio
import numpy as np
import json
import base64
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active sessions
active_sessions = {}

class SessionState:
    """Manage conversation session state"""
    def __init__(self, session_id: str, username: str):
        self.session_id = session_id
        self.username = username
        self.chat_history = []
        self.user_profile = {}
        self.gemini_handler = None
        self.vision_analyzer = None
        self.websocket = None
        self.last_audio_time = 0
        self.camera_on = False
        
    async def initialize_user_context(self):
        """Load user profile and context"""
        try:
            profile = await get_user_profile(self.username)
            self.user_profile = profile or {"name": self.username}
        except Exception as e:
            logger.warning(f"Failed to load user profile: {e}")
            self.user_profile = {"name": self.username}
    
    def add_to_history(self, role: str, message: str):
        """Track conversation history for context"""
        self.chat_history.append(f"{role}: {message}")
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
    
    async def cleanup(self):
        """Clean up resources"""
        if self.vision_analyzer:
            self.vision_analyzer.stop()
        if self.gemini_handler:
            try:
                await self.gemini_handler.stop()
            except Exception as e:
                logger.error(f"Error stopping gemini handler: {e}")


@router.websocket("/stream")
async def assistant_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time AI assistant interaction.
    Integrates reasoning and generation pipelines.
    Handles audio/video streaming bidirectionally with context awareness.
    """
    await websocket.accept()
    
    session_state = None
    
    try:
        # Wait for initialization message with username
        init_message = await websocket.receive_text()
        init_data = json.loads(init_message)
        
        username = init_data.get("username")
        if not username:
            await websocket.send_json({"error": "Username required"})
            await websocket.close()
            return
        
        # Initialize session state
        session_state = SessionState(session_id=username, username=username)
        session_state.websocket = websocket
        
        # Load user context (profile, preferences, etc.)
        await session_state.initialize_user_context()
        
        # Initialize Gemini handler for real-time audio streaming
        session_state.gemini_handler = GeminiHandler(api_key=GEMINI_API_KEY)
        
        # Initialize Vision Analyzer
        session_state.vision_analyzer = VisionAnalyzer()
        
        # System instruction — vision context is injected dynamically by the vision analyzer
        system_instruction = (
            f"You are a helpful AI companion talking to {session_state.user_profile.get('name', username)}. "
            "You may receive visual context descriptions about what the camera sees. "
            "When visual context says 'Camera is off', do NOT describe or guess what you see — "
            "clearly tell the user the camera is off. "
            "When visual context is provided, use it to respond naturally about what you see. "
            "Respond conversationally, empathetically, and concisely."
        )
        gemini_task = asyncio.create_task(session_state.gemini_handler.start(system_instruction=system_instruction))
        
        # Start Vision Analyzer loop
        session_state.vision_analyzer.start()
        
        # Store session
        active_sessions[username] = session_state
        
        # Register in session registry for inter-socket communication
        await session_registry.register_audio_socket(username, session_state, websocket)
        
        logger.info(f"[WebSocket] User {username} connected. Profile: {session_state.user_profile}")
        await websocket.send_json({
            "status": "connected", 
            "message": f"Welcome, {session_state.user_profile.get('name', username)}!",
            "user": session_state.user_profile
        })
        
        # Main loop: receive from client, process through reasoning pipeline, send to Gemini
        async def receive_from_client():
            """Receive audio/video from client and route to Gemini"""
            try:
                while True:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    msg_type = message.get("type")
                    msg_data = message.get("data")
                    
                    if msg_type == "audio":
                        # Decode base64 audio and send to Gemini for processing
                        try:
                            session_state.last_audio_time = asyncio.get_event_loop().time()
                            audio_bytes = base64.b64decode(msg_data)
                            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                            await session_state.gemini_handler.send_audio(audio_array)
                        except Exception as e:
                            logger.error(f"Error processing audio: {e}")
                    
                    elif msg_type == "video":
                        # Decode base64 image and send to both Gemini and Vision Analyzer
                        try:
                            import cv2
                            image_data = base64.b64decode(msg_data)
                            nparr = np.frombuffer(image_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                logger.debug(f"[WebSocket] Video frame received: {frame.shape}")
                                # Send to Gemini for live context
                                await session_state.gemini_handler.send_video(frame)
                                # Feed to Vision Analyzer for periodic scene descriptions
                                session_state.vision_analyzer.update_frame(frame)
                            else:
                                logger.warning("[WebSocket] Failed to decode video frame")
                        except Exception as e:
                            logger.error(f"[WebSocket] Error processing video: {e}", exc_info=True)
                    
                    elif msg_type == "camera_on":
                        # Camera turned on — update state and notify Gemini
                        session_state.camera_on = True
                        session_state.vision_analyzer.set_camera_state(True)
                        await session_state.gemini_handler.send_text(
                            "[CAMERA ON] The user's camera is now active. You can see them. "
                            "Wait for visual context updates."
                        )
                        logger.info(f"[WebSocket] Camera ON for {username}")
                    
                    elif msg_type == "camera_off":
                        # Camera turned off — update state and notify Gemini
                        session_state.camera_on = False
                        session_state.vision_analyzer.set_camera_state(False)
                        await session_state.gemini_handler.send_text(
                            "[CAMERA OFF] The user's camera is now OFF. "
                            "You CANNOT see anything. Do not guess or describe what you see."
                        )
                        logger.info(f"[WebSocket] Camera OFF for {username}")
                    
                    elif msg_type == "text":
                        current_time = asyncio.get_event_loop().time()
                        if session_state.last_audio_time > 0 and (current_time - session_state.last_audio_time < 10.0):
                            logger.info(f"[Text Ignored] Skipping text input as audio was recently processed: {msg_data[:50]}")
                            continue
                        await process_user_text(session_state, msg_data, silent=True)
                    
                    elif msg_type == "text_only":
                        await process_user_text(session_state, msg_data, silent=False)
                    
                    elif msg_type == "close":
                        break
            
            except WebSocketDisconnect:
                logger.info(f"[WebSocket] Client {session_state.username} disconnected")
            except Exception as e:
                logger.error(f"[WebSocket Receive Error] {e}")

        async def send_to_client():
            """Get audio replies from Gemini and send to client"""
            try:
                while True:
                    reply = await session_state.gemini_handler.get_audio_reply()
                    if reply:
                        sample_rate, audio_np = reply
                        audio_bytes = audio_np.tobytes()
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        await websocket.send_json({
                            "type": "audio_reply",
                            "data": audio_b64,
                            "sample_rate": sample_rate
                        })
                    
                    await asyncio.sleep(0.01)
            
            except Exception as e:
                logger.error(f"[WebSocket Send Error] {e}")

        async def forward_transcriptions():
            """Poll transcription queue and forward to Cognition Socket"""
            try:
                while True:
                    # Get transcription from Gemini
                    transcription = await session_state.gemini_handler.get_transcription()
                    if transcription:
                        logger.info(f"[Audio Socket] Forwarding transcription to Cognition: {transcription[:100]}")
                        
                        # Forward to Cognition Socket via registry
                        await session_registry.forward_to_cognition(username, {
                            "event": "transcription",
                            "text": transcription,
                            "source": "gemini_audio",
                            "timestamp": time.time()
                        })
                    
                    await asyncio.sleep(0.1)  # Poll interval
            except Exception as e:
                logger.error(f"[Audio Socket] Transcription forwarding error: {e}")

        async def inject_vision_context():
            """Periodically inject vision descriptions into Gemini session"""
            last_description = ""
            try:
                while True:
                    await asyncio.sleep(4)  # Slightly offset from vision analysis interval
                    
                    if session_state.vision_analyzer and session_state.camera_on:
                        description = session_state.vision_analyzer.latest_description
                        # Only inject if description changed
                        if description and description != last_description:
                            last_description = description
                            await session_state.gemini_handler.send_text(
                                f"[VISUAL CONTEXT] What you currently see: {description}"
                            )
                            logger.debug(f"[Vision→Gemini] Injected: {description[:80]}")
            except Exception as e:
                logger.error(f"[Vision Injection Error] {e}")

        # Run all concurrent tasks
        await asyncio.gather(
            receive_from_client(),
            send_to_client(),
            forward_transcriptions(),
            inject_vision_context(),  # Vision context injection loop
            return_exceptions=True
        )
    
    except WebSocketDisconnect:
        logger.info(f"[WebSocket] Client disconnected")
    except Exception as e:
        logger.error(f"[WebSocket Error] {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Cleanup
        if session_state:
            try:
                await session_registry.unregister_audio_socket(session_state.username)
                await session_state.cleanup()
                if session_state.session_id in active_sessions:
                    del active_sessions[session_state.session_id]
                logger.info(f"[WebSocket] Cleaned up session for {session_state.username}")
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
        
        try:
            await websocket.close()
        except:
            pass


async def process_user_text(session_state: SessionState, user_input: str, silent: bool = False):
    """
    Process user text through the reasoning and generation pipeline.
    
    Args:
        session_state: Current session state
        user_input: User's text input
        silent: If True, only run reasoning (save facts/events) without sending response.
    """
    try:
        session_state.add_to_history("user", user_input)
        
        # Include vision context so Mistral also knows what's visible
        vision_context = ""
        if session_state.vision_analyzer:
            vision_context = session_state.vision_analyzer.latest_description
        
        agent_state = {
            "input_text": user_input,
            "username": session_state.username,
            "chat_history": session_state.chat_history,
            "user_profile": session_state.user_profile,
            "reasoning_context": "",
            "final_response": "",
            "audio_mode": silent  # Skip generation when in audio mode
        }
        
        logger.info(f"[Processing] User {session_state.username}: {user_input[:100]} (silent={silent})")
        
        result = await agent_graph.ainvoke(agent_state)
        
        reasoning_context = result.get("reasoning_context", "")
        final_response = result.get("final_response", "No response generated")
        
        session_state.add_to_history("assistant", final_response)
        
        if not silent:
            await session_state.websocket.send_json({
                "type": "text_response",
                "response": final_response,
                "context": reasoning_context,
                "thinking": True
            })
            logger.info(f"[Processing Complete] Response sent to {session_state.username}")
        else:
            logger.info(f"[Reasoning Complete] Silent mode - saved context: {reasoning_context[:100]}")
        
    except Exception as e:
        logger.error(f"Error processing user text: {e}", exc_info=True)
        if not silent:
            try:
                await session_state.websocket.send_json({
                    "type": "error",
                    "error": f"Processing error: {str(e)}"
                })
            except:
                pass
