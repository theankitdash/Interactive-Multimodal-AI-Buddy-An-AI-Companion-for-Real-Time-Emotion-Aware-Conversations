"""
Cognition Socket - Event-based reasoning and memory management

This WebSocket endpoint handles all intelligence, memory, and decision-making.
Gemini Reasoning controls the conversation flow; Gemini (Audio Socket) is just voice I/O.

Architecture:
- Audio Socket: mic → Gemini → speaker (codec only)
- Cognition Socket: events → Gemini Reasoning → memory/decisions (brain)
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from graphs.agent_graph import app as agent_graph
from utils.memory import get_user_profile
from session_registry import session_registry
import asyncio
import json
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)

# Store active cognition sessions
cognition_sessions = {}


class CognitionState:
    """Manage cognition session state for Mistral reasoning"""
    def __init__(self, session_id: str, username: str):
        self.session_id = session_id
        self.username = username
        self.chat_history = []
        self.user_profile = {}
        self.websocket = None
        self.last_processing_time = 0
        self.processing_lock = asyncio.Lock()
        # Transcription batching: accumulate fragments before processing
        self._transcription_buffer: list[str] = []
        self._debounce_task: asyncio.Task | None = None
        self._debounce_delay = 1.5  # seconds to wait for more fragments
        
    async def initialize_user_context(self):
        """Load user profile and context"""
        try:
            profile = await get_user_profile(self.username)
            self.user_profile = profile or {"name": self.username}
            logger.info(f"[Cognition] Loaded profile for {self.username}: {self.user_profile}")
        except Exception as e:
            logger.warning(f"Failed to load user profile: {e}")
            self.user_profile = {"name": self.username}
    
    def add_to_history(self, role: str, message: str):
        """Track conversation history for context"""
        self.chat_history.append(f"{role}: {message}")
        # Keep only last 20 messages for context
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]


@router.websocket("/stream")
async def cognition_stream(websocket: WebSocket):
    """
    Cognition WebSocket endpoint for event-based reasoning.
    
    Handles:
    - End-of-utterance events from Audio Socket
    - Memory storage/retrieval
    - Emotion detection
    - Intent classification
    - Decision making
    
    Does NOT handle:
    - Audio generation (delegated to Audio Socket)
    - Direct user speech (received via events only)
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
        
        # Initialize cognition state
        session_state = CognitionState(session_id=username, username=username)
        session_state.websocket = websocket
        
        # Load user context (profile, preferences, etc.)
        await session_state.initialize_user_context()
        
        # Store session
        cognition_sessions[username] = session_state
        
        # Register in session registry for inter-socket communication
        await session_registry.register_cognition_socket(username, session_state, websocket)
        
        logger.info(f"[Cognition] User {username} connected. Profile: {session_state.user_profile}")
        await websocket.send_json({
            "status": "connected",
            "message": f"Cognition Socket ready for {session_state.user_profile.get('name', username)}",
            "user": session_state.user_profile
        })
        
        # Main event loop
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                event_type = message.get("event")
                
                if event_type == "end_of_utterance":
                    # Process utterance through Mistral reasoning
                    await process_utterance(session_state, message)
                
                elif event_type == "transcription":
                    # Process transcription text (from Audio Socket or direct input)
                    transcription = message.get("text", "")
                    if transcription:
                        await process_transcription(session_state, transcription, message)
                
                elif event_type == "emotion_data":
                    # Store emotion data for context
                    await process_emotion(session_state, message)
                
                elif event_type == "user_action":
                    # User-initiated events (button clicks, etc.)
                    await process_user_action(session_state, message)
                
                elif event_type == "close":
                    break
            
            except WebSocketDisconnect:
                logger.info(f"[Cognition] Client {session_state.username} disconnected")
                break
            except Exception as e:
                logger.error(f"[Cognition] Event processing error: {e}")
    
    except WebSocketDisconnect:
        logger.info(f"[Cognition] Client disconnected")
    except Exception as e:
        logger.error(f"[Cognition] Error: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
    finally:
        # Cleanup
        if session_state:
            # Unregister from session registry
            await session_registry.unregister_cognition_socket(session_state.username)
            
            if session_state.session_id in cognition_sessions:
                del cognition_sessions[session_state.session_id]
            logger.info(f"[Cognition] Cleaned up session for {session_state.username}")
        
        try:
            await websocket.close()
        except:
            pass


async def process_utterance(session_state: CognitionState, event_data: dict):
    """
    Process end-of-utterance event.
    Triggered when Audio Socket detects user finished speaking.
    """
    async with session_state.processing_lock:
        try:
            # Debounce: prevent processing too frequently
            current_time = time.time()
            if (current_time - session_state.last_processing_time) < 2.0:
                logger.debug("[Cognition] Skipping utterance due to debounce")
                return
            
            session_state.last_processing_time = current_time
            
            # Get transcription from event
            transcription = event_data.get("transcription", "")
            if not transcription:
                logger.warning("[Cognition] No transcription in end_of_utterance event")
                return
            
            logger.info(f"[Cognition] Processing utterance: {transcription[:100]}")
            
            # Process through Mistral reasoning
            await process_transcription(session_state, transcription, event_data)
            
        except Exception as e:
            logger.error(f"[Cognition] Utterance processing error: {e}")


async def process_transcription(session_state: CognitionState, transcription: str, event_data: dict):
    """
    Buffer transcription fragments and process once the user stops speaking.
    Gemini sends partial transcriptions ('Do', 'you', 'know the time...'),
    so we accumulate them with a 1.5s debounce before triggering Mistral.
    """
    # Add fragment to buffer
    session_state._transcription_buffer.append(transcription)
    logger.debug(f"[Cognition] Buffered fragment: '{transcription}' (buffer size: {len(session_state._transcription_buffer)})")

    # Cancel any existing debounce timer
    if session_state._debounce_task and not session_state._debounce_task.done():
        session_state._debounce_task.cancel()

    # Start a new debounce timer — flush only after silence
    session_state._debounce_task = asyncio.create_task(
        _flush_transcription_buffer(session_state, event_data)
    )


async def _flush_transcription_buffer(session_state: CognitionState, event_data: dict):
    """Wait for debounce delay, then process the accumulated transcription."""
    try:
        await asyncio.sleep(session_state._debounce_delay)
    except asyncio.CancelledError:
        return  # New fragment arrived, timer reset

    # Combine all buffered fragments into one sentence
    full_text = " ".join(session_state._transcription_buffer).strip()
    session_state._transcription_buffer.clear()

    if not full_text:
        return

    logger.info(f"[Cognition] Processing full transcription: {full_text[:150]}")

    try:
        # Add to history
        session_state.add_to_history("user", full_text)
        
        # Prepare state for agent graph
        agent_state = {
            "input_text": full_text,
            "username": session_state.username,
            "chat_history": session_state.chat_history,
            "user_profile": session_state.user_profile,
            "reasoning_context": "",
            "final_response": "",
            "audio_mode": True  # Always true - Gemini handles voice response
        }
        
        # Run through Mistral reasoning (no Gemini generation)
        result = await agent_graph.ainvoke(agent_state)
        
        reasoning_context = result.get("reasoning_context", "")
        
        # Send reasoning results to frontend
        if session_state.websocket:
            await session_state.websocket.send_json({
                "event": "reasoning_complete",
                "context": reasoning_context,
                "timestamp": event_data.get("timestamp", time.time())
            })
        
        logger.info(f"[Cognition] Reasoning complete: {reasoning_context[:100]}")
        
    except Exception as e:
        logger.error(f"[Cognition] Transcription processing error: {e}")
        try:
            if session_state.websocket:
                await session_state.websocket.send_json({
                    "event": "error",
                    "error": f"Processing error: {str(e)}"
                })
        except Exception:
            pass


async def process_emotion(session_state: CognitionState, event_data: dict):
    """Process emotion detection data"""
    try:
        emotion = event_data.get("emotion", "neutral")
        confidence = event_data.get("confidence", 0.0)
        
        logger.info(f"[Cognition] Emotion detected: {emotion} ({confidence:.2f})")
        
        # Store emotion context for future reasoning
        # TODO: Integrate with memory/context system
        
    except Exception as e:
        logger.error(f"[Cognition] Emotion processing error: {e}")


async def process_user_action(session_state: CognitionState, event_data: dict):
    """Process explicit user actions (button clicks, settings changes, etc.)"""
    try:
        action = event_data.get("action", "")
        logger.info(f"[Cognition] User action: {action}")
        
        # Handle specific actions
        # TODO: Implement action handlers
        
    except Exception as e:
        logger.error(f"[Cognition] User action processing error: {e}")
