import asyncio
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SessionPair:
    username: str
    audio_session: Optional[Any] = None
    cognition_session: Optional[Any] = None
    audio_websocket: Optional[Any] = None
    cognition_websocket: Optional[Any] = None


class SessionRegistry:
    
    def __init__(self):
        self._sessions: Dict[str, SessionPair] = {}
        self._lock = asyncio.Lock()
    
    async def register_audio_socket(self, username: str, session_state, websocket):
        async with self._lock:
            if username not in self._sessions:
                self._sessions[username] = SessionPair(username=username)
            
            self._sessions[username].audio_session = session_state
            self._sessions[username].audio_websocket = websocket
            logger.info(f"[Registry] Audio Socket registered for {username}")
    
    async def register_cognition_socket(self, username: str, session_state, websocket):
        async with self._lock:
            if username not in self._sessions:
                self._sessions[username] = SessionPair(username=username)
            
            self._sessions[username].cognition_session = session_state
            self._sessions[username].cognition_websocket = websocket
            logger.info(f"[Registry] Cognition Socket registered for {username}")
    
    async def unregister_audio_socket(self, username: str):
        async with self._lock:
            if username in self._sessions:
                self._sessions[username].audio_session = None
                self._sessions[username].audio_websocket = None
                
                # Remove if both sockets disconnected
                if not self._sessions[username].cognition_session:
                    del self._sessions[username]
                    logger.info(f"[Registry] Removed session for {username}")
    
    async def unregister_cognition_socket(self, username: str):
        async with self._lock:
            if username in self._sessions:
                self._sessions[username].cognition_session = None
                self._sessions[username].cognition_websocket = None
                
                # Remove if both sockets disconnected
                if not self._sessions[username].audio_session:
                    del self._sessions[username]
                    logger.info(f"[Registry] Removed session for {username}")
    
    def get_cognition_websocket(self, username: str):
        pair = self._sessions.get(username)
        return pair.cognition_websocket if pair else None
    
    def get_audio_websocket(self, username: str):
        pair = self._sessions.get(username)
        return pair.audio_websocket if pair else None
    
    def get_cognition_session(self, username: str):
        pair = self._sessions.get(username)
        return pair.cognition_session if pair else None
    
    def get_audio_session(self, username: str):
        pair = self._sessions.get(username)
        return pair.audio_session if pair else None
    
    async def forward_to_cognition(self, username: str, event: dict):
        pair = self._sessions.get(username)
        if not pair or not pair.cognition_session:
            logger.warning(f"[Registry] No Cognition session for {username}")
            return False

        try:
            # Import here to avoid circular imports
            from routes.cognition import process_transcription

            transcription = event.get("text", "")
            if transcription:
                await process_transcription(pair.cognition_session, transcription, event)
                logger.info(f"[Registry] Processed transcription via Cognition: {transcription[:80]}")
                return True
            else:
                logger.warning(f"[Registry] Empty transcription in forwarded event")
                return False
        except Exception as e:
            logger.error(f"[Registry] Failed to process transcription: {e}")
            return False
    
    async def send_to_audio(self, username: str, command: dict):
        audio_ws = self.get_audio_websocket(username)
        if audio_ws:
            try:
                await audio_ws.send_text(json.dumps(command))
                logger.debug(f"[Registry] Sent command to Audio Socket: {command.get('command')}")
                return True
            except Exception as e:
                logger.error(f"[Registry] Failed to send to Audio: {e}")
                return False
        else:
            logger.warning(f"[Registry] No Audio Socket for {username}")
            return False

    async def inject_context_to_gemini(self, username: str, context_text: str):
        """Inject text context (memories, events, reasoning) into Gemini Live Audio."""
        pair = self._sessions.get(username)
        if not pair or not pair.audio_session:
            logger.warning(f"[Registry] No Audio session for {username} — cannot inject context")
            return False

        try:
            gemini_handler = pair.audio_session.gemini_handler
            if gemini_handler and gemini_handler.session_ready.is_set():
                await gemini_handler.send_text(context_text)
                logger.info(f"[Registry] Injected context to Gemini for {username}: {context_text[:80]}")
                return True
            else:
                logger.warning(f"[Registry] Gemini session not ready for {username}")
                return False
        except Exception as e:
            logger.error(f"[Registry] Failed to inject context to Gemini: {e}")
            return False


# Global singleton instance
session_registry = SessionRegistry()
