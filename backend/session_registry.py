"""
Shared Session Registry for Inter-Socket Communication

This module provides a centralized registry for coordinating between
Audio Socket (Gemini) and Cognition Socket (Mistral).

Usage:
    from session_registry import session_registry
    
    # Register socket
    session_registry.register_audio_socket(username, session_state)
    
    # Get counterpart socket
    cognition = session_registry.get_cognition_socket(username)
"""
import asyncio
import json
import logging
from typing import Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SessionPair:
    """Holds references to both Audio and Cognition sockets for a user"""
    username: str
    audio_session: Optional[Any] = None
    cognition_session: Optional[Any] = None
    audio_websocket: Optional[Any] = None
    cognition_websocket: Optional[Any] = None


class SessionRegistry:
    """
    Central registry for managing paired Audio and Cognition sessions.
    
    Enables cross-socket communication:
    - Audio Socket can forward transcriptions to Cognition Socket
    - Cognition Socket can send directives to Audio Socket
    """
    
    def __init__(self):
        self._sessions: Dict[str, SessionPair] = {}
        self._lock = asyncio.Lock()
    
    async def register_audio_socket(self, username: str, session_state, websocket):
        """Register Audio Socket session"""
        async with self._lock:
            if username not in self._sessions:
                self._sessions[username] = SessionPair(username=username)
            
            self._sessions[username].audio_session = session_state
            self._sessions[username].audio_websocket = websocket
            logger.info(f"[Registry] Audio Socket registered for {username}")
    
    async def register_cognition_socket(self, username: str, session_state, websocket):
        """Register Cognition Socket session"""
        async with self._lock:
            if username not in self._sessions:
                self._sessions[username] = SessionPair(username=username)
            
            self._sessions[username].cognition_session = session_state
            self._sessions[username].cognition_websocket = websocket
            logger.info(f"[Registry] Cognition Socket registered for {username}")
    
    async def unregister_audio_socket(self, username: str):
        """Unregister Audio Socket"""
        async with self._lock:
            if username in self._sessions:
                self._sessions[username].audio_session = None
                self._sessions[username].audio_websocket = None
                
                # Remove if both sockets disconnected
                if not self._sessions[username].cognition_session:
                    del self._sessions[username]
                    logger.info(f"[Registry] Removed session for {username}")
    
    async def unregister_cognition_socket(self, username: str):
        """Unregister Cognition Socket"""
        async with self._lock:
            if username in self._sessions:
                self._sessions[username].cognition_session = None
                self._sessions[username].cognition_websocket = None
                
                # Remove if both sockets disconnected
                if not self._sessions[username].audio_session:
                    del self._sessions[username]
                    logger.info(f"[Registry] Removed session for {username}")
    
    def get_cognition_websocket(self, username: str):
        """Get Cognition WebSocket for a user"""
        pair = self._sessions.get(username)
        return pair.cognition_websocket if pair else None
    
    def get_audio_websocket(self, username: str):
        """Get Audio WebSocket for a user"""
        pair = self._sessions.get(username)
        return pair.audio_websocket if pair else None
    
    def get_cognition_session(self, username: str):
        """Get Cognition session state for a user"""
        pair = self._sessions.get(username)
        return pair.cognition_session if pair else None
    
    def get_audio_session(self, username: str):
        """Get Audio session state for a user"""
        pair = self._sessions.get(username)
        return pair.audio_session if pair else None
    
    async def forward_to_cognition(self, username: str, event: dict):
        """
        Forward an event from Audio Socket to Cognition Socket.
        Directly invokes process_transcription on the cognition session
        rather than sending through the WebSocket (which would just echo to the frontend).
        """
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
        """
        Send a command from Cognition Socket to Audio Socket.
        Future use: directive system for controlling Gemini.
        """
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


# Global singleton instance
session_registry = SessionRegistry()
