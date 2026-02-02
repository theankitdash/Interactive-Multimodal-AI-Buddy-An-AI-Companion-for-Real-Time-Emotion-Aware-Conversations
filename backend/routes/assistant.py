from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ai.gemini_handler import GeminiHandler
from ai.langchain_handler import LangchainHandler
from config import GEMINI_API_KEY, NVIDIA_API_KEY
import asyncio
import numpy as np
import json
import base64

router = APIRouter()

# Store active sessions
active_sessions = {}


@router.websocket("/stream")
async def assistant_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time AI assistant interaction.
    Handles audio/video streaming bidirectionally.
    """
    await websocket.accept()
    
    session_id = None
    gemini_handler = None
    langchain_handler = None
    
    try:
        # Wait for initialization message with username
        init_message = await websocket.receive_text()
        init_data = json.loads(init_message)
        
        username = init_data.get("username")
        if not username:
            await websocket.send_json({"error": "Username required"})
            await websocket.close()
            return
        
        session_id = username
        
        # Initialize handlers
        gemini_handler = GeminiHandler(api_key=GEMINI_API_KEY)
        langchain_handler = LangchainHandler(username=username, NVIDIA_API_KEY=NVIDIA_API_KEY)
        gemini_handler.attach_memory(langchain_handler)
        
        # Start Gemini session
        asyncio.create_task(gemini_handler.start())
        
        # Store session
        active_sessions[session_id] = {
            "gemini": gemini_handler,
            "langchain": langchain_handler,
            "websocket": websocket
        }
        
        await websocket.send_json({"status": "connected", "message": "Assistant ready"})
        
        # Main loop: receive from client, send to Gemini, stream back responses
        async def receive_from_client():
            """Receive audio/video from client and forward to Gemini"""
            while True:
                try:
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    msg_type = message.get("type")
                    msg_data = message.get("data")
                    
                    if msg_type == "audio":
                        # Decode base64 audio and send to Gemini
                        audio_bytes = base64.b64decode(msg_data)
                        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
                        await gemini_handler.send_audio(audio_array)
                    
                    elif msg_type == "video":
                        # Decode base64 image and send to Gemini
                        import cv2
                        image_data = base64.b64decode(msg_data)
                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            await gemini_handler.send_video(frame)
                    
                    elif msg_type == "close":
                        break
                
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"[WebSocket Receive Error] {e}")
                    break
        
        async def send_to_client():
            """Get audio replies from Gemini and send to client"""
            while True:
                try:
                    reply = await gemini_handler.get_audio_reply()
                    if reply:
                        sample_rate, audio_np = reply
                        # Convert to bytes and base64
                        audio_bytes = audio_np.tobytes()
                        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                        
                        await websocket.send_json({
                            "type": "audio_reply",
                            "data": audio_b64,
                            "sample_rate": sample_rate
                        })
                    
                    await asyncio.sleep(0.01)
                
                except Exception as e:
                    print(f"[WebSocket Send Error] {e}")
                    break
        
        # Run both tasks concurrently
        await asyncio.gather(
            receive_from_client(),
            send_to_client()
        )
    
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket Error] {e}")
    finally:
        # Cleanup
        if session_id and session_id in active_sessions:
            try:
                if gemini_handler:
                    await gemini_handler.stop()
                if langchain_handler:
                    await langchain_handler.stop()
            except:
                pass
            del active_sessions[session_id]
        
        try:
            await websocket.close()
        except:
            pass
