from fastapi import APIRouter

router = APIRouter()

# Media routes are handled directly through auth routes (capture-face)
# and assistant WebSocket (audio/video streaming)
# This file is kept for future media-related endpoints if needed

@router.get("/status")
async def media_status():
    """Check media subsystem status"""
    return {"status": "ready", "message": "Media endpoints available"}
