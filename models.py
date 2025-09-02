from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from datetime import datetime

class UserDetails(BaseModel):
    username: str
    name: str
    face_embedding: List[float]  # vector(512)

class Event(BaseModel):
    event_id: Optional[UUID]
    username: str
    type: str  # e.g., task, reminder, meeting
    description: str
    event_time: Optional[datetime] = None
    repeat_interval: Optional[str] = None
    priority: Optional[int] = 1
    status: Optional[str] = "pending"
    completed_at: Optional[datetime] = None

class UserKnowledge(BaseModel):
    knowledge_id: Optional[UUID]
    username: str
    fact: str
    category: Optional[str] = None
    importance: Optional[int] = 3