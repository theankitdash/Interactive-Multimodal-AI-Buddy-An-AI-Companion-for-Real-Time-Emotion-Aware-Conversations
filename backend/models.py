from pydantic import BaseModel
from typing import Optional, List


class RegisterRequest(BaseModel):
    username: str
    fullname: str
    face_embeddings: List[List[float]]  


class LoginRequest(BaseModel):
    face_embeddings: List[List[float]]


class UserResponse(BaseModel):
    username: str
    fullname: str
    initials: str


class FaceCaptureRequest(BaseModel):
    image_data: str 


class FaceCaptureResponse(BaseModel):
    success: bool
    message: str
    embedding: Optional[List[float]] = None


class MultiSampleRegisterRequest(BaseModel):
    username: str
    fullname: str
    sample_images: List[str]  


class StreamMessage(BaseModel):
    type: str  
    data: str  
