from pydantic import BaseModel
from typing import Optional, List


class RegisterRequest(BaseModel):
    username: str
    fullname: str
    face_embeddings: List[List[float]]  # Required list of face embeddings for authentication


class LoginRequest(BaseModel):
    face_embeddings: List[List[float]]  # List of face embeddings for matching


class UserResponse(BaseModel):
    username: str
    fullname: str
    initials: str


class FaceCaptureRequest(BaseModel):
    image_data: str  # Base64 encoded image


class FaceCaptureResponse(BaseModel):
    success: bool
    message: str
    embedding: Optional[List[float]] = None


class MultiSampleRegisterRequest(BaseModel):
    username: str
    fullname: str
    sample_images: List[str]  # Base64 encoded images (up to 50)


class StreamMessage(BaseModel):
    type: str  # "audio", "video", "text"
    data: str  # Base64 or text content
