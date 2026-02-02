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


class LivenessChallengeRequest(BaseModel):
    challenge_type: str  # "blink", "head_movement", "smile"
    image_data: str  # Base64 encoded frame
    reference_data: Optional[dict] = None  # For stateful challenges like head movement


class LivenessChallengeResponse(BaseModel):
    success: bool
    message: str
    challenge_complete: bool
    reference_data: Optional[dict] = None


class LivenessChallengesResponse(BaseModel):
    challenges: List[str]
    message: str


class MultiSampleRegisterRequest(BaseModel):
    username: str
    fullname: str
    sample_images: List[str]  # Base64 encoded images (up to 50)


class StreamMessage(BaseModel):
    type: str  # "audio", "video", "text"
    data: str  # Base64 or text content
