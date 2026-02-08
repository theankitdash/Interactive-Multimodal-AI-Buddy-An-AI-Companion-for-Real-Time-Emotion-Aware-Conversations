from fastapi import APIRouter, HTTPException, Query
from models import (
    RegisterRequest, LoginRequest, UserResponse, 
    FaceCaptureRequest, FaceCaptureResponse, MultiSampleRegisterRequest
)
from utils.db_connect import connect_db
from utils.face_utils import get_embedding
    
import numpy as np
import cv2
import base64
from config import FACE_RECOGNITION_THRESHOLD, FACE_REGISTRATION_SAMPLES

router = APIRouter()

def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        raise ValueError("Zero-norm embedding")
    return vec / norm

def get_initials(name: str) -> str:
    words = name.strip().split()
    if len(words) >= 2:
        return (words[0][0] + words[-1][0]).upper()
    return words[0][:2].upper()

def decode_base64_image(data: str):
    raw = base64.b64decode(data.split(",")[1] if "," in data else data)
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return img

@router.post("/capture-face", response_model=FaceCaptureResponse)
async def capture_face(req: FaceCaptureRequest):
    frame = decode_base64_image(req.image_data)

    embedding = get_embedding(frame)
    if embedding is None:
        return FaceCaptureResponse(
            success=False,
            message="No face detected",
            embedding=None
        )

    embedding = normalize(np.array(embedding)).tolist()

    return FaceCaptureResponse(
        success=True,
        message="Face captured",
        embedding=embedding
    )

@router.post("/register", response_model=UserResponse)
async def register_user(req: RegisterRequest):

    if not req.face_embeddings:
        raise HTTPException(status_code=400, detail="Face embeddings are required")
    
    embeddings = np.array(req.face_embeddings, dtype=np.float32)

    if embeddings.shape[1] != 512:
        raise HTTPException(400, "Invalid embedding dimension")
    
    avg = normalize(np.mean(embeddings, axis=0))
    
    conn = await connect_db()
    try:
        await conn.execute(
            """INSERT INTO user_details (username, name, face_embedding)
               VALUES ($1, $2, $3::vector)
               ON CONFLICT (username) 
               DO UPDATE SET name= EXCLUDED.name, 
                             face_embedding = EXCLUDED.face_embedding;
            """,req.username, req.fullname, avg.tolist())
    finally:
        await conn.close()
    
    return UserResponse(
        username=req.username,
        fullname=req.fullname,
        initials=get_initials(req.fullname)
    )

@router.post("/login", response_model=UserResponse)
async def login_user(req: LoginRequest):
    embeddings = np.array(req.face_embeddings, dtype=np.float32)
    probe = normalize(np.mean(embeddings, axis=0))
    
    conn = await connect_db()
    try:
        row = await conn.fetchrow("""
            SELECT username, name,
                    1 - (face_embedding <=> $1::vector) AS score
            FROM user_details 
            WHERE face_embedding IS NOT NULL
            ORDER BY face_embedding <=> $1::vector
            LIMIT 1
        """, probe.tolist())
        
        if not row or row["score"] < FACE_RECOGNITION_THRESHOLD:
            raise HTTPException(401, "Face not recognized")
        
        return UserResponse(
            username=row["username"],
            fullname=row["name"],
            initials=get_initials(row["name"])
        )
    finally:
        await conn.close()    

@router.post("/register-multi-sample", response_model=UserResponse)
async def register_multi_sample(req: MultiSampleRegisterRequest):
    
    if len(req.sample_images) > FACE_REGISTRATION_SAMPLES:
        raise HTTPException(400, "Too many samples")
    
    # Process all sample images and extract embeddings
    embeddings = []
    for img_data in req.sample_images:
        frame = decode_base64_image(img_data)
        emb = get_embedding(frame)
        if emb is not None:
            embeddings.append(normalize(np.array(emb)))

    if len(embeddings) < max(3, int(len(req.sample_images) * 0.7)):
        raise HTTPException(400, "Insufficient valid samples")

    avg = normalize(np.mean(embeddings, axis=0))
    
    # Store in database
    conn = await connect_db()
    try:
        await conn.execute(
            """INSERT INTO user_details (username, name, face_embedding)
               VALUES ($1, $2, $3::vector)
               ON CONFLICT (username) 
               DO UPDATE SET name=EXCLUDED.name,
                             face_embedding=EXCLUDED.face_embedding
            """, req.username, req.fullname, avg.tolist())
    finally:
        await conn.close()
    
    return UserResponse(
        username=req.username,
        fullname=req.fullname,
        initials=get_initials(req.fullname)
    )