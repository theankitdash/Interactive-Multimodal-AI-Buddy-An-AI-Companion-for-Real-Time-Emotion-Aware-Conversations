from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from models import (
    RegisterRequest, LoginRequest, UserResponse, 
    FaceCaptureRequest, FaceCaptureResponse,
    LivenessChallengeRequest, LivenessChallengeResponse,
    LivenessChallengesResponse, MultiSampleRegisterRequest
)
from utils.db_connect import connect_db
from utils.face_utils import (
    get_embedding, 
    analyze_liveness_frame, 
    get_random_liveness_challenges
)
import numpy as np
import cv2
import base64
from config import FACE_RECOGNITION_THRESHOLD, FACE_SAMPLES_FOR_AUTH, FACE_REGISTRATION_SAMPLES

router = APIRouter()


def get_initials(name: str) -> str:
    """Extract 2-letter initials from user name"""
    if not name:
        return "??"
    words = name.strip().split()
    if len(words) >= 2:
        return (words[0][0] + words[-1][0]).upper()
    elif len(words) == 1:
        return words[0][:2].upper()
    return "??"


@router.post("/register", response_model=UserResponse)
async def register_user(request: RegisterRequest):
    """Register a new user with username, fullname, and face embeddings"""
    username = request.username.strip()
    fullname = request.fullname.strip()
    
    if not username or not fullname:
        raise HTTPException(status_code=400, detail="Username and fullname are required")
    
    # Validate face embeddings are provided (now required)
    if not request.face_embeddings or len(request.face_embeddings) == 0:
        raise HTTPException(status_code=400, detail="Face embeddings are required for registration")
    
    # Calculate average face embedding
    face_embedding = np.mean(np.array(request.face_embeddings), axis=0).tolist()
    
    conn = await connect_db()
    try:
        await conn.execute(
            """INSERT INTO user_details (username, name, face_embedding)
               VALUES ($1, $2, $3)
               ON CONFLICT (username) DO UPDATE SET name=$2, face_embedding=$3""",
            username, fullname, face_embedding
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        await conn.close()
    
    return UserResponse(
        username=username,
        fullname=fullname,
        initials=get_initials(fullname)
    )


@router.post("/capture-face", response_model=FaceCaptureResponse)
async def capture_face(request: FaceCaptureRequest):
    """
    Capture face embedding from base64 image data.
    Used for both registration and login.
    """
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(request.image_data.split(',')[1] if ',' in request.image_data else request.image_data)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as decode_error:
            print(f"[ERROR] Image decoding failed: {decode_error}")
            raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(decode_error)}")
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data - frame is None")
        
        # Get face embedding
        try:
            embedding = get_embedding(frame)
        except Exception as embedding_error:
            print(f"[ERROR] Face embedding extraction failed: {embedding_error}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Face embedding failed: {str(embedding_error)}")
        
        if embedding is None:
            return FaceCaptureResponse(
                success=False,
                message="No face detected in image",
                embedding=None
            )
        
        # Flatten embedding from (1, 512) to (512,) before converting to list
        return FaceCaptureResponse(
            success=True,
            message="Face captured successfully",
            embedding=embedding.flatten().tolist()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in capture_face: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Face capture error: {str(e)}")


@router.post("/update-face-embedding")
async def update_face_embedding(embeddings: list, username: str = Query(...)):
    """
    Update user's face embedding in database.
    Accepts multiple embeddings and stores the average.
    """
    if not embeddings:
        raise HTTPException(status_code=400, detail="No embeddings provided")
    
    try:
        # Calculate average embedding
        avg_embedding = np.mean(np.array(embeddings), axis=0).tolist()
        
        conn = await connect_db()
        try:
            await conn.execute(
                """UPDATE user_details SET face_embedding = $1 WHERE username = $2""",
                avg_embedding, username
            )
        finally:
            await conn.close()
        
        return {"success": True, "message": "Face embedding updated"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/login", response_model=UserResponse)
async def login_user(request: LoginRequest):
    """
    Login user via face recognition.
    Accepts array of face embeddings and finds best match.
    """
    embeddings = request.face_embeddings
    
    if not embeddings:
        raise HTTPException(status_code=400, detail="No face embeddings provided")
    
    # Calculate average embedding from multiple captures
    face_embedding = np.mean(np.array(embeddings), axis=0)
    
    conn = await connect_db()
    try:
        rows = await conn.fetch(
            "SELECT username, name, face_embedding FROM user_details WHERE face_embedding IS NOT NULL"
        )
        
        if not rows:
            raise HTTPException(status_code=404, detail="No registered users found")
        
        # Cosine similarity
        def cos_sim(a, b):
            return float(np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b)))
        
        best_username, best_name, best_score = None, None, -1
        for row in rows:
            ref_emb = np.array(row["face_embedding"])
            score = cos_sim(face_embedding, ref_emb)
            if score > best_score:
                best_username, best_name, best_score = row["username"], row["name"], score
        
        if best_score > FACE_RECOGNITION_THRESHOLD:
            return UserResponse(
                username=best_username,
                fullname=best_name,
                initials=get_initials(best_name)
            )
        else:
            raise HTTPException(status_code=401, detail="Face not recognized")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Login error: {str(e)}")
    finally:
        await conn.close()


@router.post("/liveness-challenge", response_model=LivenessChallengeResponse)
async def liveness_challenge(request: LivenessChallengeRequest):
    """
    Analyze a single frame for liveness detection.
    Supports blink, head_movement, and smile challenges.
    """
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(
                request.image_data.split(',')[1] if ',' in request.image_data else request.image_data
            )
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as decode_error:
            print(f"[ERROR] Image decoding failed: {decode_error}")
            raise HTTPException(status_code=400, detail=f"Image decoding failed: {str(decode_error)}")
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Analyze frame for liveness
        result = analyze_liveness_frame(
            frame, 
            request.challenge_type,
            request.reference_data
        )
        
        return LivenessChallengeResponse(
            success=result["success"],
            message=result["message"],
            challenge_complete=result["challenge_complete"],
            reference_data=result.get("reference_data")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Liveness challenge failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Liveness challenge error: {str(e)}")


@router.get("/start-liveness", response_model=LivenessChallengesResponse)
async def start_liveness():
    """
    Get a random sequence of 3 liveness challenges.
    Returns challenges in the order they should be completed.
    """
    challenges = get_random_liveness_challenges(3)
    return LivenessChallengesResponse(
        challenges=challenges,
        message=f"Complete these challenges in order: {', '.join(challenges)}"
    )


@router.post("/register-multi-sample", response_model=UserResponse)
async def register_multi_sample(request: MultiSampleRegisterRequest):
    """
    Register a new user with multiple face samples.
    Accepts up to 50 face image samples and averages their embeddings.
    """
    username = request.username.strip()
    fullname = request.fullname.strip()
    
    if not username or not fullname:
        raise HTTPException(status_code=400, detail="Username and fullname are required")
    
    if not request.sample_images or len(request.sample_images) == 0:
        raise HTTPException(status_code=400, detail="At least one face sample is required")
    
    if len(request.sample_images) > FACE_REGISTRATION_SAMPLES:
        raise HTTPException(
            status_code=400, 
            detail=f"Maximum {FACE_REGISTRATION_SAMPLES} samples allowed"
        )
    
    # Process all sample images and extract embeddings
    embeddings = []
    failed_samples = 0
    
    for idx, image_data in enumerate(request.sample_images):
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(
                image_data.split(',')[1] if ',' in image_data else image_data
            )
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                failed_samples += 1
                print(f"[WARNING] Sample {idx+1} failed to decode")
                continue
            
            # Extract embedding
            embedding = get_embedding(frame)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                failed_samples += 1
                print(f"[WARNING] Sample {idx+1} had no face detected")
        
        except Exception as e:
            failed_samples += 1
            print(f"[ERROR] Sample {idx+1} processing failed: {e}")
            continue
    
    # Require at least 70% successful samples
    min_required = max(3, int(len(request.sample_images) * 0.7))
    if len(embeddings) < min_required:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough valid face samples. Got {len(embeddings)}, need at least {min_required}"
        )
    
    # Calculate average embedding
    face_embedding = np.mean(np.vstack(embeddings), axis=0).tolist()
    
    # Store in database
    conn = await connect_db()
    try:
        await conn.execute(
            """INSERT INTO user_details (username, name, face_embedding)
               VALUES ($1, $2, $3)
               ON CONFLICT (username) DO UPDATE SET name=$2, face_embedding=$3""",
            username, fullname, face_embedding
        )
        print(f"[INFO] Registered {username} with {len(embeddings)}/{len(request.sample_images)} samples")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        await conn.close()
    
    return UserResponse(
        username=username,
        fullname=fullname,
        initials=get_initials(fullname)
    )
