from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from db_connect import connect_db
from typing import List
import traceback
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

router = APIRouter()

# Setup models (move to top if needed)
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(img_bgr):
    face = mtcnn(img_bgr)
    if face is None: return None
    with torch.no_grad():
        return facenet(face.unsqueeze(0)).cpu().numpy()
    
@router.post("/api/register_user")
async def register_user(
    username: str = Form(...),
    name: str = Form(...),
    files: List[UploadFile] = File(...)
):
    try:
        conn = await connect_db()
        samples = []
        for file in files:
            image_bytes = await file.read()
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            emb = get_embedding(img)
            if emb is not None:
                samples.append(emb)
        if not samples:
            raise HTTPException(status_code=400, detail="No face detected in any sample")
        
        mean_emb = np.mean(np.vstack(samples), axis=0).flatten().tolist()


        await conn.execute(
            """
            INSERT INTO user_details (username, name, face_embedding)
            VALUES ($1, $2, $3)
            ON CONFLICT (username)
            DO UPDATE SET name = EXCLUDED.name, face_embedding = EXCLUDED.face_embedding;
            """,
            username, name, mean_emb
        )
        await conn.close()

        return {"message": "User registered successfully", "username": username}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/verify_face")
async def verify_face(file: UploadFile = File(...)):
    try:
        conn = await connect_db()
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        emb = get_embedding(img)
        if emb is None:
            raise HTTPException(status_code=400, detail="No face detected")
        emb = emb.flatten()

        # Get all users
        rows = await conn.fetch("SELECT username, face_embedding FROM user_details")
        await conn.close()

        def cos_sim(a, b):
            a = np.array(a)
            b = np.array(b)
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        best_user = None
        best_score = -1
        for row in rows:
            db_embedding = np.array(row["face_embedding"])
            score = cos_sim(emb, db_embedding)
            if score > best_score:
                best_user = row["username"]
                best_score = score

        if best_score > 0.75:  # threshold, adjust as needed
            return {"message": "Login successful", "username": best_user, "similarity": best_score}
        else:
            raise HTTPException(status_code=401, detail="No matching face found")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))