import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize FaceNet models with error handling
try:
    mtcnn = MTCNN(
        image_size=160, 
        margin=0, 
        min_face_size=40, 
        device='cpu',
        post_process=False  # Don't normalize, we'll handle it
    )
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    print("[INFO] Face recognition models (MTCNN + FaceNet) loaded successfully")
except Exception as e:
    print(f"[ERROR] Failed to load face recognition models: {e}")
    mtcnn = None
    facenet = None

def get_embedding(img_bgr):
    try:
        if mtcnn is None or facenet is None:
            raise Exception("Face recognition models not loaded")
        
        # Convert BGR to RGB for MTCNN
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        print(f"[DEBUG] Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        
        # Detect face and get aligned face tensor
        face_tensor = mtcnn(img_rgb)
        
        if face_tensor is None:
            print("[DEBUG] No face detected by MTCNN")
            return None
        
        # Get embedding from FaceNet
        with torch.no_grad():
            embedding = facenet(face_tensor.unsqueeze(0)).cpu().numpy()
        
        # embedding shape: (1, 512) → (512,)
        embedding = embedding.squeeze(0)

        # Normalize (strongly recommended)
        embedding = embedding / np.linalg.norm(embedding)
        
        print(f"[DEBUG] Face embedding extracted successfully, shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        print(f"[ERROR] get_embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return None