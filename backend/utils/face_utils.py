import cv2
import torch
import numpy as np
from typing import Optional
from facenet_pytorch import MTCNN, InceptionResnetV1

# Initialize MediaPipe Face Mesh (OPTIONAL - only for liveness detection)
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("[INFO] MediaPipe Face Mesh loaded (liveness detection available)")
except Exception as e:
    print(f"[WARNING] MediaPipe not available (liveness detection disabled): {e}")
    mp_face_mesh = None
    face_mesh = None

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

# MediaPipe landmark indices (only used if liveness is enabled)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]
NOSE_TIP = 1
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


def get_embedding(img_bgr):
    """
    Extract face embedding from BGR image.
    This is the core function used for both registration and login.
    """
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
        
        print(f"[DEBUG] Face embedding extracted successfully, shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        print(f"[ERROR] get_embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# LIVENESS DETECTION FUNCTIONS (Only used when LIVENESS_ENABLED = True)
# ============================================================================

def eye_aspect_ratio(landmarks, eye_indices, w: int, h: int) -> float:
    """Calculate Eye Aspect Ratio (EAR) for blink detection."""
    if face_mesh is None:
        raise Exception("MediaPipe not available for liveness detection")
    
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    
    # Vertical eye distances
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    
    # Horizontal eye distance
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    
    # EAR calculation
    ear = (A + B) / (2.0 * C)
    return ear


def check_blink(frame: np.ndarray, threshold: float = 0.22):
    """Check if a blink is detected in the frame."""
    if face_mesh is None:
        return False, "MediaPipe not available"
    
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return False, "No face detected"
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Calculate EAR for both eyes
    ear_left = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
    ear_right = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
    ear_avg = (ear_left + ear_right) / 2.0
    
    print(f"[DEBUG] EAR: {ear_avg:.3f}, Threshold: {threshold}")
    
    if ear_avg < threshold:
        return True, f"Blink detected! (EAR: {ear_avg:.3f})"
    
    return False, f"Please blink (EAR: {ear_avg:.3f})"


def check_head_movement(
    frame: np.ndarray, 
    reference_position: Optional[int] = None,
    threshold: int = 40
):
    """Check if head movement (left/right) is detected."""
    if face_mesh is None:
        return False, "MediaPipe not available", None
    
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return False, "No face detected", None
    
    landmarks = results.multi_face_landmarks[0].landmark
    nose_x = int(landmarks[NOSE_TIP].x * w)
    
    if reference_position is None:
        return False, "Initializing reference position", nose_x
    
    movement = abs(nose_x - reference_position)
    print(f"[DEBUG] Head movement: {movement}px, Threshold: {threshold}px")
    
    if movement > threshold:
        return True, f"Head movement detected! ({movement}px)", nose_x
    
    return False, f"Please turn your head (moved {movement}px)", nose_x


def check_smile(
    frame: np.ndarray,
    width_ratio_threshold: float = 0.45,
    height_ratio_threshold: float = 0.03
):
    """Check if a smile is detected in the frame."""
    if face_mesh is None:
        return False, "MediaPipe not available"
    
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if not results.multi_face_landmarks:
        return False, "No face detected"
    
    landmarks = results.multi_face_landmarks[0].landmark
    
    # Get mouth landmarks
    left = np.array([landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h])
    right = np.array([landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h])
    top = np.array([landmarks[MOUTH_TOP].x * w, landmarks[MOUTH_TOP].y * h])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h])
    
    # Calculate mouth dimensions
    mouth_width = np.linalg.norm(left - right)
    mouth_height = np.linalg.norm(top - bottom)
    
    # Get face width for normalization (distance between eyes)
    left_eye_center = np.array([landmarks[33].x * w, landmarks[33].y * h])
    right_eye_center = np.array([landmarks[263].x * w, landmarks[263].y * h])
    face_width = np.linalg.norm(left_eye_center - right_eye_center)
    
    # Calculate ratios
    width_ratio = mouth_width / face_width
    height_ratio = mouth_height / face_width
    
    print(f"[DEBUG] Smile ratios - Width: {width_ratio:.3f}, Height: {height_ratio:.3f}")
    
    if width_ratio > width_ratio_threshold and height_ratio > height_ratio_threshold:
        return True, f"Smile detected! (W:{width_ratio:.2f}, H:{height_ratio:.2f})"
    
    return False, f"Please smile (W:{width_ratio:.2f}, H:{height_ratio:.2f})"


def analyze_liveness_frame(
    frame: np.ndarray,
    challenge_type: str,
    reference_data: Optional[dict] = None
) -> dict:
    """
    Analyze a single frame for liveness detection.
    Only called when LIVENESS_ENABLED = True.
    """
    try:
        if face_mesh is None:
            return {
                "success": False,
                "challenge_complete": False,
                "message": "MediaPipe not available - liveness detection disabled",
                "reference_data": None
            }
        
        if challenge_type == "blink":
            from config import LIVENESS_BLINK_THRESHOLD
            detected, message = check_blink(frame, LIVENESS_BLINK_THRESHOLD)
            return {
                "success": True,
                "challenge_complete": detected,
                "message": message,
                "reference_data": None
            }
        
        elif challenge_type == "head_movement":
            from config import LIVENESS_HEAD_MOVEMENT_THRESHOLD
            ref_pos = reference_data.get("nose_position") if reference_data else None
            detected, message, nose_pos = check_head_movement(
                frame, ref_pos, LIVENESS_HEAD_MOVEMENT_THRESHOLD
            )
            return {
                "success": True,
                "challenge_complete": detected,
                "message": message,
                "reference_data": {"nose_position": nose_pos}
            }
        
        elif challenge_type == "smile":
            from config import LIVENESS_SMILE_WIDTH_RATIO, LIVENESS_SMILE_HEIGHT_RATIO
            detected, message = check_smile(
                frame, LIVENESS_SMILE_WIDTH_RATIO, LIVENESS_SMILE_HEIGHT_RATIO
            )
            return {
                "success": True,
                "challenge_complete": detected,
                "message": message,
                "reference_data": None
            }
        
        else:
            return {
                "success": False,
                "challenge_complete": False,
                "message": f"Unknown challenge type: {challenge_type}",
                "reference_data": None
            }
    
    except Exception as e:
        print(f"[ERROR] Liveness analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "challenge_complete": False,
            "message": f"Analysis error: {str(e)}",
            "reference_data": None
        }


def get_random_liveness_challenges(count: int = 3):
    """Get a random sequence of liveness challenges."""
    import random
    challenges = ["blink", "head_movement", "smile"]
    random.shuffle(challenges)
    return challenges[:count]