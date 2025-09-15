import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40)
facenet = InceptionResnetV1(pretrained='vggface2').eval()

def get_embedding(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    if face is None: return None
    with torch.no_grad():
        return facenet(face.unsqueeze(0)).cpu().numpy()