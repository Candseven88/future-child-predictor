from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh

def align_face(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = np.array([landmarks[33].x, landmarks[33].y])
        right_eye = np.array([landmarks[263].x, landmarks[263].y])

        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return rotated

@app.post("/blend_faces/")
async def blend_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    contents1 = await file1.read()
    contents2 = await file2.read()

    nparr1 = np.frombuffer(contents1, np.uint8)
    nparr2 = np.frombuffer(contents2, np.uint8)

    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    img1_aligned = align_face(img1)
    img2_aligned = align_face(img2)

    if img1_aligned is None or img2_aligned is None:
        return {"error": "Could not detect faces in one or both images."}

    img1_resized = cv2.resize(img1_aligned, (300, 300))
    img2_resized = cv2.resize(img2_aligned, (300, 300))

    blended = cv2.addWeighted(img1_resized, 0.5, img2_resized, 0.5, 0)

    _, buffer = cv2.imencode('.jpg', blended)
    io_buf = BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/jpeg")
