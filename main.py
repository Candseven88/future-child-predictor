from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import Delaunay
from io import BytesIO

app = FastAPI()

mp_face_mesh = mp.solutions.face_mesh

def get_landmarks(image):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        h, w, _ = image.shape
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.append((int(lm.x * w), int(lm.y * h)))
        return np.array(landmarks)

def apply_affine_transform(src, src_tri, dst_tri, size):
    if size[0] <= 0 or size[1] <= 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if src is None or src.shape[0] == 0 or src.shape[1] == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if len(src.shape) != 3 or src.shape[2] != 3:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

def morph_triangle(img1, img2, img, t1, t2, t, alpha=0.5):
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))
    
    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(3):
        t1_rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t_rect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))

    img1_rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2_rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]

    size = (r[2], r[3])
    warp_img1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_img2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    img_rect = (1.0 - alpha) * warp_img1 + alpha * warp_img2

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    img_part = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]]
    img_part = img_part * (1 - mask) + img_rect * mask
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_part

def morph_faces(img1, img2, alpha=0.5):
    points1 = get_landmarks(img1)
    points2 = get_landmarks(img2)

    if points1 is None or points2 is None:
        return None

    points = []
    for i in range(len(points1)):
        x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
        y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
        points.append((x, y))

    img1 = np.float32(img1)
    img2 = np.float32(img2)
    img_morph = np.zeros(img1.shape, dtype=img1.dtype)

    tri = Delaunay(points)

    for triangle in tri.simplices:
        x, y, z = triangle
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        morph_triangle(img1, img2, img_morph, t1, t2, t, alpha)

    img_morph = np.clip(img_morph, 0, 255).astype(np.uint8)
    return img_morph

@app.post("/blend_faces/")
async def blend_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    contents1 = await file1.read()
    contents2 = await file2.read()

    nparr1 = np.frombuffer(contents1, np.uint8)
    nparr2 = np.frombuffer(contents2, np.uint8)

    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    img1 = cv2.resize(img1, (300, 300))
    img2 = cv2.resize(img2, (300, 300))

    result = morph_faces(img1, img2)

    if result is None:
        return {"error": "Could not detect faces in one or both images."}

    _, buffer = cv2.imencode('.jpg', result)
    io_buf = BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/jpeg")
