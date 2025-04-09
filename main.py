import os
import urllib.request

# 检查模型文件是否存在，不存在就下载
model_path = "shape_predictor_68_face_landmarks.dat"
model_url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"

if not os.path.isfile(model_path):
    print("Model file not found, downloading...")
    compressed_path = model_path + ".bz2"
    urllib.request.urlretrieve(model_url, compressed_path)
    import bz2
    with bz2.BZ2File(compressed_path) as fr, open(model_path, "wb") as fw:
        fw.write(fr.read())
    os.remove(compressed_path)
    print("Download and decompress finished.")


from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import dlib
from io import BytesIO

app = FastAPI()

# 人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def get_landmarks(image):
    faces = detector(image, 1)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    return np.matrix([[p.x, p.y] for p in predictor(image, faces[0]).parts()])

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([
        np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R @ c1.T)),
        np.matrix([0., 0., 1.])
    ])

def warp_image(image, M, dshape):
    output = np.zeros(dshape, dtype=image.dtype)
    cv2.warpAffine(
        image,
        M[:2],
        (dshape[1], dshape[0]),
        dst=output,
        borderMode=cv2.BORDER_TRANSPARENT,
        flags=cv2.WARP_INVERSE_MAP
    )
    return output

def face_blend(img1, img2, alpha=0.5):
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)
    mean_landmarks = (1 - alpha) * landmarks1 + alpha * landmarks2
    M1 = transformation_from_points(mean_landmarks, landmarks1)
    M2 = transformation_from_points(mean_landmarks, landmarks2)
    warped_img1 = warp_image(img1, M1, img1.shape)
    warped_img2 = warp_image(img2, M2, img2.shape)
    output = cv2.addWeighted(warped_img1, 0.5, warped_img2, 0.5, 0)
    return output

@app.post("/blend_faces/")
async def blend_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    contents1 = await file1.read()
    contents2 = await file2.read()

    nparr1 = np.frombuffer(contents1, np.uint8)
    nparr2 = np.frombuffer(contents2, np.uint8)
    img1 = cv2.imdecode(nparr1, cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(nparr2, cv2.IMREAD_COLOR)

    if img1 is None or img2 is None:
        return {"error": "Invalid image upload."}

    try:
        result_img = face_blend(img1, img2)
    except Exception as e:
        return {"error": str(e)}

    _, encoded_img = cv2.imencode(".jpg", result_img)
    return StreamingResponse(BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
