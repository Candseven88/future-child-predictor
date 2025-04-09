from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io

from utils import extract_face, seamless_clone

app = FastAPI()

def read_image(file):
    """读取上传的图片"""
    image = np.frombuffer(file.file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

@app.post("/blend_faces/")
async def blend_faces(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = read_image(file1)
    img2 = read_image(file2)

    if img1 is None or img2 is None:
        return {"error": "无法读取上传的图片"}

    # 提取两张脸
    mask1, face1 = extract_face(img1)
    mask2, face2 = extract_face(img2)

    if face1 is None or face2 is None:
        return {"error": "未检测到人脸，请上传正面清晰的人像照片"}

    # 缩放第二张脸到第一张脸大小
    face2_resized = cv2.resize(face2, (face1.shape[1], face1.shape[0]))
    mask2_resized = cv2.resize(mask2, (face1.shape[1], face1.shape[0]))

    # 融合两张脸
    blended = cv2.addWeighted(face1, 0.5, face2_resized, 0.5, 0)

    # 把融合后的脸无缝克隆到白色背景
    background = np.full_like(face1, 255)
    output = seamless_clone(blended, mask1, background)

    # 编码成JPEG返回
    _, img_encoded = cv2.imencode('.jpg', output)
    return StreamingResponse(io.BytesIO(img_encoded.tobytes()), media_type="image/jpeg")
