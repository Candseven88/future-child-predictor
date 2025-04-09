import cv2
import numpy as np

def crop_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return img  # 没找到脸就原图返回
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (300, 300))
    return face

def seamless_clone_face(face):
    background = np.full((500, 500, 3), 255, dtype=np.uint8)
    center = (250, 250)
    mask = 255 * np.ones(face.shape, face.dtype)
    output = cv2.seamlessClone(face, background, mask, center, cv2.NORMAL_CLONE)
    return output
