import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def extract_face(image):
    """提取图片中的人脸并返回mask和裁剪后的脸部图像"""
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None, None

        h, w, _ = image.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for face_landmarks in results.multi_face_landmarks:
            points = []
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                points.append((x, y))
            points = np.array(points, np.int32)
            cv2.fillConvexPoly(mask, points, 255)

        face = cv2.bitwise_and(image, image, mask=mask)
        return mask, face

def seamless_clone(src_face, src_mask, dst_img):
    """将src_face根据mask无缝融合到dst_img"""
    center = (dst_img.shape[1] // 2, dst_img.shape[0] // 2)
    output = cv2.seamlessClone(src_face, dst_img, src_mask, center, cv2.NORMAL_CLONE)
    return output
