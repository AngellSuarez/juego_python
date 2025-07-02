import cv2
import numpy as np

def preprocess_face(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48))
    normalized = resized / 255.0
    return normalized.reshape(48, 48, 1)
