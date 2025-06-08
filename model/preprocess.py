import cv2
import numpy as np

def preprocess(image, height, width):
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    input_data = resized.astype(np.float32) / 255.0
    return input_data
