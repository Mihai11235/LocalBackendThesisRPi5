import cv2
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def postprocess(original_array, output):
    prob = sigmoid(output)
    # mask = np.squeeze(prob)
    mask_resized = cv2.resize(prob, original_array.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 0.5).astype(np.uint8)
