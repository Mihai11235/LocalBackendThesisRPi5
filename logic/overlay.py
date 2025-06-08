import numpy as np
import cv2

def add_overlay(original, mask_bin):
    overlay = original.copy()
    overlay[mask_bin == 1] = overlay[mask_bin == 1] * 0.4 + np.array([0, 153, 0])
    return overlay.astype(np.uint8)
