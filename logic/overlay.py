import numpy as np

def add_overlay(original, mask_bin):
    """
    Adds an overlay to the original image based on a binary mask.

    The function modifies the original image by applying a green overlay to the areas
    where the binary mask has a value of 1. The overlay is blended with the original
    image at 40% intensity.

    Args:
        original (numpy.ndarray): The original image as a NumPy array.
        mask_bin (numpy.ndarray): A binary mask image where overlay areas are marked with 1s.

    Returns:
        numpy.ndarray: The modified image with the overlay applied, as a NumPy array.
    """
    overlay = original.copy()
    overlay[mask_bin == 1] = overlay[mask_bin == 1] * 0.4 + np.array([0, 153, 0])
    return overlay.astype(np.uint8)
