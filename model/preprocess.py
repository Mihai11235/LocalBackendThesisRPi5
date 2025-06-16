import cv2
import numpy as np

def preprocess(image, height, width):
    """
    Preprocesses an image for model input.

    The function resizes the input image to the specified dimensions and normalizes
    its pixel values to the range [0, 1].

    Args:
        image (numpy.ndarray): The input image as a NumPy array.
        height (int): The desired height of the resized image.
        width (int): The desired width of the resized image.

    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array with normalized pixel values.
    """
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    input_data = resized.astype(np.float32) / 255.0
    return input_data
