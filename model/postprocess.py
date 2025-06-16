import cv2
import numpy as np


def sigmoid(x):
    """
    Applies the sigmoid function to the input.

    The sigmoid function is defined as 1 / (1 + exp(-x))

    Args:
        x (numpy.ndarray): The input array.

    Returns:
        numpy.ndarray: The array with the sigmoid function applied element-wise.
    """
    return 1 / (1 + np.exp(-x))

def postprocess(original_array, output):
    """
    Post-processes the output of a model to generate a binary mask.

    The function applies the sigmoid function to the model output to convert
    it into probabilities, resizes the probabilities to match the shape of the
    original image, and thresholds the resized probabilities to create a binary mask.

    Args:
        original_array (numpy.ndarray): The original image array.
        output (numpy.ndarray): The raw output from the model.

    Returns:
        numpy.ndarray: A binary mask where values greater than 0.5 are set to 1.
    """
    prob = sigmoid(output)
    mask_resized = cv2.resize(prob, original_array.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 0.5).astype(np.uint8)
