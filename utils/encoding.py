import cv2
import base64

def encode_frame_to_base64(frame):
    """
    Encodes a video frame to a Base64 string.

    The function converts the input frame into a JPEG image buffer and then encodes
    the buffer into a Base64 string for easy transmission.

    Args:
        frame (numpy.ndarray): The input video frame as a NumPy array.

    Returns:
        str: The Base64-encoded string representation of the JPEG image.
    """
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')
