from picamera2 import Picamera2
from picamera2.utils import Transform

class Camera:
    """
    A class to interface with the Picamera2 library for capturing video frames.

    Attributes:
        cam (Picamera2): An instance of the Picamera2 class used to control the camera.
    """
    def __init__(self):
        """
            Initializes the Camera object, configures the camera for video capture, and starts the camera.

            The camera is configured with the following settings:
            - Resolution: 1280x720
            - Format: RGB888
            - Autofocus mode: Enabled (AfMode set to 2)
        """
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            transform=Transform(), controls={"AfMode": 2}
        ))
        self.cam.start()

    def capture_frame(self):
        """
            Captures a single frame from the camera.

            Returns:
                numpy.ndarray: The captured frame as a NumPy array.
        """
        return self.cam.capture_array()
